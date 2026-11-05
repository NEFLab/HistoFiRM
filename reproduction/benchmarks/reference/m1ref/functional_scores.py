"""Frozen functional scoring routines used by the benchmark adapters."""
from __future__ import annotations

def normalize_symbol(value: Any) -> str:
    return str(value).strip().upper()

def parse_gmt(path: Path, source: str) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
    gene_sets: Dict[str, List[str]] = {}
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise PipelineError(f"Invalid GMT line {line_no} in {path}")
            term, description, *genes_raw = parts
            term = term.strip()
            if not term:
                raise PipelineError(f"Empty GMT term at {path}:{line_no}")
            function_id = f"{source}::{term}"
            if function_id in gene_sets:
                raise PipelineError(f"Duplicate GMT term: {function_id}")
            genes = sorted({normalize_symbol(g) for g in genes_raw if str(g).strip()})
            if not genes:
                raise PipelineError(f"Empty GMT membership: {function_id}")
            gene_sets[function_id] = genes
            rows.append({
                "function_id": function_id,
                "source": source,
                "original_term": term,
                "description": description,
                "score_type": "ordinary_ssgsea",
                "signed": False,
                "original_gene_count": len(genes_raw),
                "unique_gene_count": len(genes),
                "source_path": str(path),
            })
    return gene_sets, rows

def _gseapy_ssgsea(expr: Any, gene_sets: Mapping[str, Sequence[str]], scoring: Mapping[str, Any]) -> Tuple[Any, List[str]]:
    import gseapy
    import pandas as pd

    signature = inspect.signature(gseapy.ssgsea)
    kwargs: Dict[str, Any] = {
        "data": expr,
        "gene_sets": dict(gene_sets),
        "outdir": None,
        "sample_norm_method": scoring["sample_norm_method"],
        "correl_norm_type": scoring["correl_norm_type"],
        "weight": float(scoring["weight"]),
        "min_size": int(scoring["score_all_min_mapped_genes"]),
        "max_size": int(scoring["score_all_max_mapped_genes"]),
        "permutation_num": int(scoring["permutation_num"]),
        "ascending": bool(scoring["ascending"]),
        "threads": int(scoring["threads_per_scoring_process"]),
        "seed": int(scoring["seed"]),
        "no_plot": True,
        "verbose": False,
    }
    mandatory = {
        "data", "gene_sets", "outdir", "sample_norm_method", "correl_norm_type", "weight",
        "min_size", "max_size", "permutation_num", "ascending", "threads", "seed", "no_plot",
    }
    unsupported = [key for key in mandatory if key not in signature.parameters]
    if unsupported:
        raise PipelineError(f"Installed gseapy.ssgsea API lacks required parameters: {unsupported}")
    kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
    result = gseapy.ssgsea(**kwargs)
    if not hasattr(result, "res2d"):
        raise PipelineError("gseapy.ssgsea result lacks res2d")
    res = result.res2d.copy()
    required_cols = {"Name", "Term", "ES"}
    if not required_cols.issubset(res.columns):
        raise PipelineError(f"gseapy res2d missing columns {required_cols - set(res.columns)}; got={list(res.columns)}")
    res["Name"] = res["Name"].astype(str)
    res["Term"] = res["Term"].astype(str)
    res["ES"] = pd.to_numeric(res["ES"], errors="coerce")
    if res["ES"].isna().any():
        raise PipelineError("gseapy raw ES contains non-numeric/NA values")
    pivot = res.pivot(index="Name", columns="Term", values="ES")
    pivot.index.name = "sample_id"
    return pivot, list(res.columns)

def score_ssgsea_batched(expr: Any, gene_sets: Mapping[str, Sequence[str]], scoring: Mapping[str, Any]) -> Any:
    import pandas as pd

    sample_ids = list(expr.columns.astype(str))
    terms = sorted(gene_sets)
    sample_chunk = int(scoring["sample_chunk_size"])
    term_batch = int(scoring["term_batch_size"])
    term_frames: List[Any] = []
    for t0 in range(0, len(terms), term_batch):
        batch_terms = terms[t0:t0 + term_batch]
        batch_sets = {term: gene_sets[term] for term in batch_terms}
        sample_frames: List[Any] = []
        for s0 in range(0, len(sample_ids), sample_chunk):
            chunk_ids = sample_ids[s0:s0 + sample_chunk]
            scored, _ = _gseapy_ssgsea(expr.loc[:, chunk_ids], batch_sets, scoring)
            scored = scored.reindex(index=chunk_ids, columns=batch_terms)
            if scored.isna().any().any():
                missing = scored.columns[scored.isna().any()].tolist()[:10]
                raise PipelineError(f"Missing ssGSEA scores after alignment: {missing}")
            sample_frames.append(scored)
        term_frame = pd.concat(sample_frames, axis=0).reindex(sample_ids)
        term_frames.append(term_frame)
    final = pd.concat(term_frames, axis=1).reindex(index=sample_ids, columns=terms)
    if final.isna().any().any():
        raise PipelineError("Final ssGSEA matrix contains NA")
    final.index.name = "sample_id"
    return final

def score_progeny(expr: Any, progeny: Any) -> Tuple[Any, Any]:
    import numpy as np
    import pandas as pd

    expr_genes = set(expr.index)
    score_columns: Dict[str, Any] = {}
    audit_rows: List[Dict[str, Any]] = []
    for pathway, group in progeny.groupby("pathway", sort=True):
        group = group.copy()
        total_abs = float(group["weight"].abs().sum())
        mapped = group[group["gene"].isin(expr_genes)].copy()
        mapped_abs = float(mapped["weight"].abs().sum())
        if mapped.empty or mapped_abs <= 0:
            continue
        values = expr.loc[mapped["gene"].tolist()].to_numpy(dtype=float)
        weights = mapped["weight"].to_numpy(dtype=float)
        score = (weights[:, None] * values).sum(axis=0) / mapped_abs
        function_id = f"PROGENY::{pathway}"
        score_columns[function_id] = score
        audit_rows.append({
            "function_id": function_id,
            "original_gene_count": int(group["gene"].nunique()),
            "mapped_gene_count": int(mapped["gene"].nunique()),
            "mapping_coverage": float(mapped["gene"].nunique() / max(group["gene"].nunique(), 1)),
            "absolute_weight_sum": total_abs,
            "mapped_absolute_weight_sum": mapped_abs,
            "absolute_weight_coverage": float(mapped_abs / total_abs) if total_abs > 0 else np.nan,
        })
    result = pd.DataFrame(score_columns, index=expr.columns)
    result.index.name = "sample_id"
    if result.empty or not np.isfinite(result.to_numpy()).all():
        raise PipelineError("PROGENy scoring failed or produced non-finite values")
    return result, pd.DataFrame(audit_rows)

def filter_mapped_sets(expr: Any, gene_sets: Mapping[str, Sequence[str]], min_genes: int, max_genes: int) -> Tuple[Dict[str, List[str]], Any]:
    import pandas as pd

    expr_genes = set(expr.index)
    mapped: Dict[str, List[str]] = {}
    rows: List[Dict[str, Any]] = []
    for term in sorted(gene_sets):
        original = list(gene_sets[term])
        genes = [g for g in original if g in expr_genes]
        rows.append({
            "function_id": term,
            "original_gene_count": len(original),
            "mapped_gene_count": len(genes),
            "mapping_coverage": len(genes) / max(len(original), 1),
            "scoreable_for_full_audit": min_genes <= len(genes) <= max_genes,
        })
        if min_genes <= len(genes) <= max_genes:
            mapped[term] = genes
    return mapped, pd.DataFrame(rows)

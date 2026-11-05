"""RNA-derived functional scores on a fixed, explicitly supplied target axis."""

from __future__ import annotations
import inspect
from typing import Any, Mapping, Sequence, Tuple, Dict, List
import numpy as np
import pandas as pd


class PipelineError(ValueError):
    pass


def _gseapy_ssgsea(
    expr: Any, gene_sets: Mapping[str, Sequence[str]], scoring: Mapping[str, Any]
) -> Tuple[Any, List[str]]:
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
        "data",
        "gene_sets",
        "outdir",
        "sample_norm_method",
        "correl_norm_type",
        "weight",
        "min_size",
        "max_size",
        "permutation_num",
        "ascending",
        "threads",
        "seed",
        "no_plot",
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
        raise PipelineError(
            f"gseapy res2d missing columns {required_cols - set(res.columns)}; got={list(res.columns)}"
        )
    res["Name"] = res["Name"].astype(str)
    res["Term"] = res["Term"].astype(str)
    res["ES"] = pd.to_numeric(res["ES"], errors="coerce")
    if res["ES"].isna().any():
        raise PipelineError("gseapy raw ES contains non-numeric/NA values")
    pivot = res.pivot(index="Name", columns="Term", values="ES")
    pivot.index.name = "sample_id"
    return pivot, list(res.columns)


def score_ssgsea_batched(
    expr: Any, gene_sets: Mapping[str, Sequence[str]], scoring: Mapping[str, Any]
) -> Any:
    import pandas as pd

    sample_ids = list(expr.columns.astype(str))
    terms = sorted(gene_sets)
    sample_chunk = int(scoring["sample_chunk_size"])
    term_batch = int(scoring["term_batch_size"])
    term_frames: List[Any] = []
    for t0 in range(0, len(terms), term_batch):
        batch_terms = terms[t0 : t0 + term_batch]
        batch_sets = {term: gene_sets[term] for term in batch_terms}
        sample_frames: List[Any] = []
        for s0 in range(0, len(sample_ids), sample_chunk):
            chunk_ids = sample_ids[s0 : s0 + sample_chunk]
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
        audit_rows.append(
            {
                "function_id": function_id,
                "original_gene_count": int(group["gene"].nunique()),
                "mapped_gene_count": int(mapped["gene"].nunique()),
                "mapping_coverage": float(
                    mapped["gene"].nunique() / max(group["gene"].nunique(), 1)
                ),
                "absolute_weight_sum": total_abs,
                "mapped_absolute_weight_sum": mapped_abs,
                "absolute_weight_coverage": float(mapped_abs / total_abs)
                if total_abs > 0
                else np.nan,
            }
        )
    result = pd.DataFrame(score_columns, index=expr.columns)
    result.index.name = "sample_id"
    if result.empty or not np.isfinite(result.to_numpy()).all():
        raise PipelineError("PROGENy scoring failed or produced non-finite values")
    return result, pd.DataFrame(audit_rows)


def filter_mapped_sets(
    expr: Any, gene_sets: Mapping[str, Sequence[str]], min_genes: int, max_genes: int
) -> Tuple[Dict[str, List[str]], Any]:
    import pandas as pd

    expr_genes = set(expr.index)
    mapped: Dict[str, List[str]] = {}
    rows: List[Dict[str, Any]] = []
    for term in sorted(gene_sets):
        original = list(gene_sets[term])
        genes = [g for g in original if g in expr_genes]
        rows.append(
            {
                "function_id": term,
                "original_gene_count": len(original),
                "mapped_gene_count": len(genes),
                "mapping_coverage": len(genes) / max(len(original), 1),
                "scoreable_for_full_audit": min_genes <= len(genes) <= max_genes,
            }
        )
        if min_genes <= len(genes) <= max_genes:
            mapped[term] = genes
    return mapped, pd.DataFrame(rows)


def log_expression(tpm):
    """Input: HGNC-symbol-indexed genes x samples, in nonnegative linear TPM."""
    if not np.isfinite(tpm.to_numpy(float)).all() or (tpm.to_numpy(float) < 0).any():
        raise ValueError("Linear TPM must be finite and nonnegative.")
    if tpm.columns.duplicated().any():
        raise ValueError("Duplicate sample IDs.")
    a = tpm.copy()
    a.index = a.index.astype(str).str.strip().str.upper()
    if (a.index == "").any():
        raise ValueError("Empty gene symbol.")
    return np.log2(a.groupby(level=0, sort=True).sum() + 1.0)


def read_gmt(filename, prefix):
    sets = {}
    with open(filename, encoding="utf-8") as handle:
        for line in handle:
            cells = line.rstrip().split("\t")
            if len(cells) < 3:
                raise ValueError("Malformed GMT row.")
            key = prefix + "::" + cells[0]
            if key in sets:
                raise ValueError("Duplicate gene-set ID.")
            sets[key] = sorted(set(g.strip().upper() for g in cells[2:] if g.strip()))
    return sets


def score_fixed_axis(tpm, gene_sets, progeny, target_ids, scoring):
    """Missing/unscoreable targets remain NaN; IDs and order are never replaced."""
    import gseapy

    if gseapy.__version__ != scoring["required_gseapy_version"]:
        raise ValueError("Use the specified GSEApy version for the scoring contract.")
    expr = log_expression(tpm)
    ordinary = {k: v for k, v in gene_sets.items() if k in set(target_ids)}
    mapped, _ = filter_mapped_sets(
        expr, ordinary, scoring["score_all_min_mapped_genes"], scoring["score_all_max_mapped_genes"]
    )
    frames = []
    if mapped:
        frames.append(score_ssgsea_batched(expr, mapped, scoring))
    if progeny is not None:
        if not {"pathway", "gene", "weight"} <= set(progeny.columns):
            raise ValueError("PROGENy requires pathway, gene and weight columns.")
        if (
            progeny.duplicated(["pathway", "gene"]).any()
            or not np.isfinite(progeny.weight.to_numpy(float)).all()
        ):
            raise ValueError("Invalid or duplicated PROGENy weights.")
        frames.append(score_progeny(expr, progeny)[0])
    if not frames:
        return pd.DataFrame(np.nan, index=expr.columns, columns=target_ids)
    return pd.concat(frames, axis=1).reindex(index=expr.columns, columns=target_ids)


def target_eligibility(qc):
    """Apply the locked RNA-only per-cancer criteria to previously computed QC."""
    ordinary = qc.mapped_gene_count.between(15, 500) & qc.mapping_coverage.ge(0.5)
    progeny = qc.mapped_gene_count.ge(100) & qc.absolute_weight_coverage.ge(0.8)
    common = (
        qc.finite_fraction.ge(0.999)
        & qc.unique_value_count.ge(20)
        & qc.dominant_value_fraction.le(0.95)
        & qc.median_spearman.ge(0.8)
        & qc.p10_spearman.ge(0.6)
    )
    is_progeny = qc.function_id.str.startswith("PROGENY::")
    passed = common & np.where(is_progeny, progeny, ordinary)
    table = qc[["function_id", "cancer"]].copy()
    table["eligible"] = passed
    if table.duplicated(["function_id", "cancer"]).any():
        raise ValueError("Duplicated QC target/cancer pairs.")
    counts = table.groupby("function_id").eligible.sum()
    return table, counts.ge(12)

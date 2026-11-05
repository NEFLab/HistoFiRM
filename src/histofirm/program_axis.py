"""Exact existing Hallmark axis only. No data-driven program or target selection."""


def program_indices(target_ids, expected=50):
    ix = [
        i
        for i, t in enumerate(target_ids)
        if str(t).startswith("HALLMARK::") or str(t).startswith("HALLMARK_")
    ]
    if len(ix) != int(expected) or len(set(target_ids)) != len(target_ids):
        raise ValueError("EXACT_EXISTING_HALLMARK_AXIS_REQUIRED")
    return ix

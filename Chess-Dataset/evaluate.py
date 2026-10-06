from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

K_LIST: tuple[int, ...] = (1, 5, 10)
SELECT_METRIC = "NDCG@10"   # declared before any hyper-parameter search; used for model selection
TRUNCATE_MRR = True         # MRR is reported as MRR@max(K_LIST); see note in compute_ranking_metrics


def compute_ranking_metrics(
    preds_list: Sequence[Sequence[int]],
    targets: Sequence[int],
    k_list: Iterable[int] = K_LIST,
    truncate_mrr: bool = TRUNCATE_MRR,
) -> dict[str, float]:
    if len(preds_list) != len(targets):
        raise ValueError(f"{len(preds_list)} prediction lists for {len(targets)} targets")

    k_list = tuple(sorted(k_list))
    n = len(targets)
    if n == 0:
        return {}

    max_k = max(k_list)
    hits = {k: 0 for k in k_list}
    ndcgs = {k: 0.0 for k in k_list}
    rr_sum = 0.0

    for preds, target in zip(preds_list, targets):
        try:
            rank = list(preds).index(target) + 1
        except ValueError:
            continue                       # target absent from the returned list: counts as a miss
        if truncate_mrr and rank > max_k:
            continue
        rr_sum += 1.0 / rank
        for k in k_list:
            if rank <= k:
                hits[k] += 1
                ndcgs[k] += 1.0 / math.log2(rank + 1)

    results: dict[str, float] = {}
    for k in k_list:
        results[f"HR@{k}"] = hits[k] / n
        if k > 1:
            results[f"NDCG@{k}"] = ndcgs[k] / n
    results["MRR"] = rr_sum / n
    return results


def topk_from_scores(scores, k: int = max(K_LIST), masked: Iterable[int] = (0,)):
    import numpy as np

    s = np.asarray(scores, dtype=np.float64)
    masked = list(masked)
    if masked:
        s = s.copy()
        s[:, masked] = -np.inf
    if k >= s.shape[1]:
        return np.argsort(-s, axis=1)[:, :k]

    idx = np.argpartition(-s, k, axis=1)[:, :k]                 # k best, unordered
    order = np.argsort(-np.take_along_axis(s, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)               # ...now ordered

# derived comparison between 2 models
def relative_gain(model: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:

    return {name: (value / baseline[name]) - 1.0
            for name, value in model.items() if baseline.get(name)}

# helpers
# formatting
def format_metrics(model_name: str, metrics: dict[str, float], n_eval: int | None = None) -> str:
    """One block per model, in a fixed order so the two projects' logs line up."""
    header = f"{'=' * 20} {model_name} {'=' * 20}"
    body = [
        f"  {name:<10}: {value * 100:.2f}%" if name.startswith(("HR", "NDCG")) else
        f"  {name:<10}: {value:.4f}"
        for name, value in metrics.items()
    ]
    if n_eval is not None:
        body.append(f"  {'n':<10}: {n_eval:,} evaluation points")
    return "\n".join([header, *body])

# formatting
def print_metrics(model_name: str, metrics: dict[str, float], n_eval: int | None = None) -> None:
    print(format_metrics(model_name, metrics, n_eval))


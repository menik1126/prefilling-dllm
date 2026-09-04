#!/usr/bin/env python3
"""Compare MultifieldQA chunk-selection artifacts with a reference run.

The comparison is intentionally independent from the evaluator and serving
code.  It accepts the line-delimited JSON emitted by the SGLang evaluator and
the Fast-dLLM reference JSON, then checks structural parity, per-chunk scores,
and the selected top-k chunk set.  An optional second current run checks that
selector batching does not change drafts, scores, or selections.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_CRITICAL_INDICES = (3, 82, 94, 95, 99, 101, 102)
STRUCTURAL_FIELDS = (
    "example_id",
    "candidate_chunks",
    "raw_context_tokens",
    "prefix_tokens",
    "query_tokens",
)


@dataclass(frozen=True)
class GateConfig:
    top_k: int = 4
    min_mean_spearman: float = 0.99
    strict_max_regret: float = 1e-8
    practical_max_mismatches: int = 1
    practical_max_margin: float = 0.002
    practical_max_regret: float = 0.002
    batch_score_atol: float = 1e-6
    draft_slots: int = 4
    expected_confirmed_count: int = 2
    mask_token_id: int | None = 151666
    critical_indices: tuple[int, ...] = DEFAULT_CRITICAL_INDICES
    require_partial_draft: bool = False
    expected_examples: int | None = None
    expected_active_examples: int | None = None
    expected_active_chunks: int | None = None


def _records_from_json_value(value: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict) and isinstance(value.get("records"), list):
        records = value["records"]
    elif isinstance(value, dict) and "index" in value:
        records = [value]
    else:
        raise ValueError(f"{path} does not contain indexed records")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path} contains a non-object record")
    return records


def load_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load records from JSON arrays/objects or line-delimited JSON files."""

    records: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON in {path}:{line_number}: {error}"
                    ) from error
                records.extend(_records_from_json_value(value, path))
        else:
            records.extend(_records_from_json_value(value, path))
    return records


def index_records(
    records: Iterable[dict[str, Any]], label: str
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        if "index" not in record:
            raise ValueError(f"{label} contains a record without index")
        index = int(record["index"])
        if index in indexed:
            raise ValueError(f"{label} contains duplicate index {index}")
        indexed[index] = record
    return indexed


def selection_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("parallelcomp")
    return payload if isinstance(payload, dict) else record


def selected_indices(record: dict[str, Any]) -> list[int]:
    values = selection_payload(record).get("selected_chunk_indices", [])
    return [int(value) for value in values]


def score_map(record: dict[str, Any]) -> dict[int, float] | None:
    values = selection_payload(record).get("chunk_scores")
    if values is None:
        return None
    if isinstance(values, list):
        return {index: float(value) for index, value in enumerate(values)}
    if isinstance(values, dict):
        return {int(index): float(value) for index, value in values.items()}
    raise ValueError(f"index {record.get('index')} has invalid chunk_scores")


def candidate_count(record: dict[str, Any]) -> int:
    return int(selection_payload(record).get("candidate_chunks", 0))


def structural_value(record: dict[str, Any], field: str) -> Any:
    if field == "example_id":
        return record.get(field)
    return selection_payload(record).get(field)


def partial_draft(record: dict[str, Any]) -> tuple[list[int] | None, list[bool] | None]:
    """Return explicit partial-draft fields without treating legacy drafts as equal."""

    payload = selection_payload(record)
    ids = record.get("partial_draft_ids", payload.get("partial_draft_ids"))
    mask = record.get("draft_confirmed_mask", payload.get("draft_confirmed_mask"))
    if mask is None:
        mask = record.get("dllm_confirmed_mask", payload.get("dllm_confirmed_mask"))
    normalized_ids = None if ids is None else [int(value) for value in ids]
    normalized_mask = None if mask is None else [bool(value) for value in mask]
    return normalized_ids, normalized_mask


def ranked_indices(scores: dict[int, float]) -> list[int]:
    # Explicit chunk-index tie breaking matches stable sorting over document order.
    return sorted(scores, key=lambda index: (-scores[index], index))


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def pearson(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b):
        raise ValueError("correlation inputs have different lengths")
    if not values_a:
        return 1.0
    mean_a = statistics.fmean(values_a)
    mean_b = statistics.fmean(values_b)
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    norm_a = sum(value * value for value in centered_a)
    norm_b = sum(value * value for value in centered_b)
    if norm_a == 0 or norm_b == 0:
        return 1.0 if centered_a == centered_b else 0.0
    return sum(a * b for a, b in zip(centered_a, centered_b)) / math.sqrt(
        norm_a * norm_b
    )


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    return pearson(_rankdata(values_a), _rankdata(values_b))


def quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(
    values: Sequence[float], *, include_rmse: bool = False
) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    result: dict[str, Any] = {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": quantile(values, 0.50),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }
    if include_rmse:
        result["rmse"] = math.sqrt(statistics.fmean(value * value for value in values))
    return result


def _valid_selected(values: Sequence[int], count: int) -> bool:
    return (
        list(values) == sorted(values)
        and len(values) == len(set(values))
        and all(0 <= value < count for value in values)
    )


def _draft_issues(
    ids: list[int] | None,
    mask: list[bool] | None,
    config: GateConfig,
) -> list[str]:
    if ids is None and mask is None:
        return ["missing_partial_draft"]
    if ids is None or mask is None:
        return ["incomplete_partial_draft"]
    issues = []
    if len(ids) != config.draft_slots:
        issues.append("draft_slot_count")
    if len(mask) != config.draft_slots:
        issues.append("draft_mask_count")
    if len(mask) == config.draft_slots and sum(mask) != config.expected_confirmed_count:
        issues.append("confirmed_count")
    if len(ids) == len(mask) and config.mask_token_id is not None:
        if any(
            not confirmed and token_id != config.mask_token_id
            for token_id, confirmed in zip(ids, mask)
        ):
            issues.append("unconfirmed_not_mask_token")
        if any(
            confirmed and token_id == config.mask_token_id
            for token_id, confirmed in zip(ids, mask)
        ):
            issues.append("confirmed_is_mask_token")
    return issues


def compare_batch_peer(
    current: dict[int, dict[str, Any]],
    peer: dict[int, dict[str, Any]],
    config: GateConfig,
) -> dict[str, Any]:
    current_indices = set(current)
    peer_indices = set(peer)
    selected_mismatches: list[int] = []
    draft_mismatches: list[int] = []
    mask_mismatches: list[int] = []
    score_mismatches: list[int] = []
    structural_mismatches: list[int] = []
    score_abs_errors: list[float] = []

    for index in sorted(current_indices & peer_indices):
        left = current[index]
        right = peer[index]
        if any(
            structural_value(left, field) != structural_value(right, field)
            for field in STRUCTURAL_FIELDS
        ):
            structural_mismatches.append(index)
        if selected_indices(left) != selected_indices(right):
            selected_mismatches.append(index)
        left_ids, left_mask = partial_draft(left)
        right_ids, right_mask = partial_draft(right)
        if left_ids != right_ids:
            draft_mismatches.append(index)
        if left_mask != right_mask:
            mask_mismatches.append(index)
        left_scores = score_map(left)
        right_scores = score_map(right)
        if left_scores is None and right_scores is None:
            continue
        if (
            left_scores is None
            or right_scores is None
            or set(left_scores) != set(right_scores)
        ):
            score_mismatches.append(index)
            continue
        local_errors = [
            abs(left_scores[chunk] - right_scores[chunk]) for chunk in left_scores
        ]
        score_abs_errors.extend(local_errors)
        if local_errors and max(local_errors) > config.batch_score_atol:
            score_mismatches.append(index)

    missing = sorted(current_indices - peer_indices)
    extra = sorted(peer_indices - current_indices)
    passed = not any(
        (
            missing,
            extra,
            structural_mismatches,
            selected_mismatches,
            draft_mismatches,
            mask_mismatches,
            score_mismatches,
        )
    )
    return {
        "passed": passed,
        "missing_indices": missing,
        "extra_indices": extra,
        "structural_mismatch_indices": structural_mismatches,
        "selection_mismatch_indices": selected_mismatches,
        "draft_mismatch_indices": draft_mismatches,
        "mask_mismatch_indices": mask_mismatches,
        "score_mismatch_indices": score_mismatches,
        "score_atol": config.batch_score_atol,
        "score_abs_error": distribution(score_abs_errors),
    }


def compare_selection(
    reference_records: Sequence[dict[str, Any]],
    current_records: Sequence[dict[str, Any]],
    config: GateConfig,
    *,
    peer_records: Sequence[dict[str, Any]] | None = None,
    golden_records: Sequence[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference = index_records(reference_records, "reference")
    current = index_records(current_records, "current")
    golden = (
        index_records(golden_records, "golden") if golden_records is not None else {}
    )
    peer = (
        index_records(peer_records, "batch peer") if peer_records is not None else None
    )

    reference_indices = set(reference)
    current_indices = set(current)
    missing_indices = sorted(reference_indices - current_indices)
    extra_indices = sorted(current_indices - reference_indices)
    structural_errors: list[dict[str, Any]] = []
    per_example: list[dict[str, Any]] = []
    raw_abs_errors: list[float] = []
    centered_abs_errors: list[float] = []
    sample_pearsons: list[float] = []
    sample_spearmans: list[float] = []
    regrets: list[float] = []
    top4_mismatches: list[int] = []
    ranked_top4_mismatches: list[int] = []
    top1_mismatches: list[int] = []
    critical_mismatches: list[int] = []
    overlap_histogram: dict[str, int] = {}
    active_count = 0
    active_chunk_count = 0
    score_comparable_count = 0
    partial_present = 0
    partial_valid = 0
    partial_missing_indices: list[int] = []
    partial_invalid: list[dict[str, Any]] = []
    golden_missing_indices = sorted(set(golden) - current_indices)
    golden_draft_mismatches: list[int] = []
    golden_mask_mismatches: list[int] = []

    for index in sorted(reference_indices & current_indices):
        reference_record = reference[index]
        current_record = current[index]
        count = candidate_count(reference_record)
        current_count = candidate_count(current_record)
        issues: list[str] = []
        for field in STRUCTURAL_FIELDS:
            reference_value = structural_value(reference_record, field)
            current_value = structural_value(current_record, field)
            if reference_value != current_value:
                issues.append(f"{field}_mismatch")
                structural_errors.append(
                    {
                        "index": index,
                        "kind": f"{field}_mismatch",
                        "reference": reference_value,
                        "current": current_value,
                    }
                )

        reference_selected = selected_indices(reference_record)
        current_selected = selected_indices(current_record)
        if not _valid_selected(reference_selected, count):
            issues.append("invalid_reference_selection")
            structural_errors.append(
                {"index": index, "kind": "invalid_reference_selection"}
            )
        if not _valid_selected(current_selected, current_count):
            issues.append("invalid_current_selection")
            structural_errors.append(
                {"index": index, "kind": "invalid_current_selection"}
            )

        is_active = count > config.top_k
        if is_active:
            active_count += 1
            active_chunk_count += count

        reference_scores = score_map(reference_record)
        current_scores = score_map(current_record)
        reference_ranked: list[int] | None = None
        current_ranked: list[int] | None = None
        score_deltas: list[float] | None = None
        score_abs: list[float] | None = None
        centered_abs: list[float] | None = None
        sample_pearson: float | None = None
        sample_spearman: float | None = None
        reference_margin: float | None = None
        regret: float | None = None

        expected_score_keys = set(range(count))
        if is_active:
            if reference_scores is None or set(reference_scores) != expected_score_keys:
                issues.append("invalid_reference_scores")
                structural_errors.append(
                    {"index": index, "kind": "invalid_reference_scores"}
                )
            if current_scores is None or set(current_scores) != expected_score_keys:
                issues.append("invalid_current_scores")
                structural_errors.append(
                    {"index": index, "kind": "invalid_current_scores"}
                )

        if (
            is_active
            and reference_scores is not None
            and current_scores is not None
            and set(reference_scores) == expected_score_keys
            and set(current_scores) == expected_score_keys
        ):
            reference_values = [reference_scores[chunk] for chunk in range(count)]
            current_values = [current_scores[chunk] for chunk in range(count)]
            if not all(
                math.isfinite(value) for value in reference_values + current_values
            ):
                issues.append("non_finite_score")
                structural_errors.append({"index": index, "kind": "non_finite_score"})
            else:
                score_comparable_count += count
                score_deltas = [
                    current_value - reference_value
                    for current_value, reference_value in zip(
                        current_values, reference_values
                    )
                ]
                score_abs = [abs(value) for value in score_deltas]
                reference_mean = statistics.fmean(reference_values)
                current_mean = statistics.fmean(current_values)
                centered_abs = [
                    abs(
                        (current_value - current_mean)
                        - (reference_value - reference_mean)
                    )
                    for current_value, reference_value in zip(
                        current_values, reference_values
                    )
                ]
                raw_abs_errors.extend(score_abs)
                centered_abs_errors.extend(centered_abs)
                sample_pearson = pearson(current_values, reference_values)
                sample_spearman = spearman(current_values, reference_values)
                sample_pearsons.append(sample_pearson)
                sample_spearmans.append(sample_spearman)
                reference_ranked = ranked_indices(reference_scores)
                current_ranked = ranked_indices(current_scores)
                reference_top = reference_ranked[: config.top_k]
                current_top = current_ranked[: config.top_k]
                derived_reference_selection = sorted(reference_top)
                derived_current_selection = sorted(current_top)
                if reference_selected != derived_reference_selection:
                    issues.append("reference_selection_not_score_topk")
                    structural_errors.append(
                        {
                            "index": index,
                            "kind": "reference_selection_not_score_topk",
                        }
                    )
                if current_selected != derived_current_selection:
                    issues.append("current_selection_not_score_topk")
                    structural_errors.append(
                        {
                            "index": index,
                            "kind": "current_selection_not_score_topk",
                        }
                    )
                reference_margin = (
                    reference_scores[reference_ranked[config.top_k - 1]]
                    - reference_scores[reference_ranked[config.top_k]]
                )
                if len(current_selected) == config.top_k:
                    raw_regret = sum(
                        reference_scores[chunk] for chunk in reference_top
                    ) - sum(reference_scores[chunk] for chunk in current_selected)
                    regret = max(0.0, raw_regret)
                    regrets.append(regret)
                if reference_top != current_top:
                    ranked_top4_mismatches.append(index)
                if reference_top[0] != current_top[0]:
                    top1_mismatches.append(index)

        selection_matches = reference_selected == current_selected
        overlap = len(set(reference_selected) & set(current_selected))
        overlap_histogram[str(overlap)] = overlap_histogram.get(str(overlap), 0) + 1
        if is_active and not selection_matches:
            top4_mismatches.append(index)
            if index in config.critical_indices:
                critical_mismatches.append(index)

        draft_ids, draft_mask = partial_draft(current_record)
        draft_issues = _draft_issues(draft_ids, draft_mask, config)
        if draft_ids is not None or draft_mask is not None:
            partial_present += 1
        if not draft_issues:
            partial_valid += 1
        elif is_active:
            if draft_issues == ["missing_partial_draft"]:
                partial_missing_indices.append(index)
            else:
                partial_invalid.append({"index": index, "issues": draft_issues})

        if index in golden:
            golden_ids, golden_mask = partial_draft(golden[index])
            if draft_ids != golden_ids:
                golden_draft_mismatches.append(index)
            if draft_mask != golden_mask:
                golden_mask_mismatches.append(index)

        reference_f1 = reference_record.get("score")
        current_f1 = current_record.get("score")
        reference_f1_loss = None
        if reference_f1 is not None and current_f1 is not None:
            reference_f1_loss = float(reference_f1) - float(current_f1)

        per_example.append(
            {
                "index": index,
                "example_id": current_record.get("example_id"),
                "active": is_active,
                "candidate_chunks": count,
                "reference_selected": reference_selected,
                "current_selected": current_selected,
                "selection_exact": selection_matches,
                "selection_overlap": overlap,
                "reference_ranked": reference_ranked,
                "current_ranked": current_ranked,
                "reference_boundary_margin": reference_margin,
                "reference_score_regret": regret,
                "reference_chunk_scores": (
                    None
                    if reference_scores is None
                    else [reference_scores.get(chunk) for chunk in range(count)]
                ),
                "current_chunk_scores": (
                    None
                    if current_scores is None
                    else [current_scores.get(chunk) for chunk in range(current_count)]
                ),
                "score_deltas": score_deltas,
                "score_abs_errors": score_abs,
                "centered_score_abs_errors": centered_abs,
                "score_pearson": sample_pearson,
                "score_spearman": sample_spearman,
                "partial_draft_ids": draft_ids,
                "draft_confirmed_mask": draft_mask,
                "draft_issues": draft_issues,
                "reference_f1": reference_f1,
                "current_f1": current_f1,
                "reference_f1_loss": reference_f1_loss,
                "issues": issues,
            }
        )

    mean_spearman = statistics.fmean(sample_spearmans) if sample_spearmans else None
    strict_reasons: list[str] = []
    practical_reasons: list[str] = []
    common_reasons: list[str] = []
    if config.expected_examples is not None and (
        len(reference) != config.expected_examples
        or len(current) != config.expected_examples
    ):
        common_reasons.append("unexpected_example_count")
    if (
        config.expected_active_examples is not None
        and active_count != config.expected_active_examples
    ):
        common_reasons.append("unexpected_active_example_count")
    if (
        config.expected_active_chunks is not None
        and active_chunk_count != config.expected_active_chunks
    ):
        common_reasons.append("unexpected_active_chunk_count")
    if missing_indices or extra_indices:
        common_reasons.append("index_set_mismatch")
    if structural_errors:
        common_reasons.append("structural_or_score_shape_errors")
    if config.require_partial_draft and (
        partial_missing_indices or partial_invalid or partial_present < active_count
    ):
        common_reasons.append("partial_draft_missing_or_invalid")
    if golden_missing_indices:
        common_reasons.append("golden_indices_missing_from_current")
    if golden_draft_mismatches or golden_mask_mismatches:
        common_reasons.append("partial_draft_golden_mismatch")
    if mean_spearman is None or mean_spearman < config.min_mean_spearman:
        common_reasons.append("mean_spearman_below_threshold")

    strict_reasons.extend(common_reasons)
    practical_reasons.extend(common_reasons)
    total_regret = sum(regrets)
    if top4_mismatches:
        strict_reasons.append("top4_mismatch")
    if total_regret > config.strict_max_regret:
        strict_reasons.append("reference_score_regret")

    if len(top4_mismatches) > config.practical_max_mismatches:
        practical_reasons.append("too_many_top4_mismatches")
    mismatch_rows = {
        row["index"]: row for row in per_example if row["index"] in top4_mismatches
    }
    if any(
        mismatch_rows[index]["reference_boundary_margin"] is None
        or mismatch_rows[index]["reference_boundary_margin"]
        > config.practical_max_margin
        for index in top4_mismatches
    ):
        practical_reasons.append("top4_mismatch_not_a_reference_near_tie")
    if total_regret > config.practical_max_regret:
        practical_reasons.append("reference_score_regret")
    if critical_mismatches:
        practical_reasons.append("critical_index_top4_mismatch")

    batch_invariance = None
    if peer is not None:
        batch_invariance = compare_batch_peer(current, peer, config)
        if not batch_invariance["passed"]:
            strict_reasons.append("batch_invariance_failure")
            practical_reasons.append("batch_invariance_failure")

    failure_rows = [
        row for row in per_example if row["active"] and not row["selection_exact"]
    ]
    failure_rows.sort(
        key=lambda row: (
            row["index"] not in config.critical_indices,
            -(row["reference_f1_loss"] or 0.0),
            -(row["reference_score_regret"] or 0.0),
            row["index"],
        )
    )

    summary = {
        "schema_version": 1,
        "counts": {
            "reference_examples": len(reference),
            "current_examples": len(current),
            "compared_examples": len(reference_indices & current_indices),
            "active_examples": active_count,
            "inactive_examples": len(reference_indices & current_indices)
            - active_count,
            "active_chunks": active_chunk_count,
            "score_comparable_chunks": score_comparable_count,
            "expected_examples": config.expected_examples,
            "expected_active_examples": config.expected_active_examples,
            "expected_active_chunks": config.expected_active_chunks,
        },
        "structure": {
            "missing_indices": missing_indices,
            "extra_indices": extra_indices,
            "error_count": len(structural_errors),
            "errors": structural_errors,
        },
        "partial_draft": {
            "required": config.require_partial_draft,
            "draft_slots": config.draft_slots,
            "expected_confirmed_count": config.expected_confirmed_count,
            "mask_token_id": config.mask_token_id,
            "present_examples": partial_present,
            "valid_examples": partial_valid,
            "active_missing_indices": partial_missing_indices,
            "active_invalid": partial_invalid,
            "golden_examples": len(golden),
            "golden_missing_indices": golden_missing_indices,
            "golden_draft_mismatch_indices": golden_draft_mismatches,
            "golden_mask_mismatch_indices": golden_mask_mismatches,
        },
        "selection": {
            "top_k": config.top_k,
            "exact_active_examples": active_count - len(top4_mismatches),
            "exact_active_rate": (
                (active_count - len(top4_mismatches)) / active_count
                if active_count
                else 1.0
            ),
            "top4_mismatch_indices": top4_mismatches,
            "ranked_top4_mismatch_indices": ranked_top4_mismatches,
            "top1_mismatch_indices": top1_mismatches,
            "critical_indices": list(config.critical_indices),
            "critical_mismatch_indices": critical_mismatches,
            "overlap_histogram": overlap_histogram,
            "reference_score_regret": distribution(regrets),
            "total_reference_score_regret": total_regret,
            "failures_ranked": [
                {
                    key: row[key]
                    for key in (
                        "index",
                        "reference_selected",
                        "current_selected",
                        "selection_overlap",
                        "reference_boundary_margin",
                        "reference_score_regret",
                        "reference_f1_loss",
                    )
                }
                for row in failure_rows
            ],
        },
        "scores": {
            "raw_abs_error": distribution(raw_abs_errors, include_rmse=True),
            "centered_abs_error": distribution(centered_abs_errors, include_rmse=True),
            "sample_pearson": distribution(sample_pearsons),
            "sample_spearman": distribution(sample_spearmans),
            "min_mean_spearman": config.min_mean_spearman,
        },
        "batch_invariance": batch_invariance,
        "gates": {
            "strict": {
                "passed": not strict_reasons,
                "reasons": sorted(set(strict_reasons)),
                "max_regret": config.strict_max_regret,
            },
            "practical": {
                "passed": not practical_reasons,
                "reasons": sorted(set(practical_reasons)),
                "max_mismatches": config.practical_max_mismatches,
                "max_reference_margin_for_mismatch": config.practical_max_margin,
                "max_total_regret": config.practical_max_regret,
            },
        },
    }
    return summary, per_example


def parse_indices(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(
        sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path, nargs="+")
    parser.add_argument(
        "--batch-peer",
        type=Path,
        nargs="+",
        help="Optional second current run, e.g. selector batch=8 versus batch=1.",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        help="Optional partial-draft golden JSON/JSONL keyed by index.",
    )
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--per-example-out", type=Path)
    parser.add_argument(
        "--gate", choices=("strict", "practical", "report"), default="strict"
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--expected-examples",
        type=int,
        default=150,
        help="Expected complete dataset size; use 0 to disable the count gate.",
    )
    parser.add_argument(
        "--expected-active-examples",
        type=int,
        default=110,
        help="Expected candidate_chunks > top_k count; use 0 to disable.",
    )
    parser.add_argument(
        "--expected-active-chunks",
        type=int,
        default=990,
        help="Expected number of scored chunks in active examples; use 0 to disable.",
    )
    parser.add_argument("--min-mean-spearman", type=float, default=0.99)
    parser.add_argument("--strict-max-regret", type=float, default=1e-8)
    parser.add_argument("--practical-max-mismatches", type=int, default=1)
    parser.add_argument("--practical-max-margin", type=float, default=0.002)
    parser.add_argument("--practical-max-regret", type=float, default=0.002)
    parser.add_argument("--batch-score-atol", type=float, default=1e-6)
    parser.add_argument("--draft-slots", type=int, default=4)
    parser.add_argument("--expected-confirmed-count", type=int, default=2)
    parser.add_argument("--mask-token-id", type=int, default=151666)
    parser.add_argument(
        "--critical-indices",
        default=",".join(str(index) for index in DEFAULT_CRITICAL_INDICES),
    )
    parser.add_argument("--require-partial-draft", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _write_outputs(
    summary: dict[str, Any],
    per_example: Sequence[dict[str, Any]],
    summary_path: Path,
    per_example_path: Path,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    per_example_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with per_example_path.open("w", encoding="utf-8") as output:
        for record in per_example:
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    config = GateConfig(
        top_k=args.top_k,
        min_mean_spearman=args.min_mean_spearman,
        strict_max_regret=args.strict_max_regret,
        practical_max_mismatches=args.practical_max_mismatches,
        practical_max_margin=args.practical_max_margin,
        practical_max_regret=args.practical_max_regret,
        batch_score_atol=args.batch_score_atol,
        draft_slots=args.draft_slots,
        expected_confirmed_count=args.expected_confirmed_count,
        mask_token_id=args.mask_token_id,
        critical_indices=parse_indices(args.critical_indices),
        require_partial_draft=args.require_partial_draft,
        expected_examples=args.expected_examples or None,
        expected_active_examples=args.expected_active_examples or None,
        expected_active_chunks=args.expected_active_chunks or None,
    )
    reference_records = load_records([args.reference])
    current_records = load_records(args.current)
    peer_records = load_records(args.batch_peer) if args.batch_peer else None
    golden_records = load_records([args.golden]) if args.golden else None
    summary, per_example = compare_selection(
        reference_records,
        current_records,
        config,
        peer_records=peer_records,
        golden_records=golden_records,
    )

    default_prefix = args.current[0].with_suffix("")
    summary_path = args.summary_out or Path(f"{default_prefix}_selection_summary.json")
    per_example_path = args.per_example_out or Path(
        f"{default_prefix}_selection_per_example.jsonl"
    )
    summary["inputs"] = {
        "reference": str(args.reference),
        "current": [str(path) for path in args.current],
        "batch_peer": (
            [str(path) for path in args.batch_peer] if args.batch_peer else None
        ),
        "golden": str(args.golden) if args.golden else None,
    }
    summary["outputs"] = {
        "summary": str(summary_path),
        "per_example": str(per_example_path),
    }
    _write_outputs(summary, per_example, summary_path, per_example_path)

    selected_gate = None if args.gate == "report" else summary["gates"][args.gate]
    if not args.quiet:
        counts = summary["counts"]
        selection = summary["selection"]
        score_summary = summary["scores"]
        print(
            "Compared "
            f"{counts['compared_examples']} examples; "
            f"active={counts['active_examples']}, chunks={counts['active_chunks']}"
        )
        print(
            "Top-k exact "
            f"{selection['exact_active_examples']}/{counts['active_examples']}; "
            f"total reference regret={selection['total_reference_score_regret']:.9g}"
        )
        print(
            "Score centered MAE="
            f"{score_summary['centered_abs_error'].get('mean')}; "
            "mean Spearman="
            f"{score_summary['sample_spearman'].get('mean')}"
        )
        if selected_gate is None:
            print("Gate: report only")
        else:
            state = "PASS" if selected_gate["passed"] else "FAIL"
            print(f"Gate {args.gate}: {state}")
            if selected_gate["reasons"]:
                print("Reasons: " + ", ".join(selected_gate["reasons"]))
        print(f"Summary: {summary_path}")
        print(f"Per-example: {per_example_path}")
    return 0 if selected_gate is None or selected_gate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

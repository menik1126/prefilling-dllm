import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).with_name("compare_multifieldqa_selection.py")
SPEC = importlib.util.spec_from_file_location("compare_multifieldqa_selection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_record(index, scores, selected, *, offset=0.0, partial=True):
    record = {
        "index": index,
        "example_id": f"example-{index}",
        "candidate_chunks": len(scores),
        "raw_context_tokens": len(scores) * 10,
        "prefix_tokens": 2,
        "query_tokens": 3,
        "selected_chunk_indices": selected,
        "chunk_scores": [score + offset for score in scores],
        "score": 0.5,
    }
    if partial:
        record["partial_draft_ids"] = [10, 151666, 20, 151666]
        record["draft_confirmed_mask"] = [True, False, True, False]
    return record


def make_reference(index, scores, selected):
    record = make_record(index, scores, selected, partial=False)
    payload = {
        key: record.pop(key)
        for key in (
            "candidate_chunks",
            "raw_context_tokens",
            "prefix_tokens",
            "query_tokens",
            "selected_chunk_indices",
            "chunk_scores",
        )
    }
    payload["chunk_scores"] = {
        str(index): value for index, value in enumerate(payload["chunk_scores"])
    }
    record["parallelcomp"] = payload
    return record


def test_strict_gate_passes_with_score_offset_and_identical_top4():
    scores = [5.0, 4.0, 3.0, 2.0, 1.0]
    reference = [make_reference(0, scores, [0, 1, 2, 3])]
    current = [make_record(0, scores, [0, 1, 2, 3], offset=0.25)]
    config = MODULE.GateConfig(require_partial_draft=True, critical_indices=())

    summary, rows = MODULE.compare_selection(reference, current, config)

    assert summary["gates"]["strict"]["passed"]
    assert summary["selection"]["total_reference_score_regret"] == 0.0
    assert summary["scores"]["raw_abs_error"]["mean"] == 0.25
    assert summary["scores"]["centered_abs_error"]["max"] == 0.0
    assert rows[0]["selection_exact"]


def test_practical_gate_allows_one_small_reference_near_tie():
    scores = [float(value) for value in range(20, 0, -1)]
    scores[3] = 17.0
    scores[4] = 16.999
    reference = [make_reference(0, scores, [0, 1, 2, 3])]
    current_scores = list(scores)
    current_scores[3], current_scores[4] = current_scores[4], current_scores[3]
    current = [make_record(0, current_scores, [0, 1, 2, 4])]
    config = MODULE.GateConfig(require_partial_draft=True, critical_indices=())

    summary, _ = MODULE.compare_selection(reference, current, config)

    assert not summary["gates"]["strict"]["passed"]
    assert summary["gates"]["practical"]["passed"]
    assert summary["selection"]["top4_mismatch_indices"] == [0]
    assert math.isclose(summary["selection"]["total_reference_score_regret"], 0.001)


def test_batch_peer_detects_partial_draft_and_score_changes():
    scores = [5.0, 4.0, 3.0, 2.0, 1.0]
    reference = [make_reference(0, scores, [0, 1, 2, 3])]
    current = [make_record(0, scores, [0, 1, 2, 3])]
    peer = [make_record(0, scores, [0, 1, 2, 3])]
    peer[0]["partial_draft_ids"][0] = 11
    peer[0]["chunk_scores"][0] += 1e-3
    config = MODULE.GateConfig(require_partial_draft=True, critical_indices=())

    summary, _ = MODULE.compare_selection(reference, current, config, peer_records=peer)

    assert not summary["batch_invariance"]["passed"]
    assert summary["batch_invariance"]["draft_mismatch_indices"] == [0]
    assert summary["batch_invariance"]["score_mismatch_indices"] == [0]
    assert not summary["gates"]["strict"]["passed"]


def test_json_suffix_can_contain_jsonl_and_duplicate_indices_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reference.json"
        path.write_text(
            "\n".join(json.dumps({"index": index}) for index in (0, 1)) + "\n",
            encoding="utf-8",
        )
        records = MODULE.load_records([path])
    assert [record["index"] for record in records] == [0, 1]

    try:
        MODULE.index_records([{"index": 0}, {"index": 0}], "current")
    except ValueError as error:
        assert "duplicate index 0" in str(error)
    else:
        raise AssertionError("duplicate indices must be rejected")

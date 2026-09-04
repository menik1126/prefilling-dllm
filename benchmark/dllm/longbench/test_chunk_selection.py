import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("bench_multifieldqa_chunk_selection.py")
SPEC = importlib.util.spec_from_file_location("chunk_selection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_prompt_split():
    parts = MODULE.render_prompt_parts(
        "prefix {context} question {input}",
        {"context": "ctx", "input": "q"},
    )
    assert parts == {"prefix": "prefix ", "context": "ctx", "query": " question q"}


def test_chunking_and_bos():
    chunks = MODULE.split_token_chunks([1, 2, 3, 4, 5], 2)
    assert chunks == [[1, 2], [3, 4], [5]]
    assert MODULE.add_chunk_bos(chunks, 99, 2) == [[99, 1], [99, 3], [99, 5]]


def test_query_selection_preserves_document_order():
    selected = MODULE.select_chunk_indices(
        "query_logprob",
        chunk_count=4,
        top_k=2,
        scores=[0.1, 0.9, 0.2, 0.8],
    )
    assert selected == [1, 3]


def test_chunk_scoring_is_skipped_when_scores_cannot_change_selection():
    assert MODULE.requires_chunk_scoring("query_logprob", 5, 4)
    assert not MODULE.requires_chunk_scoring("query_logprob", 4, 4)
    assert not MODULE.requires_chunk_scoring("query_logprob", 3, 4)
    assert not MODULE.requires_chunk_scoring("query_logprob", 5, 0)
    assert not MODULE.requires_chunk_scoring("head", 5, 4)


def test_qa_f1():
    assert MODULE.qa_f1_score("The blue whale", "blue whale") == 1.0
    assert MODULE.qa_f1_score("red fox", "blue whale") == 0.0


def test_causal_prompt_logprob_marker():
    client = MODULE.SGLangClient(
        "http://example.invalid",
        timeout=1,
        causal_prompt_logprobs=True,
    )
    payloads = []
    client.post = lambda payload: payloads.append(payload) or {
        "meta_info": {"input_token_logprobs": [[-1.0, 7, None]]}
    }
    assert client.prompt_logprobs([[1, 7]], [1]) == [[[-1.0, 7, None]]]
    assert payloads[0]["sampling_params"]["custom_params"] == {
        "dream_causal_prompt_logprob": True,
    }


def test_single_generate_batch_preserves_scalar_request_shape():
    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    payloads = []
    response = {"text": "answer", "meta_info": {"row": 0}}
    client.post = lambda payload: payloads.append(payload) or response
    request_custom_params = {"tag": "one"}

    result = client.generate_batch(
        [[1, 2]],
        7,
        position_starts=[2],
        position_offsets=[3],
        custom_params=[request_custom_params],
    )

    assert result == [response]
    assert payloads == [
        {
            "input_ids": [1, 2],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 7,
                "custom_params": {
                    "tag": "one",
                    "dllm_position_start": 2,
                    "dllm_position_offset": 3,
                },
            },
        }
    ]
    assert request_custom_params == {"tag": "one"}


def test_generate_batch_preserves_request_and_response_order():
    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    payloads = []
    responses = [
        {"text": "first", "meta_info": {"row": 0}},
        {"text": "second", "meta_info": {"row": 1}},
    ]
    client.post = lambda payload: payloads.append(payload) or responses

    result = client.generate_batch(
        [[1], [2, 3]],
        7,
        position_starts=[4, 5],
        position_offsets=[6, 0],
        custom_params=[{"tag": "first"}, {"tag": "second"}],
    )

    assert result == responses
    assert payloads[0]["input_ids"] == [[1], [2, 3]]
    assert payloads[0]["sampling_params"] == [
        {
            "temperature": 0,
            "max_new_tokens": 7,
            "custom_params": {
                "tag": "first",
                "dllm_position_start": 4,
                "dllm_position_offset": 6,
            },
        },
        {
            "temperature": 0,
            "max_new_tokens": 7,
            "custom_params": {"tag": "second"},
        },
    ]


def test_generate_batch_validates_parameter_and_response_rows():
    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    client.post = lambda payload: (_ for _ in ()).throw(AssertionError(payload))

    assert client.generate_batch([], 7) == []
    with pytest.raises(ValueError, match="position_starts has 1 rows, expected 2"):
        client.generate_batch([[1], [2]], 7, position_starts=[0])

    client.post = lambda payload: {"text": "only one row"}
    with pytest.raises(RuntimeError, match="returned 1 rows, expected 2"):
        client.generate_batch([[1], [2]], 7)

    client.post = lambda payload: [{"text": "valid"}, "invalid"]
    with pytest.raises(RuntimeError, match="non-object rows at indices \\[1\\]"):
        client.generate_batch([[1], [2]], 7)


def test_draft_ids_fall_back_to_diffusion_output_ids():
    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    payloads = []
    client.post = lambda payload: payloads.append(payload) or {
        "output_ids": [10, 11, 12, 13],
        "meta_info": {"output_token_logprobs": []},
    }
    assert client.draft_ids([1, 2], 4, 32) == [10, 11, 12, 13]
    assert payloads[0]["input_ids"] == [1, 2]


def test_single_draft_batch_preserves_scalar_request_shape():
    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    payloads = []
    client.post = lambda payload: payloads.append(payload) or {
        "output_ids": [10, 11],
        "meta_info": {"output_token_logprobs": []},
    }

    assert client.draft_ids_batch([[1, 2]], 2, 32) == [[10, 11]]
    assert payloads[0]["input_ids"] == [1, 2]


def test_draft_ids_batch_preserves_response_order():
    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    payloads = []
    client.post = lambda payload: payloads.append(payload) or [
        {
            "output_ids": [10, 11],
            "meta_info": {"output_token_logprobs": []},
        },
        {
            "output_ids": [20, 21],
            "meta_info": {"output_token_logprobs": []},
        },
    ]

    assert client.draft_ids_batch([[1], [2]], 2, 32) == [[10, 11], [20, 21]]
    assert payloads[0]["input_ids"] == [[1], [2]]


def test_zero_length_draft_batch_skips_request():
    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    client.post = lambda payload: (_ for _ in ()).throw(AssertionError(payload))
    assert client.draft_ids_batch([[1], [2]], 0, 32) == [[], []]


def test_partial_draft_preserves_all_slots_and_confirmed_mask():
    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    payloads = []
    client.post = lambda payload: payloads.append(payload) or {
        "output_ids": [10, 151666, 12, 151666],
        "meta_info": {
            "output_token_logprobs": [],
            "dllm_confirmed_mask": [True, False, True, False],
        },
    }

    draft = client.partial_draft([1, 2], 4, rounds=1)

    assert draft == MODULE.PartialDraft(
        token_ids=[10, 151666, 12, 151666],
        confirmed_mask=[True, False, True, False],
    )
    assert payloads[0]["input_ids"] == [1, 2]
    assert payloads[0]["sampling_params"] == {
        "temperature": 0,
        "max_new_tokens": 4,
        "ignore_eos": True,
        "custom_params": {"dllm_partial_draft": {"rounds": 1}},
    }


def test_partial_draft_batch_preserves_response_order():
    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    payloads = []
    client.post = lambda payload: payloads.append(payload) or [
        {
            "output_ids": [10, 151666, 12, 151666],
            "meta_info": {
                "dllm_confirmed_mask": [True, False, True, False],
            },
        },
        {
            "output_ids": [20, 21, 151666, 151666],
            "meta_info": {
                "dllm_confirmed_mask": [True, True, False, False],
            },
        },
    ]

    drafts = client.partial_draft_batch([[1], [2]], 4, rounds=1)

    assert drafts == [
        MODULE.PartialDraft(
            token_ids=[10, 151666, 12, 151666],
            confirmed_mask=[True, False, True, False],
        ),
        MODULE.PartialDraft(
            token_ids=[20, 21, 151666, 151666],
            confirmed_mask=[True, True, False, False],
        ),
    ]
    assert payloads[0]["input_ids"] == [[1], [2]]
    assert payloads[0]["sampling_params"]["max_new_tokens"] == 4


def test_single_partial_draft_batch_preserves_scalar_request_shape():
    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    payloads = []
    client.post = lambda payload: payloads.append(payload) or {
        "output_ids": [10, 11, 151666, 151666],
        "meta_info": {
            "dllm_confirmed_mask": [True, True, False, False],
        },
    }

    drafts = client.partial_draft_batch([[1, 2]], 4, rounds=1)

    assert drafts[0].token_ids == [10, 11, 151666, 151666]
    assert payloads[0]["input_ids"] == [1, 2]


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (
            {"output_ids": [10, 11, 12, 13], "meta_info": {}},
            "missing meta_info.dllm_confirmed_mask",
        ),
        (
            {
                "output_ids": [10, 11, 12, 13],
                "meta_info": {"dllm_confirmed_mask": [True, False]},
            },
            "confirmed mask has 2 slots, expected 4",
        ),
        (
            {
                "output_ids": [10, 11, 12, 13],
                "meta_info": {"dllm_confirmed_mask": [1, 0, 1, 0]},
            },
            "must contain only booleans",
        ),
        (
            {
                "output_ids": [10, 11, 12, 13],
                "meta_info": {"dllm_confirmed_mask": [True, True, True, False]},
            },
            "confirmed 3 slots, expected 2",
        ),
        (
            {
                "output_ids": [10, 11, 12, 13],
                "meta_info": {"dllm_confirmed_mask": [False, True, True, False]},
            },
            "must confirm slot zero",
        ),
    ],
)
def test_partial_draft_rejects_invalid_confirmed_mask(response, error):
    with pytest.raises(RuntimeError, match=error):
        MODULE.SGLangClient._partial_draft_from_result(response, 4, 1)


def test_partial_draft_rejects_wrong_slot_count_and_negative_rounds():
    with pytest.raises(RuntimeError, match="3 slots, expected 4"):
        MODULE.SGLangClient._partial_draft_from_result(
            {
                "output_ids": [10, 11, 12],
                "meta_info": {"dllm_confirmed_mask": [True, True, False, False]},
            },
            4,
            1,
        )

    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    client.post = lambda payload: (_ for _ in ()).throw(AssertionError(payload))
    with pytest.raises(ValueError, match="rounds must be non-negative"):
        client.partial_draft([1], 4, rounds=-1)


@pytest.mark.parametrize("draft_tokens", [1, 2, 3, 5, 8])
def test_main_rejects_unsupported_partial_draft_slot_count_before_io(
    monkeypatch, tmp_path, draft_tokens
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE.__file__),
            "--model-path",
            "unused",
            "--data-path",
            str(tmp_path / "missing-data.jsonl"),
            "--prompt-config",
            str(tmp_path / "missing-prompts.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--draft-tokens",
            str(draft_tokens),
        ],
    )

    with pytest.raises(ValueError, match="must be 0 .* or 4"):
        MODULE.main()


def test_main_requires_dedicated_causal_scorer_before_io(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE.__file__),
            "--model-path",
            "unused",
            "--data-path",
            str(tmp_path / "missing-data.jsonl"),
            "--prompt-config",
            str(tmp_path / "missing-prompts.json"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    with pytest.raises(ValueError, match="requires a dedicated --score-base-url"):
        MODULE.main()


def test_score_chunk_groups_flattens_and_scatters_variable_groups():
    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    batch_shapes = []

    def prompt_logprobs(rows, starts):
        batch_shapes.append((len(rows), list(starts)))
        return [[[float(token_id), token_id, None] for token_id in row] for row in rows]

    client.prompt_logprobs = prompt_logprobs
    scores = MODULE.score_chunk_groups(
        client,
        [
            ([1], [[10], [11]], [100, 101]),
            ([2, 3], [[12]], [102]),
            ([], [[13], [14], [15]], [103, 104, 105]),
        ],
        batch_size=4,
    )

    # Prompt logprobs start one position before the scoring target so the
    # causal shift includes the first query token.
    assert batch_shapes == [(4, [1, 1, 2, 0]), (2, [0, 0])]
    assert scores == [[100.5, 100.5], [102.0], [104.0, 104.0, 104.0]]


def test_score_chunk_groups_keeps_partial_slots_but_masks_their_logprobs():
    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    seen_rows = []

    def prompt_logprobs(rows, starts):
        seen_rows.extend(rows)
        assert starts == [1, 1, 2]
        return [[[float(token_id), token_id, None] for token_id in row] for row in rows]

    client.prompt_logprobs = prompt_logprobs
    scores = MODULE.score_chunk_groups(
        client,
        [
            (
                [1],
                [[10], [11]],
                [100, 101, 151666, 102, 151666],
            ),
            (
                [2, 3],
                [[12]],
                [200, 201, 151666, 151666],
            ),
        ],
        batch_size=3,
        score_token_masks=[
            [True, True, False, True, False],
            [True, True, False, False],
        ],
    )

    assert seen_rows == [
        [1, 10, 100, 101, 151666, 102, 151666],
        [1, 11, 100, 101, 151666, 102, 151666],
        [2, 3, 12, 200, 201, 151666, 151666],
    ]
    assert scores == [[101.0, 101.0], [200.5]]


def test_masked_mean_uses_trailing_target_positions_only():
    values = [
        [-999.0, 1, None],
        [-1.0, 100, None],
        [-2.0, 101, None],
        [-100.0, 151666, None],
        [-3.0, 102, None],
        [-100.0, 151666, None],
    ]
    assert (
        MODULE.mean_query_logprob(
            values,
            5,
            [True, True, False, True, False],
        )
        == -2.0
    )


def test_mean_query_logprob_rejects_missing_scored_target():
    with pytest.raises(ValueError, match="missing a scored target at offset 0"):
        MODULE.mean_query_logprob([[None, 100, None]], 1)

    assert (
        MODULE.mean_query_logprob(
            [[None, 100, None], [-2.0, 101, None]],
            2,
            [False, True],
        )
        == -2.0
    )


def test_score_chunk_groups_rejects_invalid_mask_shape_and_values():
    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    client.prompt_logprobs = lambda rows, starts: (_ for _ in ()).throw(
        AssertionError((rows, starts))
    )
    groups = [([1], [[10]], [100, 101])]

    with pytest.raises(ValueError, match="0 groups, expected 1"):
        MODULE.score_chunk_groups(
            client,
            groups,
            batch_size=1,
            score_token_masks=[],
        )
    with pytest.raises(ValueError, match="length does not match"):
        MODULE.score_chunk_groups(
            client,
            groups,
            batch_size=1,
            score_token_masks=[[True]],
        )
    with pytest.raises(ValueError, match="only booleans"):
        MODULE.score_chunk_groups(
            client,
            groups,
            batch_size=1,
            score_token_masks=[[True, 1]],
        )


def test_main_batches_final_generation_and_scatters_in_input_order(
    monkeypatch, tmp_path
):
    data_path = tmp_path / "multifieldqa_en.jsonl"
    data_path.write_text(
        "\n".join(
            json.dumps(example)
            for example in [
                {
                    "_id": "example-0",
                    "context": "CTX0",
                    "input": "ASK0",
                    "answers": ["first"],
                },
                {
                    "_id": "example-1",
                    "context": "CTX1",
                    "input": "ASK1",
                    "answers": ["second"],
                },
                {
                    "_id": "example-2",
                    "context": "CTX2",
                    "input": "ASK2",
                    "answers": ["third"],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_path = tmp_path / "dataset2prompt.json"
    prompt_path.write_text(
        json.dumps({"multifieldqa_en": "P{context}Q{input}"}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    class FakeTokenizer:
        bos_token_id = None

        @staticmethod
        def encode(text, add_special_tokens=False):
            assert not add_special_tokens
            return {
                "P": [1],
                "CTX0": [10, 11, 12],
                "CTX1": [13, 14, 15, 16],
                "CTX2": [17, 18, 19, 20],
                "QASK0": [20],
                "QASK1": [21],
                "QASK2": [22],
            }[text]

    class FakeClient:
        instances = []

        def __init__(self, *args, **kwargs):
            self.batch_calls = []
            self.instances.append(self)

        def generate(self, *args, **kwargs):
            raise AssertionError("main must use generate_batch")

        def generate_batch(
            self,
            input_ids,
            max_new_tokens,
            *,
            position_starts=None,
            position_offsets=None,
            custom_params=None,
        ):
            self.batch_calls.append(
                {
                    "input_ids": input_ids,
                    "max_new_tokens": max_new_tokens,
                    "position_starts": position_starts,
                    "position_offsets": position_offsets,
                    "custom_params": custom_params,
                }
            )
            outputs = {
                20: ("first trailing", 0),
                21: ("second trailing", 1),
                22: ("third trailing", 2),
            }
            return [
                {
                    "text": outputs[ids[-1]][0],
                    "meta_info": {"row": outputs[ids[-1]][1]},
                }
                for ids in input_ids
            ]

    clock = iter([10.0, 14.0, 20.0, 23.0])
    monkeypatch.setattr(MODULE.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        MODULE.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(MODULE, "SGLangClient", FakeClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE.__file__),
            "--model-path",
            "unused",
            "--data-path",
            str(data_path),
            "--prompt-config",
            str(prompt_path),
            "--output-dir",
            str(output_dir),
            "--selection-mode",
            "fixed",
            "--fixed-chunk-indices",
            "1",
            "--chunk-size",
            "2",
            "--selector-microbatch-size",
            "3",
            "--generation-microbatch-size",
            "2",
            "--query-position-mode",
            "after_selected_chunks",
            "--max-new-tokens",
            "7",
            "--prediction-max-words",
            "1",
            "--num-examples",
            "3",
        ],
    )

    MODULE.main()

    generation_client = FakeClient.instances[0]
    assert generation_client.batch_calls == [
        {
            "input_ids": [[1, 12, 20], [1, 15, 16, 21]],
            "max_new_tokens": 7,
            "position_starts": [2, 3],
            "position_offsets": [1, 0],
            "custom_params": [None, None],
        },
        {
            "input_ids": [[1, 19, 20, 22]],
            "max_new_tokens": 7,
            "position_starts": [3],
            "position_offsets": [0],
            "custom_params": [None],
        },
    ]
    records = [
        json.loads(line)
        for line in (output_dir / "multifieldqa_en_fixed_continuous.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    metrics = json.loads(
        (output_dir / "multifieldqa_en_fixed_continuous_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert [record["index"] for record in records] == [0, 1, 2]
    assert [record["example_id"] for record in records] == [
        "example-0",
        "example-1",
        "example-2",
    ]
    assert [record["raw_prediction"] for record in records] == [
        "first trailing",
        "second trailing",
        "third trailing",
    ]
    assert [record["prediction"] for record in records] == [
        "first",
        "second",
        "third",
    ]
    assert [record["generation_meta"] for record in records] == [
        {"row": 0},
        {"row": 1},
        {"row": 2},
    ]
    assert [record["score"] for record in records] == [1.0, 1.0, 1.0]
    assert [record["generation_seconds"] for record in records] == [2.0, 2.0, 3.0]
    assert [record["generation_seconds_are_attributed"] for record in records] == [
        True,
        True,
        False,
    ]
    assert [record["generation_active_microbatch_size"] for record in records] == [
        2,
        2,
        1,
    ]
    assert metrics["score"] == 100.0
    assert metrics["generation_request_count"] == 2
    assert metrics["generation_batch_size_histogram"] == {"1": 1, "2": 1}
    assert metrics["total_generation_seconds"] == 7.0
    assert metrics["total_generation_batch_seconds"] == 7.0


def test_selection_only_writes_selector_artifacts_without_generation(
    monkeypatch, tmp_path
):
    data_path = tmp_path / "multifieldqa_en.jsonl"
    data_path.write_text(
        "\n".join(
            json.dumps(example)
            for example in [
                {
                    "_id": "example-0",
                    "context": "CTX",
                    "input": "ASK",
                    "answers": ["answer"],
                },
                {
                    "_id": "example-1",
                    "context": "ONE",
                    "input": "ASK",
                    "answers": ["answer"],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_path = tmp_path / "dataset2prompt.json"
    prompt_path.write_text(
        json.dumps({"multifieldqa_en": "P{context}Q{input}"}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    class FakeTokenizer:
        bos_token_id = None

        @staticmethod
        def encode(text, add_special_tokens=False):
            assert not add_special_tokens
            return {
                "P": [1],
                "CTX": [10, 11],
                "ONE": [12],
                "QASK": [20, 21],
            }[text]

    class FakeClient:
        instances = []

        def __init__(self, *args, **kwargs):
            self.partial_calls = []
            self.score_rows = []
            self.generate_calls = 0
            self.causal_prompt_logprobs = kwargs.get("causal_prompt_logprobs", False)
            self.instances.append(self)

        def partial_draft_batch(
            self, input_ids, max_new_tokens, *, rounds=MODULE.PARTIAL_DRAFT_ROUNDS
        ):
            self.partial_calls.append((input_ids, max_new_tokens, rounds))
            return [
                MODULE.PartialDraft(
                    [30, 151666, 31, 151666],
                    [True, False, True, False],
                )
                for _ in input_ids
            ]

        def prompt_logprobs(self, rows, starts):
            self.score_rows.extend(rows)
            return [
                [
                    [-float(offset + 1), token_id, None]
                    for offset, token_id in enumerate(row)
                ]
                for row in rows
            ]

        def generate(self, *args, **kwargs):
            self.generate_calls += 1
            raise AssertionError("selection-only must skip final generation")

    monkeypatch.setattr(
        MODULE.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(MODULE, "SGLangClient", FakeClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE.__file__),
            "--model-path",
            "unused",
            "--data-path",
            str(data_path),
            "--prompt-config",
            str(prompt_path),
            "--output-dir",
            str(output_dir),
            "--chunk-size",
            "1",
            "--top-k",
            "1",
            "--draft-tokens",
            "4",
            "--num-examples",
            "2",
            "--score-base-url",
            "http://score.invalid",
            "--selection-only",
        ],
    )

    MODULE.main()

    records = [
        json.loads(line)
        for line in (output_dir / "multifieldqa_en_query_logprob_continuous.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    metrics = json.loads(
        (
            output_dir / "multifieldqa_en_query_logprob_continuous_metrics.json"
        ).read_text(encoding="utf-8")
    )

    assert len(records) == 2
    assert records[0]["draft_ids"] == [30, 151666, 31, 151666]
    assert records[0]["partial_draft_ids"] == [30, 151666, 31, 151666]
    assert records[0]["draft_confirmed_mask"] == [True, False, True, False]
    assert records[0]["prediction"] == ""
    assert records[0]["raw_prediction"] == ""
    assert records[0]["score"] is None
    assert records[0]["generation_seconds"] == 0.0
    assert records[0]["generation_meta"] is None
    assert records[1]["selected_chunk_indices"] == [0]
    assert records[1]["chunk_scores"] is None
    assert records[1]["draft_ids"] == []
    assert records[1]["partial_draft_ids"] == []
    assert records[1]["draft_confirmed_mask"] == []
    assert records[1]["selector_scoring_skipped"] == "top_k_covers_all"
    assert records[1]["score_seconds"] == 0.0
    assert records[1]["generation_seconds"] == 0.0
    assert metrics["score"] is None
    assert metrics["selection_only"] is True
    assert metrics["draft_partial_rounds"] == 1
    assert metrics["total_generation_seconds"] == 0.0
    assert metrics["generation_request_count"] == 0
    assert metrics["generation_batch_size_histogram"] == {}
    assert metrics["generation_seconds_are_attributed"] is False
    assert all(record["generation_active_microbatch_size"] == 0 for record in records)
    assert FakeClient.instances[0].partial_calls == [([[1, 20, 21]], 4, 1)]
    assert not FakeClient.instances[0].causal_prompt_logprobs
    assert FakeClient.instances[1].causal_prompt_logprobs
    assert len(FakeClient.instances[1].score_rows) == 2
    assert all(client.generate_calls == 0 for client in FakeClient.instances)


def test_mean_query_logprob_rejects_empty_scoring_target():
    try:
        MODULE.mean_query_logprob([[-1.0, 1, None]], 0)
    except ValueError as error:
        assert str(error) == "expected_tokens must be positive"
    else:
        raise AssertionError("zero-token scoring target should fail")

import importlib.util
from pathlib import Path


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


def test_score_chunk_groups_flattens_and_scatters_variable_groups():
    client = MODULE.SGLangClient("http://example.invalid", timeout=1)
    batch_shapes = []

    def prompt_logprobs(rows, starts):
        batch_shapes.append((len(rows), list(starts)))
        return [
            [[float(token_id), token_id, None] for token_id in row] for row in rows
        ]

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

    assert batch_shapes == [(4, [2, 2, 3, 1]), (2, [1, 1])]
    assert scores == [[100.5, 100.5], [102.0], [104.0, 104.0, 104.0]]


def test_mean_query_logprob_rejects_empty_scoring_target():
    try:
        MODULE.mean_query_logprob([[-1.0, 1, None]], 0)
    except ValueError as error:
        assert str(error) == "expected_tokens must be positive"
    else:
        raise AssertionError("zero-token scoring target should fail")

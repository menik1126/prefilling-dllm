import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from prefilling_dllm.client import SGLangClient
from prefilling_dllm.pipeline import (
    PipelineConfig,
    PrefillingDreamPipeline,
    split_template,
)
from prefilling_dllm.server import PipelineHTTPServer

BENCH_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmark/dllm/longbench/bench_multifieldqa_chunk_selection.py"
)
BOS = 1
TEMPLATE = "Read: {context} Question: {question} Answer:"
# Six chunks of size 8; chunks 1, 5, 4 hold the most "z" in that order.
LONG_CONTEXT = "aaaaaaaa" "zzzzaaaa" "aaaaaaaa" "aaaaaaaa" "zaaaaaaa" "zzz"
SHORT_CONTEXT = "aaaaaaaa" "zz"


class CharTokenizer:
    bos_token_id = BOS

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(char) for char in text]


class FakeSGLang:
    """Stands in for SGLangClient.post; scores a row by its count of "z"."""

    def __init__(self):
        self.payloads = []
        self.error = None

    def post(self, payload):
        if self.error is not None:
            raise self.error
        self.payloads.append(payload)
        if "logprob_start_len" in payload:
            return [
                {
                    "meta_info": {
                        "input_token_logprobs": [
                            [float(row.count(ord("z"))), token_id, None]
                            for token_id in row[start:]
                        ]
                    }
                }
                for row, start in zip(
                    payload["input_ids"], payload["logprob_start_len"]
                )
            ]
        custom_params = payload["sampling_params"].get("custom_params", {})
        if "dllm_partial_draft" in custom_params:
            return {
                "output_ids": [7, 8, 9, 10],
                "meta_info": {"dllm_confirmed_mask": [True, True, False, False]},
            }
        return {
            "text": f"answer for {len(payload['input_ids'])} tokens",
            "meta_info": {"id": "generation"},
        }


def make_pipeline(fake, **config):
    client = SGLangClient("http://example.invalid", timeout=1)
    client.post = fake.post
    return PrefillingDreamPipeline(
        tokenizer=CharTokenizer(),
        client=client,
        config=PipelineConfig(chunk_size=8, top_k=2, **config),
    )


def test_split_template():
    assert split_template(TEMPLATE, question="why?") == (
        "Read: ",
        " Question: why? Answer:",
    )
    with pytest.raises(ValueError, match="context"):
        split_template("{question}", question="q")
    with pytest.raises(ValueError, match="question"):
        split_template("{context}", question="q")


def test_long_document_is_drafted_scored_and_compressed():
    fake = FakeSGLang()
    result = make_pipeline(fake).answer(
        context=LONG_CONTEXT, question="why?", template=TEMPLATE
    )

    prefix_ids = [BOS] + [ord(char) for char in "Read: "]
    query_ids = [ord(char) for char in " Question: why? Answer:"]
    draft, scoring, generation = fake.payloads
    assert draft["input_ids"] == prefix_ids + query_ids
    assert len(scoring["input_ids"]) == 6
    assert scoring["sampling_params"][0]["custom_params"] == {
        "dream_score_attention_mask": "full",
        "dream_score_prefix_len": len(prefix_ids),
        "dream_score_chunk_len": 8,
        "dream_score_query_len": len(query_ids),
        "dream_score_draft_len": 4,
    }

    assert result.num_chunks == 6
    assert result.selected_chunk_indices == [1, 5]
    assert result.chunk_scores == [0.0, 4.0, 0.0, 0.0, 1.0, 3.0]
    assert result.draft_ids == [7, 8, 9, 10]
    chunk_1 = [BOS] + [ord(char) for char in "zzzzaaa"]
    chunk_5 = [BOS] + [ord(char) for char in "zzz"]
    assert generation["input_ids"] == prefix_ids + chunk_1 + chunk_5 + query_ids
    # The four-token last chunk still occupies a full eight-position slot.
    assert generation["sampling_params"]["custom_params"] == {
        "dllm_position_start": len(prefix_ids) + 12,
        "dllm_position_offset": 4,
    }
    assert result.answer == f"answer for {result.prompt_tokens} tokens"
    assert result.generation_meta == {"id": "generation"}


def test_short_document_skips_draft_and_scoring():
    fake = FakeSGLang()
    result = make_pipeline(fake).answer(
        context=SHORT_CONTEXT, question="why?", template=TEMPLATE
    )
    assert len(fake.payloads) == 1
    assert result.selected_chunk_indices == [0, 1]
    assert result.chunk_scores is None
    assert result.draft_ids == []


def test_token_eviction_spans_follow_the_selected_chunks():
    fake = FakeSGLang()
    make_pipeline(fake, token_capacity=4, token_score_bidirectional=False).answer(
        context=LONG_CONTEXT, question="why?", template=TEMPLATE
    )
    eviction = fake.payloads[-1]["sampling_params"]["custom_params"][
        "dllm_token_eviction"
    ]
    assert eviction["capacity"] == 4
    assert eviction["chunk_lens"] == [8, 4]
    assert eviction["bidirectional"] is False


def test_scoring_requests_can_use_a_separate_pd_server():
    fake = FakeSGLang()
    score_fake = FakeSGLang()
    pipeline = make_pipeline(fake)
    score_client = SGLangClient(
        "http://score.invalid", timeout=1, score_on_pd_server=True
    )
    score_client.post = score_fake.post
    pipeline.score_client = score_client
    pipeline.answer(context=LONG_CONTEXT, question="why?", template=TEMPLATE)

    assert len(fake.payloads) == 2
    assert len(score_fake.payloads) == 1
    assert len(score_fake.payloads[0]["bootstrap_room"]) == 6


def test_config_rejects_unsupported_draft_canvas():
    with pytest.raises(ValueError, match="draft_tokens"):
        PipelineConfig(draft_tokens=2)


def test_pipeline_sends_the_same_requests_as_the_benchmark_client(
    monkeypatch, tmp_path
):
    spec = importlib.util.spec_from_file_location("bench_chunk_selection", BENCH_SCRIPT)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    examples = [
        {"context": LONG_CONTEXT, "input": "why?", "answers": ["x"]},
        {"context": SHORT_CONTEXT, "input": "who?", "answers": ["y"]},
    ]
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("".join(json.dumps(example) + "\n" for example in examples))
    prompt_config = tmp_path / "prompt.json"
    prompt_config.write_text(
        json.dumps({bench.TASK: TEMPLATE.replace("{question}", "{input}")})
    )

    bench_fake = FakeSGLang()
    monkeypatch.setattr(
        SGLangClient, "post", lambda self, payload: bench_fake.post(payload)
    )
    monkeypatch.setattr(
        bench.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: CharTokenizer()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(BENCH_SCRIPT),
            "--model-path=unused",
            f"--data-path={data_path}",
            f"--prompt-config={prompt_config}",
            f"--output-dir={tmp_path / 'out'}",
            "--query-position-mode=after_selected_chunks",
            "--chunk-size=8",
            "--top-k=2",
            "--score-batch-size=4",
            "--draft-tokens=4",
            "--token-capacity=4",
        ],
    )
    bench.main()

    fake = FakeSGLang()
    pipeline = make_pipeline(fake, score_batch_size=4, token_capacity=4)
    for example in examples:
        pipeline.answer(
            context=example["context"], question=example["input"], template=TEMPLATE
        )
    assert fake.payloads == bench_fake.payloads
    assert len(fake.payloads) == 5


@pytest.fixture
def served_pipeline():
    fake = FakeSGLang()
    server = PipelineHTTPServer(("127.0.0.1", 0), make_pipeline(fake))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", fake
    server.shutdown()
    server.server_close()


def post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def test_server_answers_one_request_with_one_response(served_pipeline):
    url, fake = served_pipeline
    body = post_json(
        f"{url}/answer",
        {"context": LONG_CONTEXT, "question": "why?", "template": TEMPLATE},
    )
    assert body["answer"] == f"answer for {body['prompt_tokens']} tokens"
    assert body["selected_chunk_indices"] == [1, 5]
    assert len(fake.payloads) == 3
    with urllib.request.urlopen(f"{url}/health", timeout=10) as response:
        assert json.loads(response.read()) == {"status": "ok"}


def test_server_rejects_malformed_requests(served_pipeline):
    url, fake = served_pipeline
    for payload in (
        {"question": "why?"},
        {"context": "a", "question": "why?", "top_k": 3},
        {"context": "a", "question": "why?", "template": "no slots"},
    ):
        with pytest.raises(urllib.error.HTTPError) as error:
            post_json(f"{url}/answer", payload)
        assert error.value.code == 400
    assert fake.payloads == []


def test_server_reports_sglang_failures_as_bad_gateway(served_pipeline):
    url, fake = served_pipeline

    fake.error = RuntimeError("SGLang returned HTTP 500: boom")
    with pytest.raises(urllib.error.HTTPError) as error:
        post_json(f"{url}/answer", {"context": SHORT_CONTEXT, "question": "why?"})
    assert error.value.code == 502
    assert "boom" in json.loads(error.value.read())["error"]

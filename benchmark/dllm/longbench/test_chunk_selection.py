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


def test_qa_f1():
    assert MODULE.qa_f1_score("The blue whale", "blue whale") == 1.0
    assert MODULE.qa_f1_score("red fox", "blue whale") == 0.0

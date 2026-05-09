import math
import os
import re
from collections import Counter


_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def path_distance(path_from, path_to):
    parts_from = os.path.normpath(path_from).split(os.sep)
    parts_to = os.path.normpath(path_to).split(os.sep)
    common = sum(1 for a, b in zip(parts_from, parts_to) if a == b)
    return (len(parts_from) - common) + (len(parts_to) - common)


def _tokenize_for_bm25(text):
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_PATTERN.findall(text)]


class SimpleBM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.doc_freqs = []
        self.doc_lens = []
        self.idf = {}

        n_docs = len(corpus_tokens)
        if n_docs == 0:
            self.avgdl = 0.0
            return

        df = Counter()
        total_len = 0
        for tokens in corpus_tokens:
            counts = Counter(tokens)
            self.doc_freqs.append(counts)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len
            for term in counts:
                df[term] += 1

        self.avgdl = total_len / n_docs if n_docs else 0.0
        for term, freq in df.items():
            self.idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens):
        if not self.corpus_tokens:
            return []
        scores = [0.0] * len(self.corpus_tokens)
        if not query_tokens:
            return scores

        query_tf = Counter(query_tokens)
        for term, qtf in query_tf.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, tf in enumerate(self.doc_freqs):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                doc_len = self.doc_lens[idx]
                denom = freq + self.k1 * (
                    1.0 - self.b + self.b * doc_len / max(self.avgdl, 1e-8)
                )
                scores[idx] += idf * (freq * (self.k1 + 1.0) / denom) * qtf
        return scores


def build_bm25_query(
    completion_filepath,
    completion_content,
    completion_lines=None,
    query_max_lines=128,
    query_max_chars=4000,
):
    lines = completion_content.split("\n") if completion_content else []
    candidate_line_nums = []
    if isinstance(completion_lines, dict):
        for line_nums in completion_lines.values():
            if not line_nums:
                continue
            candidate_line_nums.extend(int(x) for x in line_nums if isinstance(x, int))

    if candidate_line_nums:
        safe_end = max(0, min(candidate_line_nums))
    else:
        safe_end = len(lines)

    safe_end = min(safe_end, len(lines), query_max_lines)
    prefix = "\n".join(lines[:safe_end])
    query = f"{completion_filepath}\n{prefix}".strip()
    if query_max_chars > 0 and len(query) > query_max_chars:
        query = query[-query_max_chars:]
    return query


def build_repo_context_path_distance(snap, completion_filepath=None):
    filenames = snap.get("filename", [])
    contents = snap.get("content", [])
    pairs = list(zip(filenames, contents))
    if completion_filepath:
        pairs.sort(key=lambda x: -path_distance(completion_filepath, x[0]))
    parts = [f"# path: {fn}\n{ct}" for fn, ct in pairs]
    return "\n\n".join(parts), {
        "mode": "path_distance",
        "selected_files": [fn for fn, _ in pairs],
    }


def build_repo_file_segments(
    snap,
    ordered_filenames=None,
    completion_filepath=None,
):
    filenames = snap.get("filename", [])
    contents = snap.get("content", [])
    by_name = {filename: content for filename, content in zip(filenames, contents)}

    if ordered_filenames is not None:
        ordered_pairs = [
            (filename, by_name[filename])
            for filename in ordered_filenames
            if filename in by_name
        ]
    else:
        ordered_pairs = list(zip(filenames, contents))
        if completion_filepath:
            ordered_pairs.sort(key=lambda x: -path_distance(completion_filepath, x[0]))

    return [f"# path: {filename}\n{content}" for filename, content in ordered_pairs]


def build_repo_context_bm25(
    snap,
    completion_filepath,
    completion_content,
    completion_lines,
    topk_files=0,
    fill_window=False,
    target_prompt_chars=0,
    query_max_lines=128,
    query_max_chars=4000,
):
    if topk_files <= 0 and not fill_window:
        fill_window = True

    filenames = snap.get("filename", [])
    contents = snap.get("content", [])
    pairs = list(zip(filenames, contents))
    if not pairs:
        return "", {
            "mode": "bm25",
            "query": "",
            "selected_files": [],
            "scores": [],
        }

    query = build_bm25_query(
        completion_filepath=completion_filepath,
        completion_content=completion_content,
        completion_lines=completion_lines,
        query_max_lines=query_max_lines,
        query_max_chars=query_max_chars,
    )
    query_tokens = _tokenize_for_bm25(query)

    documents = []
    document_tokens = []
    for filename, content in pairs:
        doc_text = f"{filename}\n{content}"
        documents.append((filename, content, doc_text))
        document_tokens.append(_tokenize_for_bm25(doc_text))

    bm25 = SimpleBM25(document_tokens)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(
        [
            {
                "filename": filename,
                "content": content,
                "score": score,
            }
            for (filename, content, _), score in zip(documents, scores)
        ],
        key=lambda x: (x["score"], -len(x["content"])),
        reverse=True,
    )

    selected = []
    current_chars = 0
    for record in ranked:
        if topk_files > 0 and len(selected) >= topk_files:
            break

        rendered = f"# path: {record['filename']}\n{record['content']}"
        rendered_chars = len(rendered) + (2 if selected else 0)

        if fill_window and target_prompt_chars > 0 and selected:
            if current_chars + rendered_chars > target_prompt_chars:
                break

        selected.append(record)
        current_chars += rendered_chars

        if fill_window and target_prompt_chars > 0 and current_chars >= target_prompt_chars:
            break

    if not selected and ranked:
        selected = [ranked[0]]
        current_chars = len(f"# path: {ranked[0]['filename']}\n{ranked[0]['content']}")

    parts = [f"# path: {rec['filename']}\n{rec['content']}" for rec in selected]
    return "\n\n".join(parts), {
        "mode": "bm25",
        "query": query,
        "selected_files": [rec["filename"] for rec in selected],
        "scores": [
            {"filename": rec["filename"], "score": rec["score"]}
            for rec in selected
        ],
        "num_selected_files": len(selected),
        "selected_context_chars": current_chars,
    }

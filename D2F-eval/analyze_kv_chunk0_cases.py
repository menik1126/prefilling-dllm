import json


PRED_PATH = "/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/results_infinitebench_preruntime_keepfirst_128k_20260420_kv_20/dream_infinitebench_kv_retrieval_n20_predictions.jsonl"
DATA_PATH = "/home/ma-user/work/InfiniteBench/data/kv_retrieval.jsonl"


def main():
    with open(PRED_PATH, "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        examples = [json.loads(next(f)) for _ in range(20)]

    for idx in [0, 1, 2]:
        ex = examples[idx]
        pred = preds[idx]
        answer = ex["answer"][0] if isinstance(ex["answer"], list) else ex["answer"]
        answer = str(answer)
        lines = ex["input"].splitlines()
        key_line = next((line for line in lines if line.startswith("Key:")), ex["input"].strip())
        key = key_line.split(":", 1)[1].strip() if ":" in key_line else key_line
        chunk0 = ex["context"][:1200]

        print(
            json.dumps(
                {
                    "idx": idx,
                    "prediction": pred["prediction"][:240],
                    "answer": answer,
                    "key_line": key_line,
                    "answer_in_chunk0": answer in chunk0,
                    "key_in_chunk0": key in chunk0,
                    "pair_snippet_present": f"{key}: \"{answer}\"" in chunk0,
                    "chunk0_preview": chunk0[:500].replace("\n", " "),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

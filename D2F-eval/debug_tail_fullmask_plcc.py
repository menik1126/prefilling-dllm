import gc
import json
from pathlib import Path

from datasets import load_dataset

from d2f_model import generate, load_model
from eval_d2f_plcc import build_repo_context, filter_top_percent, total_context_chars


BASE = Path("/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval")
PRED_PATH = (
    BASE
    / "results_plcc_ctx8k_on_nlq_tailsplit_top3_20260407"
    / "dream_plcc_medium_context_top30pct_predictions.jsonl"
)
TARGETS = [
    ("LAION-AI__Open-Assistant", "backend/oasst_backend/api/v1/management.py", "inproject", 113),
    ("LAION-AI__Open-Assistant", "backend/oasst_backend/api/v1/management.py", "inproject", 27),
    ("bczsalba__pytermgui", "pytermgui/widgets/layouts.py", "inproject", 358),
    ("prowler-cloud__prowler", "lib/outputs/outputs_test.py", "inproject", 225),
    ("tomerfi__aioswitcher", "src/aioswitcher/api/remotes.py", "inproject", 54),
]


def load_baseline(path: Path):
    rows = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            repo = obj["repo"]
            cfile = obj["completion_file"]
            for cat, preds in obj["predictions"].items():
                for pred in preds:
                    rows[(repo, cfile, cat, pred["line_num"])] = pred
    return rows


def main():
    baseline = load_baseline(PRED_PATH)
    ds = load_dataset(
        "JetBrains-Research/lca-project-level-code-completion",
        "medium_context",
        split="test",
    )
    subset = filter_top_percent(list(ds), 30, total_context_chars)
    item_map = {
        (item.get("repo", ""), item.get("completion_file", {}).get("filename", "")): item
        for item in subset
    }

    selected = []
    for repo, cfile, cat, line_num in TARGETS:
        item = item_map[(repo, cfile)]
        snap = item.get("repo_snapshot", {})
        cf = item.get("completion_file", {})
        cf_content = cf.get("content", "")
        cf_line_list = cf_content.split("\n")
        repo_ctx = build_repo_context(snap, completion_filepath=cfile)
        prompt = repo_ctx + f"\n\n# path: {cfile}\n" + "\n".join(cf_line_list[:line_num])
        max_prompt_chars = (8192 - 128) * 6
        if len(prompt) > max_prompt_chars:
            prompt = prompt[-max_prompt_chars:]
        gt = cf_line_list[line_num] if line_num < len(cf_line_list) else ""
        selected.append(
            {
                "key": (repo, cfile, cat, line_num),
                "prompt": prompt,
                "gt": gt,
                "baseline": baseline[(repo, cfile, cat, line_num)],
            }
        )

    print("selected_examples", len(selected), flush=True)
    for ex in selected:
        print(
            "BASELINE",
            ex["key"],
            "em=",
            ex["baseline"]["em"],
            "pred=",
            repr(ex["baseline"]["pred"]),
            flush=True,
        )

    model = load_model(
        "dream",
        pretrained=str(BASE / "model_weights/Dream-v0-Base-7B"),
        lora_path=str(BASE / "model_weights/D2F_Dream_Base_7B_Lora"),
        rope_scale_factor=1.0,
        max_new_tokens=128,
        max_length=8192,
        block_size=32,
        temperature=0,
        add_bos_token=True,
        parallelcomp_mode=True,
        parallelcomp_cache_compress_mode=True,
        parallelcomp_chunk_size=1024,
        parallelcomp_topk_chunks=3,
        parallelcomp_min_prompt_tokens=1,
        parallelcomp_split_from_tail=True,
        parallelcomp_fixed_query_text="Please complete the preceding code.",
        parallelcomp_tail_replay_full_mask=True,
    )

    outputs = generate(model, [ex["prompt"] for ex in selected], stop_tokens=["\n"])
    print("=== FULL MASK RESULTS ===", flush=True)
    for ex, out in zip(selected, outputs):
        pred = out.split("\n")[0]
        em = int(pred.strip() == ex["gt"].strip())
        print("---", flush=True)
        print("KEY", ex["key"], flush=True)
        print("GT ", repr(ex["gt"]), flush=True)
        print("OLD", repr(ex["baseline"]["pred"]), "EM=", ex["baseline"]["em"], flush=True)
        print("NEW", repr(pred), "EM=", em, flush=True)

    del model
    gc.collect()


if __name__ == "__main__":
    main()

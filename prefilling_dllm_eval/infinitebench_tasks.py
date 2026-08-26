import json
import os
import re
from pathlib import Path


SUPPORTED_TASKS = (
    "passkey",
    "number_string",
    "kv_retrieval",
    "longbook_choice_eng",
    "math_find",
    "code_debug",
)

TASK_TO_FILENAME = {
    "passkey": "passkey.jsonl",
    "number_string": "number_string.jsonl",
    "kv_retrieval": "kv_retrieval.jsonl",
    "longbook_choice_eng": "longbook_choice_eng.jsonl",
    "math_find": "math_find.jsonl",
    "code_debug": "code_debug.jsonl",
}

TASK_TO_MAX_NEW_TOKENS = {
    "passkey": 6,
    "number_string": 12,
    "kv_retrieval": 50,
    "longbook_choice_eng": 40,
    "math_find": 3,
    "code_debug": 5,
}

TASK_TO_PARALLELCOMP_MAX_NEW_TOKENS = {
    "passkey": 32,
    "number_string": 32,
    "kv_retrieval": 100,
    "longbook_choice_eng": 32,
    "math_find": 32,
    "code_debug": 32,
}

PROMPT_TEMPLATES = {
    "slot_fill": {
        "passkey": (
            "There is an important info hidden inside a lot of irrelevant text. "
            "Find it and memorize it.\n\n"
            "{context}\n\n"
            "{input}\n\n"
            "pass_key ="
        ),
        "number_string": (
            "There is an important info hidden inside a lot of irrelevant text. "
            "Find it.\n\n"
            "{context}\n\n"
            "{input}\n\n"
            "sequence ="
        ),
        "kv_retrieval": (
            "Extract the value corresponding to the specified key in the JSON object below.\n\n"
            "{context}\n\n{input}"
        ),
        "longbook_choice_eng": (
            "Read the book and answer the question.\n\n"
            "{context}\n\n"
            "Question: {question}\n"
            "A. {OPTION_A}\n"
            "B. {OPTION_B}\n"
            "C. {OPTION_C}\n"
            "D. {OPTION_D}\n\n"
            "answer ="
        ),
        "math_find": "{prefix}\n\n{context}\n\n{input}",
        "code_debug": (
            "Following is a Python code where exactly one of the functions/methods "
            "has a deliberate error that makes it crash.\n\n"
            "{context}\n\n"
            "Options:\n"
            "A. {OPTION_A}\n"
            "B. {OPTION_B}\n"
            "C. {OPTION_C}\n"
            "D. {OPTION_D}\n\n"
            "answer ="
        ),
    },
    "yarn-mistral": {
        "passkey": (
            "There is an important info hidden inside a lot of irrelevant text. "
            "Find it and memorize it. I will quiz you about the important information.\n\n"
            "{context}\n\n{input}\n\nThe pass key is"
        ),
        "number_string": (
            "There is an important info hidden inside a lot of irrelevant text. "
            "Find it. I will quiz you about the important information there.\n\n"
            "{context}\n\n{input}\n\nThe sequence of digits is"
        ),
        "kv_retrieval": (
            "Extract the value corresponding to the specified key in the JSON object below.\n\n"
            "{context}\n\n{input}"
        ),
        "longbook_choice_eng": (
            "Read the book and answer the question.\n\n"
            "{context}\n\n"
            "Question: {question}\n"
            "A. {OPTION_A}\n"
            "B. {OPTION_B}\n"
            "C. {OPTION_C}\n"
            "D. {OPTION_D}\n\n"
            "The letter of the correct answer is"
        ),
        "math_find": "{prefix}\n\n{context}\n\n{input}",
        "code_debug": (
            "Following is a Python code where exactly one of the functions/methods "
            "has a deliberate error that makes it crash.\n\n"
            "{context}\n\n"
            "Options:\n"
            "A. {OPTION_A}\n"
            "B. {OPTION_B}\n"
            "C. {OPTION_C}\n"
            "D. {OPTION_D}\n\n"
            "The correct option is:"
        ),
    },
    "gpt4": {
        "passkey": (
            "There is an important info hidden inside a lot of irrelevant text. "
            "Find it and memorize them. I will quiz you about the important information there.\n\n"
            "{context}\n\n{input}"
        ),
        "number_string": (
            "There is an important info hidden inside a lot of irrelevant text. "
            "Find it. I will quiz you about the important information there.\n\n"
            "{context}\n\n{input}"
        ),
        "kv_retrieval": (
            "Extract the value corresponding to the specified key in the JSON object below.\n\n"
            "{context}\n\n{input}"
        ),
        "longbook_choice_eng": (
            "Read the book and answer the question.\n\n"
            "{context}\n\n"
            "Question: {question}\n\n"
            "Only one of the following options is correct, tell me the answer using "
            "one single letter (A, B, C, or D). Don't say anything else.\n"
            "A. {OPTION_A}\n"
            "B. {OPTION_B}\n"
            "C. {OPTION_C}\n"
            "D. {OPTION_D}"
        ),
        "math_find": "{prefix}\n\n{context}\n\n{input}",
        "code_debug": (
            "There is ONLY ONE function in the large project that is deliberately "
            "made to include an obvious error. Please find the function that "
            "contains the most obvious errors. I will give you four options to "
            "narrow your scope. You can inspect the options and think. Eventually, "
            "tell me the answer using one single letter (A, B, C, or D).\n\n"
            "{context}\n\n"
            "Which funtion has deliberate error?\n"
            "A. {OPTION_A}\n"
            "B. {OPTION_B}\n"
            "C. {OPTION_C}\n"
            "D. {OPTION_D}\n\n"
            "You should first find the functions in the options. Repeat their "
            "content, inspect through code, and at last give me your answer for "
            "the function that has the deliberate and obvious error in A, B, C, or D."
        ),
    },
    "parallelcomp_raw": {
        "passkey": (
            "There is an important info hidden inside a lot of irrelevant text. "
            "Find it and memorize them. I will quiz you about the important information there.\n\n"
            "{context}\n\n{input}"
        ),
        "number_string": (
            "There is an important info hidden inside a lot of irrelevant text. "
            "Find it. I will quiz you about the important information there.\n\n"
            "{context}\n\n{input}"
        ),
        "kv_retrieval": (
            "Extract the value corresponding to the specified key in the JSON object below.\n\n"
            "{context}\n\n{input}"
        ),
        "longbook_choice_eng": (
            "Read the book and answer the question.\n\n"
            "{context}\n\n"
            "Question: {input}\n\n"
            "Only one of the following options is correct, tell me the answer using "
            "one single letter (A, B, C, or D). Don't say anything else.\n"
            "A. {OPTION_A}\n"
            "B. {OPTION_B}\n"
            "C. {OPTION_C}\n"
            "D. {OPTION_D}"
        ),
        "math_find": "{prefix}\n\n{context}\n\n{input}",
        "code_debug": (
            "There is ONLY ONE function in the large project that is deliberately "
            "made to include an obvious error. Please find the function that "
            "contains the most obvious errors. I will give you four options to "
            "narrow your scope. You can inspect through the options and think. "
            "Eventually, tell me the answer using one single letter (A, B, C, or D).\n\n"
            "{context}\n\n"
            "Which funtion has deliberate error?\n"
            "A. {OPTION_A}\n"
            "B. {OPTION_B}\n"
            "C. {OPTION_C}\n"
            "D. {OPTION_D}\n\n"
            "You should first find the functions in the options. Repeat their "
            "content, inspect through code, and at last give me your answer for "
            "the function that has the deliberate and obvious error in A, B, C, or D."
        ),
    },
}


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_data_dir(data_dir=None):
    candidates = []
    if data_dir:
        candidates.append(Path(data_dir))
    env_value = os.environ.get("INFINITEBENCH_DATA_DIR")
    env_dir = Path(env_value) if env_value else None
    if env_dir is not None:
        candidates.append(env_dir)
    candidates.extend(
        [
            Path("/home/ma-user/work/InfiniteBench/data"),
            Path(__file__).resolve().parent.parent / "InfiniteBench" / "data",
            Path(__file__).resolve().parent / "InfiniteBench" / "data",
        ]
    )

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_dir() and any(
            (candidate / TASK_TO_FILENAME[task]).exists() for task in SUPPORTED_TASKS
        ):
            return candidate
    raise FileNotFoundError(
        "InfiniteBench data directory with task JSONL files not found. "
        "Pass --data_dir or set INFINITEBENCH_DATA_DIR."
    )


def load_task_examples(task, data_dir, max_examples=None):
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    task_path = Path(data_dir) / TASK_TO_FILENAME[task]
    if not task_path.exists():
        raise FileNotFoundError(f"InfiniteBench file not found: {task_path}")
    examples = list(iter_jsonl(task_path))
    if max_examples is not None and max_examples > 0:
        examples = examples[:max_examples]
    return examples


def create_prompt(example, task, prompt_style):
    templates = PROMPT_TEMPLATES[prompt_style]
    template = templates[task]

    if task == "longbook_choice_eng":
        return template.format(
            input=example["input"],
            question=example["input"],
            context=example["context"],
            OPTION_A=example["options"][0],
            OPTION_B=example["options"][1],
            OPTION_C=example["options"][2],
            OPTION_D=example["options"][3],
        )
    if task == "code_debug":
        return template.format(
            context=example["context"],
            OPTION_A=example["options"][0],
            OPTION_B=example["options"][1],
            OPTION_C=example["options"][2],
            OPTION_D=example["options"][3],
        )
    if task == "math_find":
        prompt = example["input"]
        match = re.findall(r"The .+ of", prompt)
        if not match:
            raise ValueError(f"Cannot parse math_find prompt: {prompt}")
        prefix = f"What is {match[0].lower()[:-3]} in the following list?"
        return template.format(prefix=prefix, context=example["context"], input=prompt)
    return template.format(context=example["context"], input=example["input"])


def create_prompt_parts(example, task, prompt_style):
    templates = PROMPT_TEMPLATES[prompt_style]
    template = templates[task]
    context_sentinel = "__INFINITEBENCH_CONTEXT_SENTINEL__"

    if task == "longbook_choice_eng":
        rendered = template.format(
            input=example["input"],
            question=example["input"],
            context=context_sentinel,
            OPTION_A=example["options"][0],
            OPTION_B=example["options"][1],
            OPTION_C=example["options"][2],
            OPTION_D=example["options"][3],
        )
    elif task == "code_debug":
        rendered = template.format(
            context=context_sentinel,
            OPTION_A=example["options"][0],
            OPTION_B=example["options"][1],
            OPTION_C=example["options"][2],
            OPTION_D=example["options"][3],
        )
    elif task == "math_find":
        prompt = example["input"]
        match = re.findall(r"The .+ of", prompt)
        if not match:
            raise ValueError(f"Cannot parse math_find prompt: {prompt}")
        prefix = f"What is {match[0].lower()[:-3]} in the following list?"
        rendered = template.format(prefix=prefix, context=context_sentinel, input=prompt)
    else:
        rendered = template.format(context=context_sentinel, input=example["input"])

    if context_sentinel not in rendered:
        raise ValueError(
            f"Prompt template for task={task} style={prompt_style} is missing a context slot"
        )

    prefix, query = rendered.split(context_sentinel, 1)
    scoring_query = example.get("input", "")
    if task == "longbook_choice_eng":
        scoring_query = (
            f"Question: {example['input']}\n"
            f"A. {example['options'][0]}\n"
            f"B. {example['options'][1]}\n"
            f"C. {example['options'][2]}\n"
            f"D. {example['options'][3]}"
        )
    elif task == "code_debug":
        scoring_query = (
            "Which function has deliberate error?\n"
            f"A. {example['options'][0]}\n"
            f"B. {example['options'][1]}\n"
            f"C. {example['options'][2]}\n"
            f"D. {example['options'][3]}"
        )

    return {
        "prefix": prefix,
        "context": example["context"],
        "query": query,
        "scoring_query": scoring_query,
    }


def _first_int_match(prediction):
    parts = re.split(r"[^0-9]", prediction)
    for part in parts:
        if part:
            return part
    return ""


def _normalize_label_list(label):
    if isinstance(label, list):
        return label
    return [label]


def score_passkey(prediction, label):
    label = _normalize_label_list(label)[0]
    return _first_int_match(prediction) == str(label)


def score_number_string(prediction, label):
    label = _normalize_label_list(label)[0]
    return _first_int_match(prediction) == str(label)


def score_kv_retrieval(prediction, label):
    label = _normalize_label_list(label)[0]
    for c in ['\n', ':', '"', "'", ".", ",", "?", "!", "{", "}"]:
        prediction = prediction.replace(c, " ")
    return str(label) in prediction.split()


def score_longbook_choice_eng(prediction, label):
    valid_labels = {str(x) for x in _normalize_label_list(label)}
    pred = prediction.strip()
    match = re.search(r"\b[A-D]\b(?!.*\b[A-D]\b)", pred)
    if match and match.group(0) in valid_labels:
        return True
    if pred and pred[0] in "ABCD":
        return pred[0] in valid_labels
    if pred in valid_labels:
        return True
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    for prefix in ("answer is:", "answer:", "answer is", "option is"):
        idx = pred.find(prefix)
        if idx == -1:
            continue
        after_prefix = pred[idx + len(prefix) + 1 :]
        return any(after_prefix.startswith(x) for x in valid_labels)
    return any(word in valid_labels for word in pred.split())


def score_math_find(prediction, label):
    label = _normalize_label_list(label)[0]
    match = re.search(r"\d+\.\d+|\d+", prediction)
    if match is None:
        return False
    found = match.group(0).strip()
    if isinstance(label, int):
        return int(found) == label
    if isinstance(label, float):
        return float(found) == label
    try:
        return float(found) == float(label)
    except Exception:
        return found == str(label)


def score_code_debug(prediction, label):
    pred = prediction.strip()
    fn_name, label_c = label[0], label[1]
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", pred)
    if match and match.group(0) == label_c:
        return True
    for c in ["\n", "`", "'", '"', "-", "*", "Option", "option"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    if pred.startswith(label_c) or pred.startswith(fn_name):
        return True
    for prefix in ("answer is:", "is:", "answer:", "correct option is:"):
        idx = pred.find(prefix)
        if idx == -1:
            continue
        after_prefix = pred[idx + len(prefix) + 1 :]
        return after_prefix.startswith(label_c) or after_prefix.startswith(fn_name)
    return False


TASK_SCORERS = {
    "passkey": score_passkey,
    "number_string": score_number_string,
    "kv_retrieval": score_kv_retrieval,
    "longbook_choice_eng": score_longbook_choice_eng,
    "math_find": score_math_find,
    "code_debug": score_code_debug,
}


def score_prediction(task, prediction, label):
    return bool(TASK_SCORERS[task](prediction, label))


def normalize_answer_label(task, example):
    if task not in {"longbook_choice_eng", "code_debug"}:
        return example["answer"]

    options = example["options"]
    answer = example["answer"]
    option_letters = "ABCD"

    if isinstance(answer, str):
        return [answer, option_letters[options.index(answer)]]
    if isinstance(answer, list):
        if len(answer) == 1:
            return [answer[0], option_letters[options.index(answer[0])]]
        if len(answer) == 2 and answer[1] in option_letters:
            return answer
    raise ValueError(f"Unsupported answer format for {task}: {answer}")

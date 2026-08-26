from datasets import load_dataset
from torch.utils.data import DataLoader,Dataset
from peft import PeftModel, PeftConfig, get_peft_model
# from modelscope.msdatasets import MsDataset
import torch
import json
import re


ROLE_ALIASES = {
    "human": "user",
    "gpt": "assistant",
}


def _normalise_messages(messages):
    """Convert LongAlign's conversational records to tokenizer chat roles."""
    normalised = []
    for message in messages:
        role = ROLE_ALIASES.get(message["role"], message["role"])
        content = message.get("content", "")
        if role in {"system", "user", "assistant"} and content:
            normalised.append({"role": role, "content": content})
    return normalised


class LongAlignCollator:
    """Create dynamically padded full-diffusion SFT batches from LongAlign messages."""

    def __init__(self, tokenizer, max_length, max_answer_tokens):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_answer_tokens = max_answer_tokens

    def _tokenize_example(self, example):
        messages = _normalise_messages(example["messages"])
        answer_index = next(
            (index for index in range(len(messages) - 1, -1, -1)
             if messages[index]["role"] == "assistant"),
            None,
        )
        if answer_index is None:
            raise ValueError("LongAlign example has no assistant response.")

        prompt_messages = messages[:answer_index]
        answer = messages[answer_index]["content"]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).input_ids[0]
        answer_ids = self.tokenizer(
            answer, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]

        answer_limit = min(self.max_answer_tokens, self.max_length)
        answer_ids = answer_ids[:answer_limit]
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("tokenizer does not define eos_token_id")
        if answer_ids.numel() == 0 or answer_ids[-1].item() != eos_token_id:
            if answer_ids.numel() == answer_limit:
                answer_ids[-1] = eos_token_id
            else:
                answer_ids = torch.cat(
                    [answer_ids, torch.tensor([eos_token_id], dtype=answer_ids.dtype)]
                )

        prompt_budget = self.max_length - answer_ids.numel()
        if prompt_budget <= 0:
            prompt_ids = prompt_ids[:0]
        else:
            # Preserve the query at the end of a long prompt.
            prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = torch.cat([prompt_ids, answer_ids])
        return input_ids, prompt_ids.numel()

    def __call__(self, batch):
        encoded_examples = [self._tokenize_example(example) for example in batch]
        sequence_lengths = torch.tensor(
            [input_ids.numel() for input_ids, _ in encoded_examples], dtype=torch.long
        )
        question_lengths = torch.tensor(
            [prompt_length for _, prompt_length in encoded_examples], dtype=torch.long
        )
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        batch_width = sequence_lengths.max().item()
        padded = torch.full(
            (len(encoded_examples), batch_width),
            pad_token_id,
            dtype=torch.long,
        )
        for row, (input_ids, _) in enumerate(encoded_examples):
            padded[row, :input_ids.numel()] = input_ids
        return {
            "data": padded,
            "question_length": question_lengths,
            "sequence_length": sequence_lengths,
        }


def get_longalign_dataloader(tokenizer, config, max_length):
    """Load LongAlign's public conversational SFT data lazily."""
    global_config = getattr(config, "_parent", config)
    dataset_path = global_config.paths.data.longalign
    split = global_config.paths.data.get("longalign_split", "train")
    dataset = load_dataset(dataset_path, split=split)
    max_answer_tokens = config.get("max_answer_tokens", max_length)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=LongAlignCollator(
            tokenizer=tokenizer,
            max_length=max_length,
            max_answer_tokens=max_answer_tokens,
        ),
        num_workers=config.get("num_workers", 0),
        shuffle=True,
        pin_memory=True,
    )
def extract_answer(text):
    pattern = r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        solution_content = match.group(1).strip()
        # print("Extracted content:\n")
        # print(solution_content)
        return solution_content
    else:
        # print("No matching content found.")
        return None
def collate_fn(batch, tokenizer, max_length):
    """
    batch: list of raw text samples (str)
    tokenizer: huggingface tokenizer
    max_length: maximum length to pad to (int)
    """
    encoded_batch = []
    for text in batch:
        # Encode text, return dictionary, note no automatic padding
        enc = tokenizer(text["text"], add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)  # (seq_len,)

        # Add eos_token_id
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("tokenizer does not have eos_token_id")

        input_ids = torch.cat([input_ids, torch.tensor([eos_id], device=input_ids.device)])

        # Padding to max_length
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer does not have pad_token_id")

        seq_len = input_ids.size(0)
        if seq_len > max_length:
            # Truncate if too long
            input_ids = input_ids[:max_length]
        else:
            # Pad right side if not long enough
            pad_len = max_length - seq_len
            padding = torch.full((pad_len,), pad_id, device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, padding])

        encoded_batch.append(input_ids)

    return torch.stack(encoded_batch)

def prepare_dataloader(data, tokenizer, batch_size, max_length):
    dataset = CustomDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size  = batch_size,
        collate_fn  = lambda x: collate_fn(x, tokenizer, max_length=max_length),
        num_workers = 0,
        shuffle     = True,
        pin_memory  = True,
    )

    return dataloader

def read_math():
    math_data = []
    dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    for item in dataset:
        math_data.append({"question": item['question'], "answer": item['answer']})
    return math_data

def read_python():
    python_data = []
    dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    for item in dataset:
        python_data.append({"question": item['question'], "answer": item['answer']})
    return python_data

def read_numinamath():
    math_data = read_math()
    python_data = read_python()
    return math_data + python_data

def read_bs(config=None):
    data=[]
    # Get path from config, use default path if no config
    if config and hasattr(config, 'paths') and hasattr(config.paths, 'data') and hasattr(config.paths.data, 'bs'):
        dataset_path = config.paths.data.bs
    else:
        dataset_path = "/data1/xck/dllm_block_wx/data/Lansechen/bs17k_collection_filtered_hard_maxlength600"
    
    dataset=load_dataset(dataset_path, split="train")
    for item in dataset:
        data.append({"question": item['question'], "answer": item['qwen7b_answer']})
    return data

def read_bs_easy(config=None):
    data=[]
    # Get path from config, use default path if no config
    if config and hasattr(config, 'paths') and hasattr(config.paths, 'data') and hasattr(config.paths.data, 'bs_easy'):
        dataset_path = config.paths.data.bs_easy
    else:
        dataset_path = "/data1/xck/dllm_block_wx/data/Lansechen/bs17k_collection_filtered_easy_maxlength600"
    
    dataset=load_dataset(dataset_path, split="train")
    for item in dataset:
        data.append({"question": item['question'], "answer": item['qwen7b_answer']})
    return data

def read_bs_17k():
    data=[]
    dataset=load_dataset("/data/wx/dataset/bespokelabs/Bespoke-Stratos-17k",split="train")
    for item in dataset:
        item=item["conversations"]
        data.append({"question": item[0]['value'], "answer": extract_answer(item[1]['value'])})
    return data
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
def read_llada(file_path="/home/wx/dllm_block/data/merged_bs17k_easy_hard_llada_collected.jsonl"):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError:
                print(f'JSONDecodeError: {line}')
    return data
def get_bs17k_dataloader(tokenizer, config, max_length=1024):
    train_dataset = []
    # Pass global config to data reading functions
    global_config = getattr(config, '_parent', config)  # Try to get parent config
    data_dict=read_bs(global_config)+read_bs_easy(global_config)
    for data in data_dict:
        question = data['question']
        answer = data['answer']

        # messages = [
        #     {"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"},
        # ]
        messages = [
            {"role": "user", "content": question}
        ]
        question = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).input_ids[0]

        # question = tokenizer(question, return_tensors='pt')['input_ids'][0]
        answer = tokenizer(answer, return_tensors='pt')['input_ids'][0]
        answer = torch.cat((answer, torch.tensor([tokenizer.eos_token_id])), dim=-1)

        question_length = question.shape[-1]
        answer_length = answer.shape[-1]
        combined_length = question_length + answer_length
        if question_length > max_length-100:
            continue 
        if combined_length > max_length:
            padded_data = torch.cat((question, answer), dim=-1)
            padded_data = padded_data[:max_length]  # Truncate to max_length
        else:
            padding_length = max_length - combined_length
            padding = torch.full((padding_length,), tokenizer.eos_token_id, dtype=question.dtype)
            padded_data = torch.cat((question, answer, padding), dim=-1)

        train_dataset.append(
            dict(
                data = padded_data,
                question_length = question_length,
                length = combined_length,
            )
        )

    dataset = CustomDataset(train_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size  = config.batch_size,
        num_workers = 0,
        shuffle     = True,
        pin_memory  = True,
    )

    return dataloader

# def get_gsm8k_dataloader(tokenizer, config, max_length=1024):
#     train_dataset = []
#     data_dict = read_numinamath()
#     for data in data_dict:
#         question = data['question']
#         answer = data['answer']

#         question = tokenizer(question, return_tensors='pt')['input_ids'][0]
#         answer = tokenizer(answer, return_tensors='pt')['input_ids'][0]
#         answer = torch.cat((answer, torch.tensor([tokenizer.eos_token_id])), dim=-1)

#         question_length = question.shape[-1]
#         answer_length = answer.shape[-1]
#         combined_length = question_length + answer_length

#         if combined_length > max_length:
#             continue

#         padding_length = max_length - combined_length
#         padding = torch.full((padding_length,), tokenizer.eos_token_id, dtype=question.dtype)
#         padded_data = torch.cat((question, answer, padding), dim=-1)

#         train_dataset.append(
#             dict(
#                 data = padded_data,
#                 question_length = question_length,
#                 length = combined_length,
#             )
#         )

#     dataset = CustomDataset(train_dataset)
#     dataloader = DataLoader(
#         dataset,
#         batch_size  = config.batch_size,
#         collate_fn  = lambda x: collate_fn_pad(x, tokenizer, max_length=max_length),
#         num_workers = 0,
#         shuffle     = True,
#         pin_memory  = True,
#     )

#     return dataloader
def get_llada_bs17k_dataloader(tokenizer, config, max_length=1024):
    train_dataset = []
    # Pass global config to data reading functions
    global_config = getattr(config, '_parent', config)  # Try to get parent config
    data_dict = read_bs(global_config)
    python_dict=read_bs_easy(global_config)
    data_dict=data_dict+python_dict
    print("Data length:",len(data_dict))
    # data_dict = read_llada()
    for data in data_dict:
        question = data['question']
        answer = data['answer']

        # messages = [
        #     {"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"},
        # ]
        messages = [
            {"role": "user", "content": question}
        ]
        question = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).input_ids[0]

        # question = tokenizer(question, return_tensors='pt')['input_ids'][0]
        answer = tokenizer(answer, return_tensors='pt')['input_ids'][0]
        answer = torch.cat((answer, torch.tensor([126348])), dim=-1)

        question_length = question.shape[-1]
        answer_length = answer.shape[-1]
        combined_length = question_length + answer_length

        if combined_length > max_length:
            continue

        padding_length = max_length - combined_length
        padding = torch.full((padding_length,), tokenizer.eos_token_id, dtype=question.dtype)
        padded_data = torch.cat((question, answer, padding), dim=-1)

        train_dataset.append(
            dict(
                data = padded_data,
                question_length = question_length,
                length = combined_length,
            )
        )

    dataset = CustomDataset(train_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size  = config.batch_size,
        num_workers = 0,
        shuffle     = True,
        pin_memory  = True,
    )

    return dataloader
if __name__ == "__main__":
      text="<|begin_of_thought|>\n\nOkay, let me try to figure out this problem. So, we have this operation defined as a⊗b = a²/b. And we need to compute [(1⊗2)⊗3] - [1⊗(2⊗3)]. Then choose the correct answer from the options given. Alright, let's break it down step by step.\n\nFirst, I need to remember that the operation ⊗ is not associative, right? Because the problem is asking for the difference between two different groupings: (1⊗2)⊗3 and 1⊗(2⊗3). So, the order in which we perform the operations matters here. That's probably why there's a subtraction between them.\n\nLet me start by computing each part separately. Let's tackle the first part: (1⊗2)⊗3.\n\nStarting with the innermost operation, which is 1⊗2. According to the definition, a⊗b = a²/b. So here, a is 1 and b is 2. Plugging those in: 1² / 2 = 1/2. So, 1⊗2 equals 1/2.\n\nNow, we take that result and perform the next operation with 3. So, (1⊗2)⊗3 becomes (1/2)⊗3. Again, using the same definition: a is now 1/2 and b is 3. So, ( (1/2)² ) / 3 = (1/4) / 3 = 1/12. So, (1⊗2)⊗3 equals 1/12.\n\nAlright, that's the first part. Now let's compute the second part: 1⊗(2⊗3). Again, starting with the innermost operation, which is 2⊗3. Applying the definition: a is 2 and b is 3. So, 2² / 3 = 4/3. Therefore, 2⊗3 equals 4/3.\n\nNow, we need to compute 1⊗(4/3). Here, a is 1 and b is 4/3. Using the operation definition: 1² / (4/3) = 1 / (4/3) = 3/4. So, 1⊗(2⊗3) equals 3/4.\n\nNow, the problem asks for the difference between the two results: [(1⊗2)⊗3] - [1⊗(2⊗3)] = (1/12) - (3/4). To subtract these fractions, they need a common denominator. The denominators are 12 and 4, so 12 is the common denominator.\n\nConverting 3/4 to twelfths: 3/4 = 9/12. So, 1/12 - 9/12 = (1 - 9)/12 = -8/12. Simplifying that fraction by dividing numerator and denominator by 4: -8/12 = -2/3.\n\nHmm, looking at the answer choices, option A is -2/3. So, is that the answer? Wait, but let me double-check my calculations to make sure I didn't make a mistake somewhere.\n\nFirst, checking (1⊗2): 1² / 2 = 1/2. Correct. Then, (1/2)⊗3: (1/2)² / 3 = (1/4)/3 = 1/12. That seems right.\n\nNow, for 2⊗3: 2² / 3 = 4/3. Correct. Then, 1⊗(4/3): 1² / (4/3) = 1 / (4/3) = 3/4. Yes, that's correct.\n\nSubtracting 3/4 from 1/12: 1/12 - 3/4. Convert 3/4 to 9/12, so 1/12 - 9/12 = -8/12 = -2/3. Yes, that all checks out. So the answer should be -2/3, which is option A.\n\nWait, but let me think again. The operation is defined for all nonzero numbers, so we don't have any issues with division by zero here. 2⊗3 is 4/3, which is fine, and then 1⊗(4/3) is 3/4. Correct.\n\nAlternatively, maybe there's a different way to approach the problem? Let me try expanding both expressions using variables to see if there's a pattern.\n\nLet's denote the first expression: (a⊗b)⊗c. Using the definition:\n\nFirst, compute a⊗b = a²/b.\n\nThen, take that result and ⊗ with c: (a²/b)⊗c = ( (a²/b)² ) / c = a⁴ / (b² c).\n\nNow, the second expression: a⊗(b⊗c). First compute b⊗c = b²/c.\n\nThen, a⊗(b²/c) = a² / (b²/c) = a² * (c / b²) = (a² c) / b².\n\nTherefore, the difference between the two expressions is:\n\n(a⁴ / (b² c)) - (a² c / b²) = (a⁴ - a² c²) / (b² c) = a² (a² - c²) / (b² c).\n\nHmm, factoring that, it's a² (a - c)(a + c) / (b² c).\n\nBut in our specific problem, a = 1, b = 2, c = 3. Plugging those values in:\n\n1² (1 - 3)(1 + 3) / (2² * 3) = 1 * (-2)(4) / (4 * 3) = (-8) / 12 = -2/3. Same result. So that confirms the answer is indeed -2/3.\n\nTherefore, I think my initial calculation was correct, and the answer is option A.\n\n**Final Answer**\n\\boxed{A}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\nTo determine the value of \\([(1 \\otimes 2) \\otimes 3] - [1 \\otimes (2 \\otimes 3)]\\) where the operation \\(\\otimes\\) is defined by \\(a \\otimes b = \\frac{a^2}{b}\\), we proceed as follows:\n\nFirst, compute \\(1 \\otimes 2\\):\n\\[\n1 \\otimes 2 = \\frac{1^2}{2} = \\frac{1}{2}\n\\]\nNext, use this result to compute \\((1 \\otimes 2) \\otimes 3\\):\n\\[\n\\left(\\frac{1}{2}\\right) \\otimes 3 = \\frac{\\left(\\frac{1}{2}\\right)^2}{3} = \\frac{\\frac{1}{4}}{3} = \\frac{1}{12}\n\\]\n\nNow, compute \\(2 \\otimes 3\\):\n\\[\n2 \\otimes 3 = \\frac{2^2}{3} = \\frac{4}{3}\n\\]\nThen, use this result to compute \\(1 \\otimes (2 \\otimes 3)\\):\n\\[\n1 \\otimes \\left(\\frac{4}{3}\\right) = \\frac{1^2}{\\frac{4}{3}} = \\frac{1}{\\frac{4}{3}} = \\frac{3}{4}\n\\]\n\nFinally, find the difference between the two results:\n\\[\n\\frac{1}{12} - \\frac{3}{4} = \\frac{1}{12} - \\frac{9}{12} = \\frac{1 - 9}{12} = \\frac{-8}{12} = -\\frac{2}{3}\n\\]\n\nThus, the answer is \\(\\boxed{A}\\).\n\n<|end_of_solution|>"
      print(extract_answer(text))

def get_dataloader_by_config(tokenizer, config, global_config=None, max_length=1024):
    """Select different data loaders based on config file"""
    if global_config is None:
        global_config = config
    
    training_mode = global_config.get('training_mode', 'dream')
    
    # Add reference to global config for data loading functions to access
    config._parent = global_config
    
    if training_mode == 'llada':
        return get_llada_bs17k_dataloader(tokenizer, config, max_length)
    elif training_mode == 'dream':
        return get_bs17k_dataloader(tokenizer, config, max_length)
    elif training_mode == 'dream_full_long':
        return get_longalign_dataloader(tokenizer, config, max_length)
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")

from torch.utils.data import Dataset
from tqdm import tqdm
import json


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
            self,
            dataset,
            tokenizer,
            strategy,
            input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]


_common_model_template = {
    "qwen2-math": "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
                  "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
}


class PromptLabelDataset(Dataset):
    def __init__(self,
                 file_path,
                 tokenizer,
                 strategy,
                 input_template=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy

        self.data = json.load(open(file_path, "r"))
        self.input_template = _common_model_template[input_template]

        new_data = []
        for item in self.data:
            input_key = getattr(self.strategy.args, "question_key", "prompt")
            prompt = self.input_template.replace("{prompt}", item[input_key])
            label_key = getattr(self.strategy.args, "label_key", "label")
            new_data.append({
                "prompt": prompt,
                "question": item[input_key],
                "label": item[label_key],
            })
        self.data = new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

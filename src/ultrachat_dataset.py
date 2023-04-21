from openprompt.data_utils.utils import InputExample
import os
import json
from typing import *

from openprompt.data_utils.data_processor import DataProcessor

import torch
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm

from transformers.tokenization_utils import PreTrainedTokenizer
import copy

class UltraChatProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = None

    def get_examples(self, data_path: str, tokenizer) -> List[InputExample]:
        examples = []
        j = 0
        with open(data_path) as f:
            for line in tqdm(f.readlines()):
                if line.strip():
                    data = json.loads(line)
                    id_ = data["id"]
                    dialogue = data["data"]
                    tags = [i for _ in range(len(dialogue)//2) for i in ["### User", "### Assistant"]]
                    for i in range(0, len(dialogue), 2):
                        tgt_text = dialogue[i+1]+tokenizer.eos_token
                        context = dialogue[:i+1]
                        context = zip(tags[:i+1], context)
                        context = [": ".join(item) for item in context]
                        example = InputExample(guid=str(j), text_a="", tgt_text=tgt_text, meta={"context": context})
                        examples.append(example)
                        j += 1
        return examples


    def get_src_tgt_len_ratio(self,):
        pass
    
IGNORE_INDEX=-100


def collator(tokenizer, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class PromptIterableDataset(IterableDataset):
    def __init__(self,
                 raw_dataset: Union[Dataset, List],
                 template: Text,
                 tokenizer: PreTrainedTokenizer = None,
                 max_seq_length: Optional[int] = 512,
                 teacher_forcing: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                ):
        assert hasattr(raw_dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {raw_dataset}"
        assert hasattr(raw_dataset, "__len__"), f"The dataset must have __len__ method. dataset is {raw_dataset}"
        self.raw_dataset = raw_dataset
        self.template = template
        self.teacher_forcing = teacher_forcing

        self.tokenizer = tokenizer
        self.truncate_method = truncate_method
        self.max_seq_length = max_seq_length
        assert self.truncate_method == "head", print("only head truncate support")

    def tokenize_example(self, example):
        chat_history = "\n\n".join(example.meta["context"])
        src_txt = self.template.format(chat_history)
        tgt_txt = example.tgt_text

        tokenized_src = self.tokenizer(src_txt, return_tensors="pt")

        input_len = len(tokenized_src["input_ids"][0])

        tokenized_example = self.tokenizer(src_txt + " " + tgt_txt, return_tensors="pt")
        tokenized_example = {k:v[0] for k, v in tokenized_example.items()}

        labels = copy.deepcopy(tokenized_example["input_ids"])
        labels[:input_len] = IGNORE_INDEX

        return {**tokenized_example, "labels": labels}

    def truncate(self, tokenized_example):
        old_len = len(tokenized_example["input_ids"])
        if old_len > self.max_seq_length:
            for k in tokenized_example:
                tokenized_example[k] = tokenized_example[k][old_len - self.max_seq_length:]

        return tokenized_example


    def __iter__(self):
        for example in self.raw_dataset:
            tokenized_example = self.tokenize_example(example)
            tokenized_example = self.truncate(tokenized_example)
            yield tokenized_example

    def __len__(self):
        return len(self.raw_dataset)




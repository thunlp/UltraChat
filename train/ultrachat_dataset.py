from openprompt.data_utils.utils import InputExample
import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

from openprompt.utils.logging import logger
from openprompt.data_utils.data_processor import DataProcessor

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

class UltraChatProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = None

    def get_examples(self, data_path: str) -> List[InputExample]:
        examples = []
        j = 0
        with open(data_path) as f:
            for line in tqdm(f.readlines()):
                if line.strip():
                    data = json.loads(line)
                    id_ = data["id"]
                    dialogue = data["data"]
                    tags = [i for _ in range(len(dialogue)//2) for i in ["User", "Assistant"]]
                    for i in range(0, len(dialogue), 2):
                        tgt_text = dialogue[i+1]
                        context = dialogue[:i+1]
                        context = zip(tags[:i+1], context)
                        context = [": ".join(item) for item in context]
                        example = InputExample(guid=str(j), text_a="", tgt_text=tgt_text, meta={"context": context})
                        examples.append(example)
                        j += 1
        return examples


    def get_src_tgt_len_ratio(self,):
        pass

if __name__ == "__main__":
    processor = UltraChatProcessor()
    dataset = processor.get_examples("./data/ultrachat_release_230407.json")
    print(dataset[0])
    print(dataset[5])


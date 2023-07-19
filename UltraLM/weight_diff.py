#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Optional, Dict

import fire
import torch
import tqdm
import transformers
# from train import smart_tokenizer_and_embedding_resize

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@torch.inference_mode()
def make_diff(
    path_raw: str, path_tuned: str, path_diff: str, device="cuda",  # "cuda" or "cpu"
):
    """Make the weight diff.

    This function is given to present full transparency of how the weight diff was created.

    Run:
        python weight_diff.py make_diff --path_raw <your_path_raw> --path_tuned <your_path_tuned> --path_diff <your_path_diff>
    """
    print("load tuned model...")
    model_tuned: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_tuned,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    print("load raw model...")
    model_raw: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    tokenizer_tuned: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_tuned
    )

    tokenizer_raw: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_raw
    )
    if tokenizer_raw.pad_token is None and tokenizer_tuned.pad_token is not None:
        print("add pad token")
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="<PAD>"),
            model=model_raw,
            tokenizer=tokenizer_raw,
        )
    state_dict_tuned = model_tuned.state_dict()
    state_dict_raw = model_raw.state_dict()
    for key in tqdm.tqdm(state_dict_tuned, desc="make diff"):
        print(key)
        state_dict_tuned[key].add_(-state_dict_raw[key])
    print("saving diff weights...")
    model_tuned.save_pretrained(path_diff)
    kwargs = {"push_to_hub": False, "repo_id": ""}
    tokenizer_tuned.save_pretrained(path_diff, **kwargs)


@torch.inference_mode()
def recover(
    path_raw,
    path_diff,
    path_tuned: Optional[str] = None,
    device="cpu",
    test_inference=True,
    check_integrity_naively=True,
):
    model_raw: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model_recovered: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_diff,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    tokenizer_raw: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_raw
    )
    tokenizer_recovered: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_diff
    )
    if tokenizer_raw.pad_token is None and tokenizer_recovered.pad_token is not None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="<PAD>"),
            model=model_raw,
            tokenizer=tokenizer_raw,
        )
    tokenizer_recovered: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_diff
    )

    state_dict_recovered = model_recovered.state_dict()
    state_dict_raw = model_raw.state_dict()
    for key in tqdm.tqdm(state_dict_recovered):
        state_dict_recovered[key].add_(state_dict_raw[key])

    if path_tuned is not None:
        model_recovered.save_pretrained(path_tuned)
        tokenizer_recovered.save_pretrained(path_tuned)

    if test_inference:
        input_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\r\n\r\n"
            "### Instruction:\r\nList three technologies that make life easier.\r\n\r\n### Response:"
        )
        inputs = tokenizer_recovered(input_text, return_tensors="pt")
        out = model_recovered.generate(inputs=inputs.input_ids, max_new_tokens=100)
        output_text = tokenizer_recovered.batch_decode(out, skip_special_tokens=True)[0]
        output_text = output_text[len(input_text) :]
        print(f"User: {input_text}\nAssistant: {output_text}")

    return model_recovered, tokenizer_recovered


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
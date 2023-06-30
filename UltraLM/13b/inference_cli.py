import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import argparse
from util.inference import generate_stream, SimpleChatIO

def get_model_tokenizer(model_name_or_path):
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float16)
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def chat_loop(
    model,
    tokenizer,
    system_prompt: str,
    temperature: float = 0.7,
    max_new_tokens: int = 2000,
    chatio = SimpleChatIO(),
    device = "cuda",
    debug: bool = False
):
    conv = [system_prompt]

    while True:
        try:
            inp = chatio.prompt_for_input("User")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append("User: " + inp.strip() + tokenizer.eos_token)

        
        prompt = "\n".join(conv) + "\nAssistant: "

        gen_params = {
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "echo": False,
        }

        chatio.prompt_for_output("Assistant")
        with torch.inference_mode():
            output_stream = generate_stream(model, tokenizer, gen_params, device)
        outputs = chatio.stream_output(output_stream)
        # NOTE: strip is important to align with the training data.
        conv.append("Assistant: " + outputs.strip() + tokenizer.eos_token)

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/path/to/ultralm")
    args = parser.parse_args()

    model, tokenizer = get_model_tokenizer(args.model_path)

    system_prompt = "User: A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, very detailed, and polite answers to the user's questions.</s>"

    chat_loop(model, tokenizer, system_prompt)

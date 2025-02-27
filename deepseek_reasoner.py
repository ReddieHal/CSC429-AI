import argparse
import os
import time
import json
from tqdm import tqdm

from openai import OpenAI

from utils import SYS_INST, PROMPT_INST, PROMPT_INST_COT, ONESHOT_ASSISTANT, ONESHOT_USER, TWOSHOT_USER, TWOSHOT_ASSISTANT

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

def get_openrouter_chat(prompt, args):
    if args.fewshot_eg:
        messages = [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": ONESHOT_USER},
            {"role": "assistant", "content": ONESHOT_ASSISTANT},
            {"role": "user", "content": TWOSHOT_USER},
            {"role": "assistant", "content": TWOSHOT_ASSISTANT},
            {"role": "user", "content": prompt["prompt"]}
        ]
    else:
        messages = [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": prompt["prompt"]}
        ]
    
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=messages,
            max_tokens=args.max_gen_length,
            temperature=args.temperature,
            seed=args.seed,
            extra_body={},
            stream=False
        )
        return response.choices[0].message.content, None  # OpenRouter may not support logprobs
    except Exception as error:
        print(f"API Error: {error}. Retrying in 5 seconds...")
        time.sleep(5)
        return get_openrouter_chat(prompt, args)

def construct_prompts(input_file, inst):
    with open(input_file, "r") as f:
        samples = f.readlines()
    samples = [json.loads(sample) for sample in samples]
    prompts = []
    for sample in samples:
        key = sample["project"] + "_" + sample["commit_id"]
        p = {"sample_key": key}
        p["func"] = sample["func"]
        p["target"] = sample["target"]
        p["prompt"] = inst.format(func=sample["func"])
        prompts.append(p)
    return prompts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="deepseek/deepseek-r1:free", help='OpenRouter model name')
    parser.add_argument('--prompt_strategy', type=str, choices=["std_cls", "cot"], default="standard", help='Prompt strategy')
    parser.add_argument('--data_path', type=str, help='Data path')
    parser.add_argument('--output_folder', type=str, help='Output folder')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_gen_length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--fewshot_eg', action="store_true", help='Use few-shot examples')
    args = parser.parse_args()

    output_file = os.path.join(args.output_folder, f"{args.model}_{args.prompt_strategy}_fewshoteg{args.fewshot_eg}.jsonl")
    if args.prompt_strategy == "std_cls":
        inst = PROMPT_INST
    elif args.prompt_strategy == "cot":
        inst = PROMPT_INST_COT
    else:
        raise ValueError("Invalid prompt strategy")
    prompts = construct_prompts(args.data_path, inst)

    with open(output_file, "w") as f:
        print(f"Requesting {args.model} to respond to {len(prompts)} prompts ...")
        for p in tqdm(prompts):
            response, _ = get_openrouter_chat(p, args)
            if response is None:
                response = "ERROR"
            p["response"] = response
            f.write(json.dumps(p))
            f.write("\n")
            f.flush()

if __name__ == "__main__":
    main()


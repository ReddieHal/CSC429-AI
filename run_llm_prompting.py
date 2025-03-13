import argparse
import os
import time
import json
from tqdm import tqdm

import openai
from openai import OpenAI
from openai._types import NOT_GIVEN

from utils import SYS_INST, PROMPT_INST, PROMPT_INST_COT, ONESHOT_ASSISTANT, ONESHOT_USER, TWOSHOT_USER, TWOSHOT_ASSISTANT

import tiktoken

# Configure client to use OpenRouter instead of OpenAI
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)


def truncate_tokens_from_messages(messages, model, max_gen_length):
    """
    Count the number of tokens used by a list of messages, 
    and truncate the messages if the number of tokens exceeds the limit.
    Reference: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    # Map model to context limits for OpenRouter models
    # These are approximate and may need adjustment
    if "gpt-3.5" in model:
        max_tokens = 16385 - max_gen_length
    elif "gpt-4" in model:
        max_tokens = 128000 - max_gen_length
    elif "claude" in model:
        max_tokens = 100000 - max_gen_length
    elif "deepseek" in model:
        max_tokens = 32768 - max_gen_length
    elif "gemini" in model:
        max_tokens = 32000 - max_gen_length
    elif "qwen" in model or "qwq" in model:
        max_tokens = 32000 - max_gen_length
    elif "llama-3" in model:
        max_tokens = 32000 - max_gen_length
    elif "dolphin" in model or "mistral" in model:
        max_tokens = 32000 - max_gen_length
    else:
        # Default case
        max_tokens = 8192 - max_gen_length
        
    # Most open models use cl100k_base encoding or something compatible
    encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens_per_message = 3

    num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
    trunc_messages = []
    for message in messages:
        tm = {}
        num_tokens += tokens_per_message
        for key, value in message.items():
            encoded_value = encoding.encode(value)
            num_tokens += len(encoded_value)
            if num_tokens > max_tokens:
                print(f"Truncating message: {value[:100]}...")
                tm[key] = encoding.decode(encoded_value[:max_tokens - num_tokens])
                break
            else:
                tm[key] = value
        trunc_messages.append(tm)
    return trunc_messages


# Updated to work with OpenRouter
def get_openai_chat(
    prompt,
    args
):
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
    
    # count the number of tokens in the prompt
    messages = truncate_tokens_from_messages(messages, args.model, args.max_gen_length)
    
    # get response from OpenRouter
    try:
        # Define parameters that all models support
        params = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_gen_length,
            "temperature": args.temperature,
        }
        
        # Add seed parameter if the model supports it
        # Some models like older Llama versions might not support seed
        if args.seed is not None and not ("llama-2" in args.model.lower() or "dolphin" in args.model.lower()):
            params["seed"] = args.seed
            
        # Add logprobs only for models that support it (mainly OpenAI-based)
        if args.logprobs and ("openai" in args.model.lower() or "gpt" in args.model.lower()):
            params["logprobs"] = args.logprobs
            params["top_logprobs"] = 5
        
        response = client.chat.completions.create(**params)
        
        response_content = response.choices[0].message.content
        
        # Handle logprobs if available and requested
        log_prob_mapping = {}
        if args.logprobs and hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            try:
                response_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                for topl in response_logprobs:
                    log_prob_mapping[topl.token] = topl.logprob
            except (AttributeError, IndexError, TypeError):
                print("Logprobs requested but not available in response")
        
        return response_content, log_prob_mapping

    # Handle errors with OpenRouter
    except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as error:
        retry_time = error.retry_after if hasattr(error, "retry_after") else 5
        print(f"Rate Limit or Connection Error. Sleeping for {retry_time} seconds ...")
        time.sleep(retry_time)
        return get_openai_chat(
            prompt,
            args,
        )
    except openai.BadRequestError as error:
        print(f"Bad Request Error: {error}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None


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
    parser.add_argument('--model', type=str, default="openai/gpt-3.5-turbo", 
                        help='Model name (OpenRouter format)')
    parser.add_argument('--prompt_strategy', type=str, choices=["std_cls", "cot"], default="standard", help='Prompt strategy')
    parser.add_argument('--data_path', type=str, help='Data path')
    parser.add_argument('--output_folder', type=str, help='Output folder')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_gen_length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--logprobs', action="store_true", help='Return logprobs (only works with OpenAI models)')
    parser.add_argument('--fewshot_eg', action="store_true", help='Use few-shot examples')
    parser.add_argument('--data', type=str, default="", help='Data descriptor for output naming')
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    # Sanitize model name for filenames (replace / and : with -)
    safe_model_name = args.model.replace('/', '-').replace(':', '-')
    output_file = os.path.join(args.output_folder, f"{safe_model_name}_{args.prompt_strategy}_logprobs{args.logprobs}_fewshoteg{args.fewshot_eg}.jsonl")
    
    if args.prompt_strategy == "std_cls":
        inst = PROMPT_INST
    elif args.prompt_strategy == "cot":
        inst = PROMPT_INST_COT
    else:
        raise ValueError("Invalid prompt strategy")
    prompts = construct_prompts(args.data_path, inst)

    with open(output_file, "w") as f:
        print(f"Requesting {args.model} to respond to {len(prompts)} {args.data} prompts ...")
        for p in tqdm(prompts):
            response, logprobs = get_openai_chat(p, args)
            if logprobs:
                p["logprobs"] = logprobs
            if response is None:
                response = "ERROR"
            p["response"] = response
            f.write(json.dumps(p))
            f.write("\n")
            f.flush()


if __name__ == "__main__":
    main()

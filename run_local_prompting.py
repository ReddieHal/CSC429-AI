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

# Initialize OpenAI client to connect to Ollama
def get_openai_client(args):
    # Use the base_url parameter to point to your Ollama instance
    # Default Ollama API endpoint is typically http://localhost:11434/api
    return OpenAI(
        base_url=args.base_url,
        # No API key needed for Ollama by default, but can be passed if configured
        api_key=args.api_key or "ollama"  # Using "ollama" as a placeholder key
    )


def truncate_tokens_from_messages(messages, model, max_gen_length):
    """
    Count the number of tokens used by a list of messages, 
    and truncate the messages if the number of tokens exceeds the limit.
    Reference: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    # Adjust token limits based on specific Ollama models
    if "llama" in model.lower():
        max_tokens = 4096 - max_gen_length  # Adjust based on your model's context window
    elif "mistral" in model.lower():
        max_tokens = 8192 - max_gen_length
    elif "deepseek" in model.lower():
        max_tokens = 8192 - max_gen_length
    else:
        # Default token limit for other models
        max_tokens = 4096 - max_gen_length
    
    # Ollama models may not directly map to OpenAI's tiktoken encodings
    # This is a best-effort approximation
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Fallback for non-OpenAI models
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
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


# get completion from an Ollama model via OpenAI client
def get_ollama_chat(
    prompt,
    args,
    client
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
        # select the correct in-context learning prompt based on the task
        messages = [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": prompt["prompt"]}
            ]
    
    # count the number of tokens in the prompt
    messages = truncate_tokens_from_messages(messages, args.model, args.max_gen_length)
    
    # get response from Ollama via OpenAI client
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_gen_length,
            temperature=args.temperature,
            # Ollama may not support all OpenAI parameters, so we handle them conditionally
            seed=args.seed if args.use_seed else NOT_GIVEN,
            # Ollama may not support logprobs in the same way OpenAI does
            logprobs=args.logprobs if args.use_logprobs else NOT_GIVEN,
            top_logprobs=5 if args.logprobs and args.use_logprobs else NOT_GIVEN,
            )
        
        response_content = response.choices[0].message.content
        
        # Handle logprobs if available
        response_logprobs = None
        log_prob_mapping = {}
        if args.logprobs and args.use_logprobs and hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            try:
                response_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                for topl in response_logprobs:
                    log_prob_mapping[topl.token] = topl.logprob
            except (AttributeError, IndexError):
                print("Logprobs not available in the response format")
        
        # Small delay to prevent overwhelming the local Ollama server
        time.sleep(0.1)
        return response_content, log_prob_mapping

    # Error handling for Ollama API
    except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as error:
        retry_time = getattr(error, "retry_after", 5)
        print(f"Rate Limit or Connection Error. Sleeping for {retry_time} seconds ...")
        time.sleep(retry_time)
        return get_ollama_chat(prompt, args, client)
    
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
    parser.add_argument('--model', type=str, default="llama3", help='Model name in Ollama (e.g., llama3, mistral, mistral-openorca)')
    parser.add_argument('--prompt_strategy', type=str, choices=["std_cls", "cot"], default="standard", help='Prompt strategy')
    parser.add_argument('--data_path', type=str, help='Data path')
    parser.add_argument('--output_folder', type=str, help='Output folder')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_gen_length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--use_seed', action="store_true", help='Use seed parameter (may not be supported by all Ollama models)')
    parser.add_argument('--logprobs', action="store_true", help='Return logprobs')
    parser.add_argument('--use_logprobs', action="store_true", help='Use logprobs parameter (may not be supported by all Ollama models)')
    parser.add_argument('--fewshot_eg', action="store_true", help='Use few-shot examples')
    parser.add_argument('--base_url', type=str, default="http://localhost:11434/v1", help='Ollama API base URL')
    parser.add_argument('--api_key', type=str, default=None, help='API key if required')
    parser.add_argument('--data', type=str, default="test", help='Data type identifier')
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    output_file = os.path.join(args.output_folder, f"{args.model}_{args.prompt_strategy}_logprobs{args.logprobs}_fewshoteg{args.fewshot_eg}.jsonl")
    if args.prompt_strategy == "std_cls":
        inst = PROMPT_INST
    elif args.prompt_strategy == "cot":
        inst = PROMPT_INST_COT
    else:
        raise ValueError("Invalid prompt strategy")
    
    prompts = construct_prompts(args.data_path, inst)
    
    # Initialize OpenAI client with Ollama base URL
    client = get_openai_client(args)

    with open(output_file, "w") as f:
        print(f"Requesting {args.model} to respond to {len(prompts)} {args.data} prompts via Ollama backend...")
        for p in tqdm(prompts):
            response, logprobs = get_ollama_chat(p, args, client)
            if logprobs:
                p["logprobs"] = logprobs
                print(logprobs)
            if response is None:
                response = "ERROR"
            p["response"] = response
            f.write(json.dumps(p))
            f.write("\n")
            f.flush()


if __name__ == "__main__":
    main()
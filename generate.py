import os
import re
import time
import json
import argparse


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_file", type=str, default="data/charthal.json")
    args.add_argument("--model_id", type=str, required=True)
    args.add_argument("--model_type", type=str, required=True)
    args.add_argument("--mode", type=str, default="transformers", choices=["transformers", "vllm"])
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--save_dir", type=str, default="results")
    args = args.parse_args()

    if args.save_dir is None:
        args.save_dir = "."

    model_name = args.model_id.split("/")[-1]
    resp_dir = os.path.join(args.save_dir, model_name)
    os.makedirs(resp_dir, exist_ok=True)
    save_file = os.path.join(resp_dir, "response.json")

    # open dataset file
    with open(args.data_file, "r", encoding='utf-8') as f:
        eval_data = json.load(f)

    # generate response
    if args.model_type == "qwen25vl":
        if args.mode == "vllm":
            from models.qwen25vl import generate_resp_qwen25vl_vllm
            eval_results = generate_resp_qwen25vl_vllm(args.model_id, eval_data)
        else:
            from models.qwen25vl import generate_resp_qwen25vl
            eval_results = generate_resp_qwen25vl(args.model_id, eval_data)
    elif args.model_type == "qwen3vl":
        if args.mode == "vllm":
            from models.qwen3vl import generate_resp_qwen3vl_vllm
            eval_results = generate_resp_qwen3vl_vllm(args.model_id, eval_data)
        else:
            from models.qwen3vl import generate_resp_qwen3vl
            eval_results = generate_resp_qwen3vl(args.model_id, eval_data)
    elif args.model_type == "internvl25":
        if args.mode == "vllm":
            print("ERROR: InternVL2.5 VLLM generation not implemented yet. Use transformers mode.")
        from models.internvl25 import generate_resp_internvl25
        eval_results = generate_resp_internvl25(args.model_id, eval_data)
    elif args.model_type == "llama32v":
        if args.mode == "vllm":
            print("ERROR: Llama32V VLLM generation not implemented yet. Use transformers mode.")
        from models.llama32v import generate_resp_llama32v
        eval_results = generate_resp_llama32v(args.model_id, eval_data)
    else:
        print(f"ERROR: Unknown model type: {args.model_type}")
        exit(0)

    # save result file
    with open(save_file, "w", encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {save_file}")

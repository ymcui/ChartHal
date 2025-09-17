import os
import re
import time
import json
import argparse
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import EVAL_RESP_MAP, EVAL_SUFFIX

def parse_score(text: str) -> int:
    head = "\n".join(text.splitlines()[:3])  # only inspect the first few lines
    m = re.search(r'(?i)\bscore\b\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)', head) \
        or re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', head) \
        or re.search(r'(?<!\d)(10|[0-9])(?!\d)', head)
    if m:
        try:
            return int(round(max(0, min(10, float(m.group(1))))))
        except Exception:
            pass
    return -1

def build_prompt(item: dict):
    key = f"{item.get('q_type')}_{item.get('q_relation')}"
    tmpl = EVAL_RESP_MAP.get(key)
    if not tmpl:
        return None, f"Request failed after: eval template not found for key '{key}'"
    prompt = (tmpl
              .replace("<|QUESTION|>", item.get("question", ""))
              .replace("<|REF_ANS|>", item.get("ref_answer", ""))
              .replace("<|RESPONSE|>", item.get("response", "")))
    prompt += EVAL_SUFFIX
    return prompt, None

def evaluate_resp(
    client: OpenAI,
    model_id: str,
    resp_data,
    batch_size: int = 8,
):
    # gather items needing evaluation (keeps references back to resp_data)
    items = resp_data.values() if isinstance(resp_data, dict) else resp_data
    pending = []
    for it in items:
        need = ("score" not in it) or (it.get("score", -1) == -1) or str(it.get("eval_resp", "")).startswith("Request failed after")
        if need:
            it["eval_resp"] = ""
            it["score"] = -1
            pending.append(it)

    if not pending:
        return resp_data

    def worker(idx: int, sample: dict, retries: int = 2, backoff: float = 1.5):
        last_err = None
        for attempt in range(retries):
            try:
                prompt, err = build_prompt(sample)
                if err:
                    return idx, err
                r = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                )
                return idx, r.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                time.sleep(min((backoff ** attempt), 10.0))
        return idx, f"Request failed after: {last_err}"

    max_workers = max(1, min(int(batch_size), len(pending)))
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, sample in enumerate(pending):
            futures.append(ex.submit(worker, i, sample))

        with tqdm(total=len(futures), ncols=100) as pbar:
            for fut in as_completed(futures):
                idx, eval_text = fut.result()
                pending[idx]["eval_resp"] = eval_text
                s = parse_score(eval_text)
                if s != -1:
                    pending[idx]["score"] = s
                pbar.update(1)

    return resp_data


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--resp_file", type=str, required=True)
    args.add_argument("--model_id", type=str, default='gpt-4o-2024-11-20')
    args.add_argument("--api_key", type=str, default=None)
    args.add_argument("--batch_size", type=int, default=8)
    args = args.parse_args()

    # open response file
    with open(args.resp_file, "r", encoding='utf-8') as f:
        resp_data = json.load(f)

    # init OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY") if args.api_key is None else args.api_key
    client = OpenAI(api_key=api_key)

    # generate response
    eval_results = evaluate_resp(client=client, model_id=args.model_id,
                                 resp_data=resp_data, batch_size=args.batch_size)

    # save eval file
    save_file = args.resp_file.replace("response.json", "eval.json")
    with open(save_file, "w", encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {save_file}")

import os
import re
import time
import json
import argparse
import base64
from io import BytesIO
from openai import OpenAI
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

def encode_image(image: Image.Image):
    assert isinstance(image, Image.Image)
    # Use lossless PNG for best readability on diagrams/plots
    # Preserve alpha if present; otherwise use RGB
    image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "png", base64_image

def generate_resp(client: OpenAI, model_id: str, eval_data: dict, batch_size: int = 8):
    items = eval_data.values() if isinstance(eval_data, dict) else eval_data
    input_data_list = []
    for item in items:
        if "response" not in item or not item["response"] or item["response"].startswith("Request failed"):
            item["response"] = ""
            input_data_list.append(item)

    if not input_data_list:
        return eval_data  # nothing to do

    def worker(idx: int, sample: dict, retries: int = 2, backoff: float = 1.5):
        last_err = None
        for attempt in range(retries):
            try:
                # sanity checks (optional)
                if "figure_path" not in sample or "question" not in sample:
                    return idx, "Request failed: missing 'figure_path' or 'question'"

                with Image.open(sample["figure_path"]) as img:
                    ext_name, base64_image = encode_image(img)

                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": sample["question"]},
                            {"type": "image_url", "image_url": {"url": f"data:image/{ext_name};base64,{base64_image}"}}
                        ]
                    }]
                )
                return idx, resp.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                time.sleep(min((backoff ** attempt), 10.0))
        return idx, f"Request failed: {last_err}"

    max_workers = max(1, min(int(batch_size), len(input_data_list)))
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, sample in enumerate(input_data_list):
            futures.append(ex.submit(worker, i, sample))

        with tqdm(total=len(futures), ncols=100) as pbar:
            for fut in as_completed(futures):
                idx, content = fut.result()
                input_data_list[idx]["response"] = content
                pbar.update(1)

    return eval_data

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_file", type=str, default="data/charthal.json")
    args.add_argument("--model_id", type=str, required=True)
    args.add_argument("--api_key", type=str, default=None)
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--save_dir", type=str, default="results")
    args = args.parse_args()

    if args.save_dir is None:
        args.save_dir = "."

    resp_dir = os.path.join(args.save_dir, args.model_id)
    os.makedirs(resp_dir, exist_ok=True)
    save_file = os.path.join(resp_dir, "response.json")

    # open dataset file
    with open(args.data_file, "r", encoding='utf-8') as f:
        eval_data = json.load(f)

    # init OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY") if args.api_key is None else args.api_key
    client = OpenAI(api_key=api_key)

    # generate response
    eval_results = generate_resp(client=client, model_id=args.model_id,
                                 eval_data=eval_data, batch_size=args.batch_size)

    # save result file
    with open(save_file, "w", encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {save_file}")
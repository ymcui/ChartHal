from PIL import Image
from transformers import Qwen3VLMoeForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import torch

def generate_resp_qwen3vl(model_id: str, eval_data: dict):
    items = eval_data.values() if isinstance(eval_data, dict) else eval_data
    input_data_list = []
    for item in items:
        if "response" not in item or not item["response"] or item["response"].startswith("Request failed"):
            item["response"] = ""
            input_data_list.append(item)

    if not input_data_list:
        return eval_data  # nothing to do

    # init model
    torch.set_grad_enabled(False)
    if ("30b" in model_id.lower()) or ("235b" in model_id.lower()):
        model_loader = Qwen3VLMoeForConditionalGeneration 
    else:
        model_loader = Qwen3VLForConditionalGeneration
    model = model_loader.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    for i in tqdm(range(0, len(input_data_list)), ncols=100):
        sample = input_data_list[i]
        image_path = sample["figure_path"]
        input_prompt = sample["question"]

        messages = [{"role": "user", 
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": input_prompt}
                    ]
        }]

        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        sample["response"] = response.strip()
        
    return eval_data

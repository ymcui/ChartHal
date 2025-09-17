from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import torch

def generate_resp_qwen25vl(model_id: str, eval_data: dict):
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) #, use_fast=True)

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

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        sample["response"] = response.strip()
        
    return eval_data

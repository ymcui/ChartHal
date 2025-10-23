from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration
from tqdm import tqdm
import torch

def generate_resp_llama32v(model_id: str, eval_data: dict):
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
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    for i in tqdm(range(0, len(input_data_list)), ncols=100):
        sample = input_data_list[i]
        image_pil = Image.open(sample["figure_path"])
        input_prompt = sample["question"]

        messages = [{"role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": input_prompt}
                    ]
        }]

        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image_pil, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        sample["response"] = response.strip()
        
    return eval_data

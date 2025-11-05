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
        dtype=torch.bfloat16,
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

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def generate_resp_qwen25vl_vllm(model_id: str, eval_data: dict):
    from vllm import LLM, SamplingParams
    import os
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
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
    processor = AutoProcessor.from_pretrained(model_id)
    model = LLM(
        model=model_id,
        mm_encoder_tp_mode="data",
        max_model_len=16384,
        #enable_expert_parallel=True,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=42
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        top_k=-1,
        stop_token_ids=[],
    )

    all_messages = []
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
        all_messages.append(messages)

    inputs = [prepare_inputs_for_vllm(message, processor) for message in all_messages]
    outputs = model.generate(inputs, sampling_params=sampling_params)

    for i, output in enumerate(outputs):
        sample = input_data_list[i]
        generated_text = output.outputs[0].text
        sample["response"] = generated_text.strip()
        
    return eval_data

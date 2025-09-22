# 🌀 ChartHal

**ChatHal** is a benchmark to comprehensively evaluate hallucination of visual language models (VLMs) in chart understanding. 

We develop a comprehensive taxonomy categorizing various scenarios that trigger model hallucinations and construct a **human-validated** chart comprehension dataset based on this framework. Our experimental results show that current SOTA VLMs generally perform poorly when handling hallucinations in chart understanding, demonstrating that ChatHal is a challenging yet valuable resource for advancing the field.

- 🌀 Project Page: http://ymcui.com/ChartHal/ 
- 🤗 Hugging Face: https://huggingface.co/datasets/hfl/ChartHal

## Dataset

The proposed ChatHal can be accessed under `data` directory. Alternatively, you can also access through HuggingFace Hub.

## Evaluation Guidelines

Please follow the steps below to evaluate the model's performance on the ChatHal. Before that, please first clone the repository.

```bash
git clone https://github.com/ymcui/ChartHal.git && cd ChartHal
```

The evaluation includes three steps: generate response → evaluate response → scoring results.

### Step 1a: Generate response for GPT-series

For proprietory models, we provide `generate_gpt.py` to showcase how to perform inference on GPT-series models (e.g., GPT-4o, o4-mini, GPT-5, etc.).

Dependency:
```bash
pip install openai
```

Run:
```bash
python generate_gpt.py \
    --data_file data/charthal.json \
    --model_id gpt-5-mini \
    --batch_size 16 \
    --api_key your-openai-key \
    --save_dir results
```

Params:
- `data_file`: Path to the input data file (JSON format).
- `model_id`: The official API name of the GPT model to use (e.g., `gpt-5-mini`).
- `batch_size`: The number of samples to process in each batch.
- `api_key`: Your OpenAI API key. Alternatively, you can also store under `OPENAI_API_KEY` environment variable.
- `save_dir`: Directory to save the generated responses.

After the inference, the response file will be saved at `results/gpt-5-mini/response.json`.

> [!NOTE]
> If you want to run other proprietary models or modify the parameter (like `reasoning_effort`), please refer to their specific documentation and modify `generate_gpt.py` accordingly.

### Step 1b: Generate response for Open-source models

Similarly, we provide `generate.py` to showcase how to perform inference on open-source models (e.g., Qwen2.5-VL, InternVL-2.5, etc.).

Dependency (take `Qwen2.5-VL` for example): 
```bash
pip install -U transformers
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

Run:
```bash
python generate.py \
    --data_file data/charthal.json \
    --model_id /content/qwen2.5-vl-7b-inst \
    --model_type qwen25vl \
    --save_dir results
```

Params:
- `data_file`: Path to the input data file (JSON format).
- `model_id`: Path to the open-source model (e.g., `/content/qwen2.5-vl-7b-inst`).
- `model_type`: The type of the model (e.g., `qwen25vl`, `internvl25`, `llama32v`).
- `save_dir`: Directory to save the generated responses.

After the inference, the response file will be saved at `results/qwen2.5-vl-7b-inst/response.json`.

> [!NOTE]
> If you want to run other open-source models, please refer to their specific documentation and add an entry in `generate.py` accordingly. A model-specific generation function file (like `models/qwen25vl.py`) should be implemented in the `models` directory.

### Step 2: Evaluate the response

After getting the response, we can use GPT model to evaluate the quality of the generated responses. In our paper, we use `gpt-4o-2024-11-20` as the evaluation model. **You are strongly recommended to use the same model for consistency.**

```bash
python evaluate.py \
    --resp_file results/gpt-5-mini/response.json \
    --batch_size 16
```

This will evaluate the response of GPT-5-mini (in Step 1a), and save the evaluation results to `results/gpt-5-mini/eval.json`.

> [!WARNING] 
> This step requires to use GPT API. Be aware of the usage limits and costs associated with the API. In our trials, it tooks about 600K input tokens, and <100K output tokens for each run.

### Step 3: Get the final results

After evaluating the responses, you can obtain the final results by aggregating the evaluation metrics. This can be done using the `get_result.py` script.

```bash
python get_result.py results/gpt-5-mini/eval.json
```

The output looks like:
```bash
Scoring results saved to results/gpt-5-mini/score.json
  total_queries: 1062
  total_correct: 287
  overall_score: 27.02
```

Showing the total number of queries, correct responses, and overall score for the evaluated model. For the detailed full results, please check `results/gpt-5-mini/score.json`.


### Citation

If you are using our benchmark in your work, please consider cite:

```bibtex
@article{charthal,
      title={{ChartHal}: A Fine-grained Framework Evaluating Hallucination of Large Vision Language Models in Scientific Chart Understanding}, 
      author={Wang, Xingqi and Cui, Yiming and Yao, Xin and Wang, Shijin and Hu, Guoping and Qin, Xiaoyu},
      year={2025},
      eprint={2509.xxxxx},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Acknowledgment

ChartHal is built upon [CharXiv](http://charxiv.github.io). We sincerely thank the authors for their contributions to the community.

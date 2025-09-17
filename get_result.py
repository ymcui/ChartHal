import os
import json
import re
import sys
from pathlib import Path
from itertools import product

QUESTION_TYPES = ["desc", "reason", "open"]
QUESTION_RELATIONS = ["irrel", "inexist",  "contra", "normal"]

def process_eval(eval_data):
    metrics = {
        "total_queries": 0,
        "total_correct": 0,
        "overall_score": 0,
        "exceptions": 0,
        "score_per_type": {q_type: 0 for q_type in QUESTION_TYPES},
        "score_per_relation": {q_relation: 0 for q_relation in QUESTION_RELATIONS},
        "score_per_type_relation": {f"{q_type}_{q_relation}": 0 for q_type, q_relation in product(QUESTION_TYPES, QUESTION_RELATIONS)},
        "ratio_per_type": {q_type: 0 for q_type in QUESTION_TYPES},
        "ratio_per_relation": {q_relation: 0 for q_relation in QUESTION_RELATIONS},
        "ratio_per_type_relation": {f"{q_type}_{q_relation}": 0 for q_type, q_relation in product(QUESTION_TYPES, QUESTION_RELATIONS)},
    }

    for item in eval_data.values():
        metrics["total_queries"] += 1
        if "score" not in item or item["score"] not in [0, 1]:
            metrics["exceptions"] += 1
            continue

        metrics["total_correct"] += item["score"]
        metrics["score_per_type"][item["q_type"]] += item["score"]
        metrics["score_per_relation"][item["q_relation"]] += item["score"]
        metrics["score_per_type_relation"][f"{item['q_type']}_{item['q_relation']}"] += item["score"]

        metrics["ratio_per_type"][item["q_type"]] += 1
        metrics["ratio_per_relation"][item["q_relation"]] += 1
        metrics["ratio_per_type_relation"][f"{item['q_type']}_{item['q_relation']}"] += 1

    # calculate ratios
    metrics["overall_score"] = round(metrics["total_correct"] / (metrics["total_queries"] - metrics["exceptions"]) * 100, 2)

    # express ratio in "score/total_queries"
    for q_type in metrics["ratio_per_type"]:
        metrics["ratio_per_type"][q_type] = f"{metrics['score_per_type'][q_type]}/{metrics['ratio_per_type'][q_type]}={metrics['score_per_type'][q_type] / metrics['ratio_per_type'][q_type] * 100:.2f}"

    for q_relation in metrics["ratio_per_relation"]:
        metrics["ratio_per_relation"][q_relation] = f"{metrics['score_per_relation'][q_relation]}/{metrics['ratio_per_relation'][q_relation]}={metrics['score_per_relation'][q_relation] / metrics['ratio_per_relation'][q_relation] * 100:.2f}"

    for q_type_relation in metrics["ratio_per_type_relation"]:
        metrics["ratio_per_type_relation"][q_type_relation] = f"{metrics['score_per_type_relation'][q_type_relation]}/{metrics['ratio_per_type_relation'][q_type_relation]}={metrics['score_per_type_relation'][q_type_relation] / metrics['ratio_per_type_relation'][q_type_relation] * 100:.2f}"

    # 将所有的 ratio_per_type 和 ratio_per_relation 百分比指标用 & 连接起来，便于打印 Latex 表格
    all_ratios = [v.split('=')[-1] for v in list(metrics["ratio_per_type"].values()) + list(metrics["ratio_per_relation"].values())]
    all_ratios.append(str(metrics["overall_score"]))
    metrics["latex_line"] = " & ".join(all_ratios)

    tiny_results = {
        "total_queries": metrics["total_queries"],
        "total_correct": metrics["total_correct"],
        "overall_score": metrics["overall_score"],
    }

    return metrics, tiny_results

if __name__ == "__main__":
    eval_file = sys.argv[1]
    save_file = eval_file.replace("eval.json", "score.json")

    # open eval data
    with open(eval_file, "r", encoding='utf-8') as f:
        eval_data = json.load(f)

    # scoring
    scoring_result, tiny_results = process_eval(eval_data)

    # save scoring result
    with open(save_file, "w", encoding='utf-8') as f:
        json.dump(scoring_result, f, ensure_ascii=False, indent=4)
    print(f"Scoring results saved to {save_file}")

    # print tiny results
    #print("Tiny scoring results:")
    for key, value in tiny_results.items():
        print(f"  {key}: {value}")

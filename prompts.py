EVAL_IRREL_QUERY = """You are given a question about a chart and a corresponding response. The question is completely unrelated to the chart. Your task is to evaluate whether the response correctly identifies that the question cannot be answered from the chart. 

Guidelines:
- The response should explicitly indicate that the question is unrelated to the chart.
- It should not attempt to fabricate an answer.
- If the response fails to point out the irrelevance or tries to answer the question, it should be considered incorrect.

Question: <|QUESTION|>
Response: <|RESPONSE|>

Score the response above as follows:
- 1 = Correctly identifies irrelevance and avoids answering.
- 0 = Does not identify irrelevance OR tries to answer the question.
"""

EVAL_INEXIST_CLOSE_QUERY = """You are given a question about a chart and a corresponding response. The question is related to the chart but asks about information that does not exist in the chart. Your task is to evaluate whether the response correctly identifies that the information is missing and that the question cannot be answered from the chart.

Guidelines:
- The response should clearly state that the requested information does not exist in the chart.
- It should not attempt to guess or fabricate an answer.
- If the response ignores the missing information issue or provides an answer, it should be considered incorrect.

Question: <|QUESTION|>
Response: <|RESPONSE|>

Score the response above as follows:
- 1 = Correctly identifies that the information is not in the chart and avoids guessing.
- 0 = Does not identify missing information OR gives a fabricated answer.
"""

EVAL_INEXIST_OPEN_QUERY = """You are given a question about a chart and a corresponding response. The question is related to the chart but asks about information that does not exist in the chart. Your task is to evaluate whether the response correctly identifies that the information is missing and and either states the question is unanswerable or provides a speculative answer.

Guidelines:
- The response should clearly state that the requested information does not exist in the chart.
- It should either state that the question is unanswerable OR provide a speculative answer with an speculative tone.
- If the response ignores the missing information issue and directly provides an definitive answer, it should be considered incorrect.

Question: <|QUESTION|>
Response: <|RESPONSE|>

Score the response above as follows:
- 1 = Correctly identifies that the information is not in the chart and avoids guessing.
- 0 = Does not identify missing information OR gives a fabricated answer.
"""

EVAL_CONTRA_CLOSE_QUERY = """You are given a question about a chart, a corresponding response, and a ground truth answer. The question contains a statement that contradicts the chart, and then asks a follow-up based on that contradiction. Your task is to evaluate whether the response correctly identifies the contradiction and either states the question is unanswerable or provides a correct answer ignoring the contradiction in the question.

Guidelines:
- The response should explicitly point out that the question contains incorrect or contradictory information relative to the chart.
- It should either state that the question is unanswerable OR provide a correct answer that aligns with the given Ground Truth.
    - If the ground truth does not provide an answer, the response should also not provide an answer.
    - When the response tries to provide an answer, for questions expecting numerical answers, if the ground truth specifies an acceptable range (e.g., "Answers in the range of [35%, 40%] are ACCEPTABLE" or similar phrasing), then the answer is considered correct if it falls within this range. Otherwise, the semantic meaning of the answer should match the ground truth without omissions or additional unrelated information.
- If the response fails to mention the contradiction OR answers the question based on the contradictory statement, it is incorrect.

Question: <|QUESTION|>
Ground Truth: <|REF_ANS|>
Response: <|RESPONSE|>

Score the response above as follows:
- 1 = Correctly identifies the contradiction AND either states the question is unanswerable or provides a correct answer that matches the given Ground Truth.
- 0 = Does not mention the contradiction OR answers the question as if it were valid.
"""

EVAL_CONTRA_OPEN_QUERY = """You are given a question about a chart and a corresponding response. The question contains a statement that contradicts the chart, and then asks a follow-up based on that contradiction. Your task is to evaluate whether the response correctly identifies the contradiction and either states the question is unanswerable or provides a tentative answer ignoring the contradiction in the question.

Guidelines:
- The response should explicitly point out that the question contains incorrect or contradictory information relative to the chart.
- It should either state that the question is unanswerable or provide a speculative answer with an appropriately uncertain tone.
- If the response fails to mention the contradiction, answers the question based on the contradictory statement, sounds overconfident OR states an absolute answer, it is incorrect.

Question: <|QUESTION|>
Response: <|RESPONSE|>

Score the response above as follows:
- 1 = Correctly identifies the contradiction AND either states the question is unanswerable or provides a tentative answer with an appropriately uncertain tone.
- 0 = Does not mention the contradiction OR answers the question as if it were valid OR sounds overconfident.
"""

EVAL_NORMAL_CLOSE_QUERY = """You are given a question about a chart, a corresponding response, and a ground truth answer. The question is answerable from the chart and has a definitive correct answer. Your task is to evaluate how accurate the model response is compared to the ground truth.

1. **If the expected answer is related to numeric values:**
   - If the ground truth specifies an acceptable range (e.g., "Answers in the range of [35%, 40%] are ACCEPTABLE" or similar phrasing), then:
     - Give a score of 1 if the response contains a numeric value within this range.
     - Otherwise, give a score of 0.
   - If no range is specified:
     - Give a score of 1 if and only if the two numbers are exactly equal in value. Different notations (e.g., `0.01` and `10^-2`, `1500` and `1.5e3`) are acceptable.
     - Give a score of 0 if the numbers differ in value.

2. **If the expected answer is related to predefined terms:**
   - Give a score of 1 if and only if the final answer and the ground truth refer to the same term(s). Different forms or representations (e.g., α and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m) are acceptable.
   - If the question requires multiple terms, different order is acceptable as long as all terms match exactly.
   - Give a score of 0 if any required term is incorrect, missing, or replaced with a different term (e.g., ACC+ vs ACC; P-101 vs P=101).

3. **For all other cases:**
   - Give a score of 1 only if the semantic meaning of the model response matches the ground truth without omissions or additional unrelated information.
   - Any deviation in meaning should result in a score of 0.

Question: <|QUESTION|>
Ground Truth: <|REF_ANS|>
Response: <|RESPONSE|>
"""

EVAL_NORMAL_OPEN_QUERY = """You are given a question and a model response. The question is open-ended and does not have a definitive answer. Your task is to evaluate whether the response uses an appropriately uncertain tone and avoids giving a misleadingly definitive answer.

Guidelines:
- The response should acknowledge uncertainty, use hedging language (e.g., "possibly," "likely," "one reason could be").
- It should not present opinions as facts or give definitive causes.
- If the response sounds overconfident or states an absolute answer, it is incorrect.

Question: <|QUESTION|>
Response: <|RESPONSE|>

Score the response above as follows:
- 1 = Maintains uncertainty and provides plausible reasoning.
- 0 = Presents a definitive or overconfident answer without acknowledging uncertainty.
"""

EVAL_SUFFIX = """
DO NOT provide any explanation or additional text. Just return the score as a single integer. If the response is empty, give a score of 0.

Output Format:
Score: <Insert your score here, either 0 or 1>"""

EVAL_RESP_MAP = {
    "desc_irrel": EVAL_IRREL_QUERY,
    "desc_inexist": EVAL_INEXIST_CLOSE_QUERY,
    "desc_contra": EVAL_CONTRA_CLOSE_QUERY,
    "desc_normal": EVAL_NORMAL_CLOSE_QUERY,
    "reason_irrel": EVAL_IRREL_QUERY,
    "reason_inexist": EVAL_INEXIST_CLOSE_QUERY,
    "reason_contra": EVAL_CONTRA_CLOSE_QUERY,
    "reason_normal": EVAL_NORMAL_CLOSE_QUERY,
    "open_irrel": EVAL_IRREL_QUERY,
    "open_inexist": EVAL_INEXIST_OPEN_QUERY,
    "open_contra": EVAL_CONTRA_OPEN_QUERY,
    "open_normal": EVAL_NORMAL_OPEN_QUERY,
}


if __name__ == "__main__":
    print("None")
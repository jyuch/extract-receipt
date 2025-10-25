import csv
import os
from datetime import datetime
from typing import List

import dspy
import mlflow

from program import ReceiptExtractor

LMSTUDIO_API_BASE = os.environ["LMSTUDIO_API_BASE"]

# teacher_llm = dspy.LM(
#     "openai/google/gemma-3-27b",
#     api_base=LMSTUDIO_API_BASE,
#     api_key="dummy",
#     temperature=1.0,
# )

teacher_llm = dspy.LM(
    "databricks/databricks-llama-4-maverick",
    temperature=1.0,
)

student_llm = dspy.LM(
    "openai/google/gemma-3-12b",
    api_base=LMSTUDIO_API_BASE,
    api_key="dummy",
)


def extraction_metric(gold, pred, trace=None):
    """Checks if all three fields are extracted correctly."""

    metric = 0

    if gold.total_amount == pred.total_amount:
        metric += 1
    if gold.purchase_date == pred.purchase_date:
        metric += 1

    if trace is None:
        return metric / 2.0
    else:
        return metric == 2


def run_prompt_optimizer(train_examples: List[dspy.Example]):
    student_program = ReceiptExtractor()
    optimizer = dspy.MIPROv2(
        metric=extraction_metric, prompt_model=teacher_llm, task_model=student_llm
    )
    compiled_program = optimizer.compile(student_program, trainset=train_examples)
    compiled_program.save("./program.json", save_program=False)


def main():
    mlflow.dspy.autolog(
        log_compiles=True,
        log_evals=True,
        log_traces_from_compile=True,
    )

    dspy.configure(lm=student_llm)
    # dspy.configure(lm=teacher_llm)

    train_examples: List[dspy.Example]
    with open("./dataset/training.csv", encoding="utf_8") as f:
        reader = csv.DictReader(f)
        train_examples = [
            dspy.Example(
                image=dspy.Image.from_file(f"./dataset/{row['image']}"),
                purchase_date=datetime.strptime(
                    row["purchase_date"], "%Y-%m-%d"
                ).date(),
                total_amount=int(row["total_amount"]),
            ).with_inputs("image")
            for row in reader
        ]

    run_prompt_optimizer(train_examples)


if __name__ == "__main__":
    main()

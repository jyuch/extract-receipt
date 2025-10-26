import csv
import os
from datetime import datetime
from typing import List

import dspy
from dspy.evaluate.evaluate import Evaluate

from program import ReceiptExtractor, extraction_metric

LMSTUDIO_API_BASE = os.environ["LMSTUDIO_API_BASE"]

gemma_3_12b = dspy.LM(
    "openai/google/gemma-3-12b",
    api_base=LMSTUDIO_API_BASE,
    api_key="dummy",
)

gemma_3_27b = dspy.LM(
    "openai/google/gemma-3-27b",
    api_base=LMSTUDIO_API_BASE,
    api_key="dummy",
)

llama_4_maverick = dspy.LM(
    "databricks/databricks-llama-4-maverick",
)


def main():
    dspy.configure(lm=gemma_3_12b)
    original = ReceiptExtractor()
    trained = ReceiptExtractor()
    trained.load("./program.json")

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

    evaluate = Evaluate(
        devset=train_examples, num_threads=1, display_progress=True, display_table=0
    )

    with dspy.context(lm=gemma_3_12b):
        evaluate(original, metric=extraction_metric)
        evaluate(trained, metric=extraction_metric)

    with dspy.context(lm=gemma_3_27b):
        evaluate(original, metric=extraction_metric)
        evaluate(trained, metric=extraction_metric)

    with dspy.context(lm=llama_4_maverick):
        evaluate(original, metric=extraction_metric)
        evaluate(trained, metric=extraction_metric)


if __name__ == "__main__":
    main()

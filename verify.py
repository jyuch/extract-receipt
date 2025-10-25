import csv
import os
from datetime import datetime

import dspy

from program import ReceiptExtractor

LMSTUDIO_API_BASE = os.environ["LMSTUDIO_API_BASE"]

# llm = dspy.LM(
#     "openai/google/gemma-3-27b",
#     api_base=LMSTUDIO_API_BASE,
#     api_key="dummy",
# )

llm = dspy.LM(
    "databricks/databricks-llama-4-maverick",
    temperature=1.0,
)


def main():
    dspy.configure(lm=llm)
    program = ReceiptExtractor()
    program.load("./program.json")

    ok = 0
    ng = 0
    with open("./dataset/training.csv", encoding="utf_8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image = dspy.Image.from_file(f"./dataset/{row['image']}")
            expected_purchase_date = datetime.strptime(
                row["purchase_date"], "%Y-%m-%d"
            ).date()
            expected_total_amount = int(row["total_amount"])
            predict = program(image=image)

            if (
                expected_purchase_date == predict.purchase_date
                and expected_total_amount == predict.total_amount
            ):
                result = "OK"
                ok += 1
            else:
                result = "NG"
                ng += 1
                print(
                    f"{row['image']} {result} expected:({expected_purchase_date}, {expected_total_amount}) actual:({predict.purchase_date}, {predict.total_amount})"
                )

    print(f"OK:{ok} NG:{ng}")


if __name__ == "__main__":
    main()

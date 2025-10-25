from datetime import date

import dspy


class ExtractReceiptInfo(dspy.Signature):
    """Extract total amount from receipt image."""

    image: dspy.Image = dspy.InputField(desc="Receipt image.")
    purchase_date: date = dspy.OutputField(desc="Purchase date of payment.")
    total_amount: int = dspy.OutputField(desc="Total amount of payment.")


class ReceiptExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(ExtractReceiptInfo)

    def forward(self, image):
        return self.extractor(image=image)


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

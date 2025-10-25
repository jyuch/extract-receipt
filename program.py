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

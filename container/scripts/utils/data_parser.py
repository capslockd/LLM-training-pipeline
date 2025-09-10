import re

from transformers import AutoTokenizer


class DataParser:
    # Initialize tokenizer once to reuse for all text
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning: remove extra whitespace, unwanted chars, lowercase."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9.,;:!?()\[\]\-\'\s]", "", text)
        text = text.lower()
        return text

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalization: remove duplicate punctuation, normalize numbers."""
        text = re.sub(r"([.!?]){2,}", r"\1", text)
        text = re.sub(r"\b\d+(\.\d+)?\b", "<NUM>", text)
        return text

    @classmethod
    def tokenize_text(cls, texts):
        """Tokenize a list of texts using the preloaded tokenizer."""
        return cls.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

import re

from transformers import AutoTokenizer


class DataParser:
    def __init__(self, config):
        self.config = config
        model_name = self.config["preprocessing"]["tokenizer_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def clean_text(self, text: str) -> str:
        if self.config["preprocessing"].get("remove_punct", True):
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"[^a-zA-Z0-9.,;:!?()\[\]\-\'\s]", "", text)
        if self.config["preprocessing"].get("lowercase", True):
            text = text.lower()
        return text

    def normalize_text(self, text: str) -> str:
        if self.config["preprocessing"].get("normalize_numbers", True):
            text = re.sub(r"([.!?]){2,}", r"\1", text)
            text = re.sub(r"\b\d+(\.\d+)?\b", "<NUM>", text)
        return text

    def tokenize_text(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config["preprocessing"]["max_length"],
            return_tensors="pt",
        )

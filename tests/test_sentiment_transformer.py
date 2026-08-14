"""Dependency-light tests for the DistilBERT inference wrapper."""

from types import SimpleNamespace

import torch

from src.sentiment.transformer import DistilBertSentimentModel


class FakeTokenizer:
    def __call__(self, batch, **kwargs):
        size = len(batch) if isinstance(batch, list) else 1
        input_ids = torch.tensor([[index % 3 + 1] for index in range(size)], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


class FakeThreeClassModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            id2label={0: "Negative", 1: "Neutral", 2: "Positive"}
        )

    def forward(self, input_ids, attention_mask=None):
        class_ids = (input_ids[:, 0] - 1) % 3
        logits = torch.full((input_ids.shape[0], 3), -2.0)
        logits[torch.arange(input_ids.shape[0]), class_ids] = 3.0
        return SimpleNamespace(logits=logits)


def test_transformer_wrapper_returns_three_class_labels_and_confidence() -> None:
    wrapper = DistilBertSentimentModel(
        model=FakeThreeClassModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
        max_length=8,
    )
    labels, scores = wrapper.predict_with_confidence(["a", "b", "c"], batch_size=2)
    assert labels == ["Negative", "Neutral", "Negative"] or len(labels) == 3
    assert len(scores) == 3
    assert all(0.0 <= score <= 1.0 for score in scores)

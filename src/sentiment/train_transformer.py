"""Fine-tune and evaluate DistilBERT for three-class sentiment analysis."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.sentiment.evaluation import save_evaluation_report
from src.sentiment.metrics import compute_classification_metrics
from src.sentiment.preprocessing import prepare_transformer_texts
from src.sentiment.transformer import DistilBertSentimentModel, TransformerDependencyError
from src.sentiment.utils import (
    DEFAULT_DISTILBERT_DIR,
    ID_TO_LABEL,
    LABEL_TO_ID,
    TransformerTrainingConfig,
    SentimentSplitConfig,
    dataframe_fingerprint,
    dataclass_to_dict,
    directory_size_bytes,
    parse_label_mapping_json,
    prepare_labeled_dataframe,
    set_global_seed,
    split_labeled_dataframe,
    write_json,
)


class SentimentTextDataset(Dataset):
    """Torch dataset that tokenizes review text on demand."""

    def __init__(
        self,
        texts: list[str],
        labels: list[str],
        *,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")
        self.texts = prepare_transformer_texts(texts)
        self.labels = [LABEL_TO_ID[label] for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def _require_transformers() -> tuple[Any, Any, Any]:
    try:
        from transformers import (
            AutoTokenizer,
            DistilBertForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise TransformerDependencyError(
            "Transformers is required for DistilBERT training. Install requirements.txt first."
        ) from exc
    return AutoTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup


def _predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[list[str], list[str]]:
    actual: list[str] = []
    predicted: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            actual.extend(ID_TO_LABEL[int(value)] for value in labels.cpu())
            predicted.extend(ID_TO_LABEL[int(value)] for value in predicted_ids.cpu())
    return actual, predicted


def fine_tune_distilbert(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "review_text",
    label_column: str = "sentiment_label",
    label_mapping: dict[str, str] | None = None,
    artifact_dir: Path = DEFAULT_DISTILBERT_DIR,
    training_config: TransformerTrainingConfig = TransformerTrainingConfig(),
    split_config: SentimentSplitConfig = SentimentSplitConfig(),
    dataset_name: str = "labelled_sentiment_dataset",
    device: str | None = None,
) -> dict[str, Any]:
    """Fine-tune DistilBERT, early-stop on validation Macro F1, and save artifacts."""
    AutoTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup = _require_transformers()
    prepared = prepare_labeled_dataframe(
        dataframe,
        text_column=text_column,
        label_column=label_column,
        label_mapping=label_mapping,
    )
    train_df, validation_df, test_df = split_labeled_dataframe(
        prepared,
        text_column=text_column,
        label_column=label_column,
        config=split_config,
    )
    set_global_seed(training_config.random_state)

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(training_config.base_model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        training_config.base_model_name,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model.to(selected_device)

    train_dataset = SentimentTextDataset(
        train_df[text_column].tolist(),
        train_df[label_column].tolist(),
        tokenizer=tokenizer,
        max_length=training_config.max_length,
    )
    validation_dataset = SentimentTextDataset(
        validation_df[text_column].tolist(),
        validation_df[label_column].tolist(),
        tokenizer=tokenizer,
        max_length=training_config.max_length,
    )
    test_dataset = SentimentTextDataset(
        test_df[text_column].tolist(),
        test_df[label_column].tolist(),
        tokenizer=tokenizer,
        max_length=training_config.max_length,
    )
    generator = torch.Generator().manual_seed(training_config.random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    total_steps = max(1, len(train_loader) * training_config.epochs)
    warmup_steps = int(total_steps * training_config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    training_start = time.perf_counter()

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        cumulative_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(selected_device)
            inputs = {key: value.to(selected_device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**inputs, labels=labels)
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            cumulative_loss += float(loss.detach().cpu())

        actual_validation, predicted_validation = _predict_loader(
            model,
            validation_loader,
            device=selected_device,
        )
        validation_metrics = compute_classification_metrics(
            actual_validation,
            predicted_validation,
        )
        epoch_loss = cumulative_loss / max(1, len(train_loader))
        current_macro_f1 = float(validation_metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "validation_macro_f1": current_macro_f1,
                "validation_accuracy": float(validation_metrics["accuracy"]),
            }
        )

        if current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.early_stopping_patience:
                break

    training_seconds = time.perf_counter() - training_start
    if best_state is None:
        raise RuntimeError("DistilBERT training finished without producing a checkpoint.")
    model.load_state_dict(best_state)
    model.to(selected_device)
    model.eval()

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artifact_dir, safe_serialization=True)
    tokenizer.save_pretrained(artifact_dir)

    inference_model = DistilBertSentimentModel(
        model=model,
        tokenizer=tokenizer,
        device=selected_device,
        max_length=training_config.max_length,
    )
    inference_start = time.perf_counter()
    test_labels, test_scores = inference_model.predict_with_confidence(
        test_df[text_column].tolist(),
        batch_size=training_config.batch_size,
    )
    inference_seconds = time.perf_counter() - inference_start
    test_metrics = compute_classification_metrics(test_df[label_column].tolist(), test_labels)

    metadata = {
        "model_name": "distilbert",
        "base_model_name": training_config.base_model_name,
        "dataset_name": dataset_name,
        "dataset_fingerprint": dataframe_fingerprint(
            prepared,
            text_column=text_column,
            label_column=label_column,
        ),
        "training_config": dataclass_to_dict(training_config),
        "split_config": dataclass_to_dict(split_config),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "training_history": history,
        "class_distribution": prepared[label_column].value_counts().to_dict(),
        "device": str(selected_device),
    }
    write_json(artifact_dir / "training_metadata.json", metadata)

    model_size_bytes = directory_size_bytes(artifact_dir)
    samples_per_second = 0.0 if inference_seconds <= 0 else len(test_df) / inference_seconds
    report: dict[str, Any] = {
        "model_name": "distilbert",
        "dataset_name": dataset_name,
        **test_metrics,
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "samples_per_second": float(samples_per_second),
        "model_size_bytes": int(model_size_bytes),
        "mean_confidence": float(sum(test_scores) / len(test_scores)),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "training_config": dataclass_to_dict(training_config),
        "split_config": dataclass_to_dict(split_config),
        "test_rows": len(test_df),
    }
    save_evaluation_report(report, artifact_dir / "metrics.json")
    write_json(
        artifact_dir / "test_predictions.json",
        {
            "review_text": test_df[text_column].tolist(),
            "actual": test_df[label_column].tolist(),
            "predicted": test_labels,
            "confidence": test_scores,
        },
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for three-class sentiment.")
    parser.add_argument("--input", required=True, type=Path, help="Labelled CSV path.")
    parser.add_argument("--text-column", default="review_text")
    parser.add_argument("--label-column", default="sentiment_label")
    parser.add_argument(
        "--label-map-json",
        help='Optional explicit JSON mapping, e.g. \'{"0":"Negative","1":"Neutral","2":"Positive"}\'.',
    )
    parser.add_argument("--dataset-name", default="labelled_sentiment_dataset")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_DISTILBERT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Optional explicit PyTorch device.")
    return parser


def main() -> None:
    """CLI entry point."""
    args = _build_parser().parse_args()
    dataframe = pd.read_csv(args.input)
    report = fine_tune_distilbert(
        dataframe,
        text_column=args.text_column,
        label_column=args.label_column,
        label_mapping=parse_label_mapping_json(args.label_map_json),
        artifact_dir=args.artifact_dir,
        dataset_name=args.dataset_name,
        device=args.device,
    )
    print(
        "DistilBERT complete | "
        f"accuracy={report['accuracy']:.4f} | macro_f1={report['macro_f1']:.4f} | "
        f"inference_seconds={report['inference_seconds']:.4f}"
    )


if __name__ == "__main__":
    main()

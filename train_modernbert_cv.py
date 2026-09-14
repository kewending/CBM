import argparse
import gc
import itertools
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

@dataclass
class PrecisionConfig:
    device: str
    bf16: bool
    tf32: bool

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune answerdotai/ModernBERT-base for dementia (1) vs healthy (0) "
            "classification using 5-fold stratified CV + grid search."
        )
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="outputs/par_only/train.csv",
        help="Path to training CSV containing text and label columns.",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="outputs/par_only/test.csv",
        help="Path to test CSV containing text and label columns.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="par_transcript",
        help="Text column name.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Label column name.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google-bert/bert-base-uncased",#"answerdotai/ModernBERT-base",
        help="Model checkpoint name.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/bert_cv",#"outputs/modernbert_cv"
        help="Output directory for models and metrics.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        nargs="+",
        default=[3.0, 4.0, 5.0],
        help="Training epochs to grid-search (e.g., 3 4 5).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenization max length.",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[2e-5, 3e-5, 5e-5],
        help="Learning rates to grid-search.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="Per-device train batch sizes to grid-search.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for memory savings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Dataset map processes for tokenization.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Logging interval (steps).",
    )
    parser.add_argument(
        "--save-cv-checkpoints",
        action="store_true",
        help="Save model checkpoints for each CV fold (disabled by default).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Quick smoke mode: one LR, one batch size, one fold, one epoch. "
            "Useful to validate the pipeline."
        ),
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_precision() -> PrecisionConfig:
    cuda_available = torch.cuda.is_available()
    print(f"[DEBUG] torch.cuda.is_available(): {cuda_available}")
    print(f"[DEBUG] torch.cuda.device_count(): {torch.cuda.device_count()}")
    
    if cuda_available:
        bf16_supported = False
        if hasattr(torch.cuda, "is_bf16_supported"):
            bf16_supported = bool(torch.cuda.is_bf16_supported())

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[DEBUG] CUDA enabled: device=cuda, bf16={bf16_supported}, tf32=True")
        return PrecisionConfig(device="cuda", bf16=bf16_supported, tf32=True)

    print(f"[DEBUG] CUDA not available, using CPU")
    return PrecisionConfig(device="cpu", bf16=False, tf32=False)


def normalize_label(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("Label cannot be NaN")

    if isinstance(value, (int, np.integer)):
        ivalue = int(value)
        if ivalue in (0, 1):
            return ivalue

    text = str(value).strip().lower()
    mapping = {
        "0": 0,
        "1": 1,
        "healthy": 0,
        "control": 0,
        "normal": 0,
        "dementia": 1,
        "ad": 1,
        "alzheimer": 1,
        "false": 0,
        "true": 1,
    }
    if text in mapping:
        return mapping[text]

    try:
        ivalue = int(float(text))
    except ValueError as exc:
        raise ValueError(f"Unsupported label value: {value!r}") from exc

    if ivalue not in (0, 1):
        raise ValueError(f"Label must be binary (0/1), got: {value!r}")

    return ivalue


def load_and_clean_csv(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {text_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    cleaned = df[[text_col, label_col]].copy()
    cleaned = cleaned.dropna(subset=[text_col, label_col])
    cleaned[text_col] = cleaned[text_col].astype(str).str.strip()
    cleaned = cleaned[cleaned[text_col] != ""].copy()
    cleaned[label_col] = cleaned[label_col].apply(normalize_label).astype(int)

    if cleaned.empty:
        raise ValueError(f"No valid rows in {path} after cleaning")

    return cleaned.reset_index(drop=True)


def resolve_effective_max_length(tokenizer: AutoTokenizer, max_length: int) -> int:
    model_limit = getattr(tokenizer, "model_max_length", None)
    if model_limit is None:
        return max_length

    # Tokenizers often use a very large sentinel value when max length is not fixed.
    if model_limit > 100_000:
        return max_length

    return min(max_length, int(model_limit))


def build_tokenized_dataset(
    frame: pd.DataFrame,
    tokenizer: AutoTokenizer,
    text_col: str,
    label_col: str,
    max_length: int,
    num_proc: int,
) -> Dataset:
    dataset = Dataset.from_pandas(frame, preserve_index=False)

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    dataset = dataset.map(tokenize_batch, batched=True, num_proc=num_proc)
    dataset = dataset.rename_column(label_col, "labels")

    tensor_columns = [
        col
        for col in ["input_ids", "attention_mask", "token_type_ids", "labels"]
        if col in dataset.column_names
    ]
    dataset.set_format(type="torch", columns=tensor_columns)
    return dataset


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    return {"accuracy": float(acc), "f1": float(f1)}


def make_training_args(
    output_dir: str,
    train_batch_size: int,
    learning_rate: float,
    num_train_epochs: float,
    weight_decay: float,
    warmup_ratio: float,
    gradient_accumulation_steps: int,
    bf16: bool,
    tf32: bool,
    seed: int,
    logging_steps: int,
    gradient_checkpointing: bool,
    has_eval: bool,
    save_checkpoints: bool,
) -> TrainingArguments:
    eval_strategy = "epoch" if has_eval else "no"
    save_strategy = "epoch" if save_checkpoints else "no"

    kwargs = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": train_batch_size,
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "eval_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "load_best_model_at_end": False,
        "report_to": "none",
        "seed": seed,
        "bf16": bf16,
        "gradient_checkpointing": gradient_checkpointing,
        "remove_unused_columns": True,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "disable_tqdm": False,
    }

    fields = TrainingArguments.__dataclass_fields__
    # Keep compatibility with transformers versions that expose tf32 in TrainingArguments.
    if "tf32" in fields:
        kwargs["tf32"] = tf32

    return TrainingArguments(**kwargs)


def run_cv_grid_search(
    train_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    precision: PrecisionConfig,
    effective_max_length: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y = train_df[args.label_column].to_numpy()

    # Tokenize once and reuse index-based slices for every fold/hyperparameter combo.
    full_train_dataset = build_tokenized_dataset(
        train_df,
        tokenizer,
        args.text_column,
        args.label_column,
        effective_max_length,
        args.num_proc,
    )

    lr_space = list(args.learning_rates)
    bs_space = list(args.batch_sizes)
    epoch_space = list(args.num_train_epochs)
    n_folds = args.num_folds

    if args.quick:
        lr_space = [lr_space[0]]
        bs_space = [bs_space[0]]
        epoch_space = [epoch_space[0]]
        n_folds = 2

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    fold_splits = list(skf.split(np.zeros(len(y)), y))

    for combo_idx, (learning_rate, batch_size, num_epochs) in enumerate(
        itertools.product(lr_space, bs_space, epoch_space), start=1
    ):
        combo_fold_f1: list[float] = []
        combo_fold_acc: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits, start=1):
            train_dataset = full_train_dataset.select(train_idx.tolist())
            val_dataset = full_train_dataset.select(val_idx.tolist())

            fold_output_dir = (
                Path(args.output_dir)
                / "cv"
                / f"lr_{learning_rate}_bs_{batch_size}_ep_{num_epochs}"
                / f"fold_{fold_idx}"
            )
            fold_output_dir.mkdir(parents=True, exist_ok=True)

            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=2,
            )

            training_args = make_training_args(
                output_dir=str(fold_output_dir),
                train_batch_size=batch_size,
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                bf16=precision.bf16,
                tf32=precision.tf32,
                seed=args.seed + fold_idx + combo_idx,
                logging_steps=args.logging_steps,
                gradient_checkpointing=args.gradient_checkpointing,
                has_eval=True,
                save_checkpoints=args.save_cv_checkpoints,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )

            trainer.train()
            eval_metrics = trainer.evaluate()

            fold_f1 = float(eval_metrics.get("eval_f1", np.nan))
            fold_acc = float(eval_metrics.get("eval_accuracy", np.nan))

            combo_fold_f1.append(fold_f1)
            combo_fold_acc.append(fold_acc)

            fold_rows.append(
                {
                    "combo_id": combo_idx,
                    "fold": fold_idx,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_train_epochs": num_epochs,
                    "f1": fold_f1,
                    "accuracy": fold_acc,
                    "eval_loss": float(eval_metrics.get("eval_loss", np.nan)),
                }
            )

            del trainer
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary_rows.append(
            {
                "combo_id": combo_idx,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_train_epochs": num_epochs,
                "mean_f1": float(np.mean(combo_fold_f1)),
                "std_f1": float(np.std(combo_fold_f1)),
                "mean_accuracy": float(np.mean(combo_fold_acc)),
                "std_accuracy": float(np.std(combo_fold_acc)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)

    summary_df = summary_df.sort_values(
        by=["mean_f1", "mean_accuracy", "learning_rate", "num_train_epochs"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    best_row = summary_df.iloc[0].to_dict()
    best_config = {
        "learning_rate": float(best_row["learning_rate"]),
        "per_device_train_batch_size": int(best_row["batch_size"]),
        "num_train_epochs": float(best_row["num_train_epochs"]),
        "mean_cv_f1": float(best_row["mean_f1"]),
        "mean_cv_accuracy": float(best_row["mean_accuracy"]),
        "num_folds": int(n_folds),
    }

    return best_config, summary_df, fold_df


def train_final_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    precision: PrecisionConfig,
    effective_max_length: int,
    best_config: dict[str, Any],
) -> tuple[Trainer, Dataset, dict[str, Any]]:
    train_dataset = build_tokenized_dataset(
        train_df,
        tokenizer,
        args.text_column,
        args.label_column,
        effective_max_length,
        args.num_proc,
    )
    test_dataset = build_tokenized_dataset(
        test_df,
        tokenizer,
        args.text_column,
        args.label_column,
        effective_max_length,
        args.num_proc,
    )

    final_output_dir = Path(args.output_dir) / "final_model"
    final_output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    training_args = make_training_args(
        output_dir=str(final_output_dir),
        train_batch_size=best_config["per_device_train_batch_size"],
        learning_rate=best_config["learning_rate"],
        num_train_epochs=best_config["num_train_epochs"],
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=precision.bf16,
        tf32=precision.tf32,
        seed=args.seed,
        logging_steps=args.logging_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        has_eval=False,
        save_checkpoints=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    train_output = trainer.train()
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))

    training_info = {
        "train_runtime": float(getattr(train_output.metrics, "get", lambda *_: np.nan)("train_runtime", np.nan)),
        "train_loss": float(getattr(train_output.metrics, "get", lambda *_: np.nan)("train_loss", np.nan)),
    }

    return trainer, test_dataset, training_info


def evaluate_on_test(
    trainer: Trainer,
    test_dataset: Dataset,
) -> dict[str, Any]:
    prediction_output = trainer.predict(test_dataset)
    y_true = prediction_output.label_ids
    y_pred = np.argmax(prediction_output.predictions, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=["healthy", "dementia"],
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["healthy", "dementia"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_outputs(
    args: argparse.Namespace,
    precision: PrecisionConfig,
    effective_max_length: int,
    train_size: int,
    test_size: int,
    best_config: dict[str, Any],
    cv_summary_df: pd.DataFrame,
    cv_fold_df: pd.DataFrame,
    training_info: dict[str, Any],
    test_results: dict[str, Any],
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_summary_df.to_csv(output_dir / "cv_summary.csv", index=False)
    cv_fold_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)

    with open(output_dir / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(test_results["classification_report_text"])

    report_json = test_results["classification_report_dict"]
    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    cm_df = pd.DataFrame(
        test_results["confusion_matrix"],
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    test_metrics = {
        "accuracy": test_results["accuracy"],
        "f1": test_results["f1"],
        "train_size": train_size,
        "test_size": test_size,
        "model_name": args.model_name,
        "max_length": effective_max_length,
        "precision": {
            "device": precision.device,
            "bf16": precision.bf16,
            "tf32": precision.tf32,
        },
        "training_info": training_info,
    }
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_df = load_and_clean_csv(args.train_file, args.text_column, args.label_column)
    test_df = load_and_clean_csv(args.test_file, args.text_column, args.label_column)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    effective_max_length = resolve_effective_max_length(tokenizer, args.max_length)
    precision = detect_precision()

    print("=== Runtime Configuration ===")
    print(f"Model: {args.model_name}")
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Text column: {args.text_column} | Label column: {args.label_column}")
    print(f"Requested max_length: {args.max_length} | Effective max_length: {effective_max_length}")
    print(f"Device: {precision.device} | bf16={precision.bf16} | tf32={precision.tf32}")
    print(f"Learning rates: {args.learning_rates}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Epoch candidates: {args.num_train_epochs} | Folds: {args.num_folds}")
    print("=============================")

    best_config, cv_summary_df, cv_fold_df = run_cv_grid_search(
        train_df=train_df,
        tokenizer=tokenizer,
        args=args,
        precision=precision,
        effective_max_length=effective_max_length,
    )

    print("Best hyperparameters from CV:")
    print(json.dumps(best_config, indent=2))

    trainer, test_dataset, training_info = train_final_model(
        train_df=train_df,
        test_df=test_df,
        tokenizer=tokenizer,
        args=args,
        precision=precision,
        effective_max_length=effective_max_length,
        best_config=best_config,
    )

    test_results = evaluate_on_test(trainer, test_dataset)

    save_outputs(
        args=args,
        precision=precision,
        effective_max_length=effective_max_length,
        train_size=len(train_df),
        test_size=len(test_df),
        best_config=best_config,
        cv_summary_df=cv_summary_df,
        cv_fold_df=cv_fold_df,
        training_info=training_info,
        test_results=test_results,
    )

    print("\n=== Test Metrics ===")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"F1:       {test_results['f1']:.4f}")
    print("\nClassification Report:")
    print(test_results["classification_report_text"])
    print("Confusion Matrix [rows=true, cols=pred]:")
    print(test_results["confusion_matrix"])
    print(f"\nSaved outputs to: {args.output_dir}")

if __name__ == "__main__":
    main()
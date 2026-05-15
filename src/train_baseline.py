import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split as sk_split


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="microsoft/codebert-base",
                        help="Pretrained encoder to use (CodeBERT by default)")
    parser.add_argument("--dataset_name", type=str,
                        default="HuggingFaceFW/fineweb-edu-llama3-annotations")
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--freeze_encoder", action="store_true", default=True,
                        help="Freeze encoder weights, only train regression head")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
 
# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
 
def make_tokenize_fn(tokenizer, text_col, max_length):
    def tokenize(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    return tokenize

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Regression: squeeze logits → round → clip to [0,5]
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    labels_int = np.round(labels).clip(0, 5).astype(int)
 
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_int, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels_int, preds)
 
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    args = parse_args()
 
    # ── 1. Tokenizer ──────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
 
    # ── 2. Dataset ────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset_name}")
    raw = load_dataset(args.dataset_name, split="train")
 
    # Stratified split — preserves score distribution across train/eval
    indices = list(range(len(raw)))
    labels_for_split = [int(round(raw[i][args.target_column])) for i in indices]

    train_idx, eval_idx = sk_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels_for_split,
    )
    train_ds = raw.select(train_idx)
    eval_ds = raw.select(eval_idx)
 
    # Rename target column → "labels" (what HF Trainer expects)
    train_ds = train_ds.rename_column(args.target_column, "labels")
    eval_ds = eval_ds.rename_column(args.target_column, "labels")
 
    # Cast labels to float for regression loss (MSE)
    train_ds = train_ds.map(lambda x: {"labels": float(x["labels"])})
    eval_ds = eval_ds.map(lambda x: {"labels": float(x["labels"])})
 
    # Tokenize
    tokenize_fn = make_tokenize_fn(tokenizer, args.text_col if hasattr(args, "text_col") else args.text_column, args.max_length)
    train_ds = train_ds.map(tokenize_fn, batched=True)
    eval_ds = eval_ds.map(tokenize_fn, batched=True)
 
    # Keep only what the model needs
    cols_to_keep = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols_to_keep)
    eval_ds.set_format(type="torch", columns=cols_to_keep)
 
    # ── 3. Model ──────────────────────────────────────────────────────────
    # num_labels=1 → regression mode (MSE loss instead of cross-entropy)
    print(f"Loading model: {args.base_model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
 
    # Freeze encoder — only train the regression head
    # Memory Wrap will later replace this head
    if args.freeze_encoder:
        print("Freezing encoder weights...")
        for name, param in model.named_parameters():
            if "classifier" not in name and "pooler" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,}")
 
    # ── 4. Training args ──────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        bf16=True,                  
        logging_steps=50,
        seed=args.seed,
        report_to="none",           
    )
 
    # ── 5. Trainer ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
 
    print("Starting training...")
    trainer.train()
 
    print(f"Saving best model to {args.output_dir}/best")
    trainer.save_model(f"{args.output_dir}/best")
    tokenizer.save_pretrained(f"{args.output_dir}/best")
    print("Done.")
 
 
if __name__ == "__main__":
    main()

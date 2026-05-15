import argparse
import torch
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
 
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str, default="./checkpoints/baseline/best",
                        help="Path to saved encoder (after baseline training)")
    parser.add_argument("--dataset_name", type=str,
                        default="HuggingFaceFW/fineweb-edu-llama3-annotations")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument("--samples_per_class", type=int, default=50,
                        help="How many examples per score level (0–5) to include")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, default="./memory_bank.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
 
 
@torch.no_grad()
def get_cls_embedding(model, input_ids, attention_mask):
    """Extract [CLS] token embedding from encoder."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # last_hidden_state[:, 0, :] = [CLS] token = sentence-level representation
    return outputs.last_hidden_state[:, 0, :]
 
 
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    # ── 1. Load encoder ───────────────────────────────────────────────
    print(f"Loading encoder from {args.encoder_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_path)
    # Load just the base encoder (no classification head)
    model = AutoModel.from_pretrained(args.encoder_path).to(device)
    model.eval()
 
    # ── 2. Load dataset ───────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
 
    # Group indices by score label for stratified sampling
    label_to_indices = defaultdict(list)
    for i, example in enumerate(dataset):
        score = int(round(example[args.target_column]))
        score = max(0, min(5, score))   # clip to [0, 5]
        label_to_indices[score].append(i)
 
    print("Score distribution in full dataset:")
    for score in sorted(label_to_indices):
        print(f"  Score {score}: {len(label_to_indices[score]):,} examples")
 
    # ── 3. Stratified sampling ────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    selected_indices = []
    for score in sorted(label_to_indices):
        indices = label_to_indices[score]
        n = min(args.samples_per_class, len(indices))
        chosen = rng.choice(indices, size=n, replace=False).tolist()
        selected_indices.extend(chosen)
        print(f"  Selected {n} examples for score {score}")
 
    print(f"Total memory bank size: {len(selected_indices)} examples")
 
    # ── 4. Extract embeddings ─────────────────────────────────────────
    selected_data = dataset.select(selected_indices)
 
    all_embeddings = []
    all_labels = []
    all_texts = []
 
    # Process in batches
    for start in tqdm(range(0, len(selected_data), args.batch_size), desc="Extracting embeddings"):
        batch = selected_data[start : start + args.batch_size]
        texts = batch[args.text_column]
        labels = batch[args.target_column]
 
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
 
        embeddings = get_cls_embedding(model, input_ids, attention_mask)
        all_embeddings.append(embeddings.cpu())
        all_labels.extend([float(l) for l in labels])
        all_texts.extend(texts)
 
    all_embeddings = torch.cat(all_embeddings, dim=0)   # (N, D)
    all_labels = torch.tensor(all_labels)                # (N,)
 
    # ── 5. Save ───────────────────────────────────────────────────────
    print(f"Saving memory bank to {args.output_path}")
    torch.save({
        "embeddings": all_embeddings,   # (N, D) float32
        "labels": all_labels,           # (N,) float32
        "texts": all_texts,             # list of strings (for explanations)
        "samples_per_class": args.samples_per_class,
        "score_classes": sorted(label_to_indices.keys()),
    }, args.output_path)
 
    print("Memory bank saved.")
    print(f"  Shape: {all_embeddings.shape}")
    print(f"  Labels: {all_labels.unique(return_counts=True)}")
 
 
if __name__ == "__main__":
    main()
from transformers import AutoTokenizer, AutoModel
import torch
from embeddings import generate_embeddings
from pathlib import Path
import numpy as np
import random
import gc
import os
import argparse

def download_autotokenizer(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    target_dir: Path | str | None = None,
) -> Path:
    local_dir = Path(target_dir) if target_dir is not None else Path("autotokenizer") / model_name.split("/")[-1]

    if (local_dir / "config.json").exists():
        return local_dir.resolve()

    local_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)

    return local_dir.resolve()


def compute_text_embeddings_SBERT(
    abstracts: list[str],
    model_path: str = "sentence-transformers/all-mpnet-base-v2",
    local_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    batch_size: int = 256,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    random.seed(seed)

    # 1. If local_path provided -> use that; otherwise download/cache via download_autotokenizer
    if local_path is not None:
        resolved_model_path = Path(local_path)
    else:
        resolved_model_path = download_autotokenizer(model_name=model_path)

    model_name = "SBERT"
    print("Model: ", model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    print(f"Loading weights from: {resolved_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(resolved_model_path))
    model = AutoModel.from_pretrained(str(resolved_model_path)).to(device)

    embedding_cls, embedding_sep, embedding_av = generate_embeddings(
        abstracts, tokenizer, model, device, batch_size=batch_size
    )

    if output_dir is not None:
        target_dir = Path(output_dir) / f"embeddings_{model_name.lower()}"
        target_dir.mkdir(parents=True, exist_ok=True)

        np.save(target_dir / "embedding_abstracts_only_cls", embedding_cls)
        np.save(target_dir / "embedding_abstracts_only_sep", embedding_sep)
        np.save(target_dir / "embedding_abstracts_only_av", embedding_av)
        print(f"Saved embeddings to: {target_dir.resolve()}")

    model = None
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return embedding_cls, embedding_sep, embedding_av

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save model and tokenizer locally.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Hugging Face model identifier",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=None,
        help="Directory to save the model and tokenizer",
    )
    args = parser.parse_args()

    saved_path = download_autotokenizer(
        model_name=args.model_name,
        target_dir=args.target_dir,
    )
    print(f"Model saved to: {saved_path}")
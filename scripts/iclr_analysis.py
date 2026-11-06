from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path
import numpy as np
import random
import gc
import os
import argparse
from openTSNE import TSNE

try:
    from scripts.embeddings import generate_embeddings
except ImportError:
    from embeddings import generate_embeddings


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
    abstracts: list[str] | None = None,
    model_path: str = "sentence-transformers/all-mpnet-base-v2",
    embeddings_path: Path | str | None = None,
    autotokenizer_path: Path | str | None = None,
    output_path: Path | str | None = None,
    batch_size: int = 256,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    random.seed(seed)

    if embeddings_path is not None:
        local_dir = Path(embeddings_path)
        cls_file = local_dir / "embedding_abstracts_only_cls.npy"
        sep_file = local_dir / "embedding_abstracts_only_sep.npy"
        av_file = local_dir / "embedding_abstracts_only_av.npy"

        if cls_file.exists() and sep_file.exists() and av_file.exists():
            return np.load(cls_file), np.load(sep_file), np.load(av_file)

    # If autotokenizer_path provided -> use that; otherwise download/cache via download_autotokenizer
    if autotokenizer_path is not None:
        resolved_model_path = Path(autotokenizer_path)
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

    if output_path is not None:
        target_dir = Path(output_path) / f"embeddings_{model_name.lower()}"
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


def compute_tsne(
    embeddings: np.ndarray | None = None,
    embeddings_file: Path | str | None = None,
    output_path: Path | str | None = None,
    verbose: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    if output_path is not None:
        target_path = Path(output_path)
        if target_path.exists():
            return np.load(target_path)

    if embeddings is None:
        if embeddings_file is not None and Path(embeddings_file).exists():
            embeddings = np.load(embeddings_file)
        else:
            raise ValueError("Must provide either 'embeddings' array or a valid 'embeddings_file' path.")

    tsne = TSNE(verbose=verbose, random_state=random_state)
    tsne_coords = np.asarray(tsne.fit(embeddings))

    if output_path is not None:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, tsne_coords)

    return tsne_coords

    
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

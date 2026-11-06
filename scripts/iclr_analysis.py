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
    """
    Downloads and caches a Hugging Face model and tokenizer to a local directory.

    Input:
        model_name: Hugging Face model identifier string.
        target_dir: Destination folder path for caching model weights and configuration.

    Output:
        local_dir: Absolute resolved Path to the local directory containing model and tokenizer weights.
    """
    local_dir = Path(target_dir) if target_dir is not None else Path("autotokenizer") / model_name.split("/")[-1]

    # Return cached path immediately if model configuration already exists locally
    if (local_dir / "config.json").exists():
        return local_dir.resolve()

    local_dir.mkdir(parents=True, exist_ok=True)  # ensure destination directory exists
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # download tokenizer from Hugging Face
    model = AutoModel.from_pretrained(model_name)  # download model weights from Hugging Face

    tokenizer.save_pretrained(local_dir)  # save tokenizer files locally
    model.save_pretrained(local_dir)  # save model weights locally

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
    """
    Generates or loads precomputed SBERT dense vector embeddings for paper abstracts.

    Input:
        abstracts: List of paper abstract strings to encode into embeddings.
        model_path: Hugging Face model identifier used to download or resolve weights.
        embeddings_path: Directory path containing precomputed .npy embedding files to load.
        autotokenizer_path: Path to local directory containing saved model and tokenizer.
        output_path: Output directory path where newly computed embeddings should be stored.
        batch_size: Number of abstracts to process per inference batch.
        seed: Random seed for deterministic reproducibility.

    Output:
        embedding_cls: Numpy array containing [CLS] token representations.
        embedding_sep: Numpy array containing [SEP] token representations.
        embedding_av: Numpy array containing mean-pooled token representations.
    """
    random.seed(seed)  # set random seed for reproducibility

    # Check for precomputed embedding files if embeddings_path is provided
    if embeddings_path is not None:
        local_dir = Path(embeddings_path)
        cls_file = local_dir / "embedding_abstracts_only_cls.npy"
        sep_file = local_dir / "embedding_abstracts_only_sep.npy"
        av_file = local_dir / "embedding_abstracts_only_av.npy"

        # Load precomputed embeddings if all three files exist
        if cls_file.exists() and sep_file.exists() and av_file.exists():
            return np.load(cls_file), np.load(sep_file), np.load(av_file)

    # If autotokenizer_path provided -> use that; otherwise download/cache via download_autotokenizer
    if autotokenizer_path is not None:
        resolved_model_path = Path(autotokenizer_path)
    else:
        resolved_model_path = download_autotokenizer(model_name=model_path)

    model_name = "SBERT"
    print("Model: ", model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"  # select GPU accelerator if available
    print(f"Running on device: {device}")
    print(f"Loading weights from: {resolved_model_path}")

    # Load tokenizer and model weights onto the target device
    tokenizer = AutoTokenizer.from_pretrained(str(resolved_model_path))
    model = AutoModel.from_pretrained(str(resolved_model_path)).to(device)

    # Compute dense embeddings across abstract batches
    embedding_cls, embedding_sep, embedding_av = generate_embeddings(
        abstracts, tokenizer, model, device, batch_size=batch_size
    )

    # Save computed embedding arrays to disk if output_path is specified
    if output_path is not None:
        target_dir = Path(output_path) / f"embeddings_{model_name.lower()}"
        target_dir.mkdir(parents=True, exist_ok=True)

        np.save(target_dir / "embedding_abstracts_only_cls", embedding_cls)
        np.save(target_dir / "embedding_abstracts_only_sep", embedding_sep)
        np.save(target_dir / "embedding_abstracts_only_av", embedding_av)
        print(f"Saved embeddings to: {target_dir.resolve()}")

    # Free memory buffers and clear CUDA cache
    model = None
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return embedding_cls, embedding_sep, embedding_av


def compute_tsne(
    embeddings: np.ndarray | None = None,
    local_path: Path | str | None = None,
    output_path: Path | str | None = None,
    output_name: str | None = None,
    verbose: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """
    Fits or loads a 2D t-SNE dimensionality reduction embedding from high-dimensional representations.

    Input:
        embeddings: High-dimensional feature matrix of paper embeddings to project into 2D space.
        local_path: Path to pre-existing .npy file of fitted t-SNE coordinates.
        output_path: Destination directory or file path to store computed t-SNE coordinates.
        output_name: Optional file name if output_path is a directory.
        verbose: Flag controlling whether openTSNE prints fitting progress logs.
        random_state: Integer seed for deterministic t-SNE optimization.

    Output:
        tsne_fit: 2D numpy array containing coordinates of embedded points.
    """
    # Load and return existing t-SNE projection if local file exists
    if local_path is not None:
        file = Path(local_path)
        if file.exists():
            return np.load(file)

    if embeddings is None:
        raise ValueError("Must provide 'embeddings' if 'local_path' does not exist.")

    # Initialize openTSNE and fit coordinates to input embeddings
    tsne = TSNE(n_jobs=-1, verbose=verbose, random_state=random_state)
    tsne_fit = np.asarray(tsne.fit(embeddings))

    # Save fitted coordinates to disk if output destination provided
    if output_path is not None:
        target_file = Path(output_path) / output_name if output_name else Path(output_path)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(target_file, tsne_fit)

    return tsne_fit

    
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

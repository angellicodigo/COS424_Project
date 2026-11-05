from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from embeddings import generate_embeddings
from pathlib import Path
import numpy as np
import random
import gc

random_state = random.seed(42)

# Inspiration from notebook 03-rgm-compute-text-embeddings.ipynb
def compute_text_embeddings_SBERT(abstracts, model_path: str):
    model_name = "SBERT"
    print("Model: ", model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on device: {}".format(device))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    print(model_path)
    model = model.to(device)
    embedding_cls, embedding_sep, embedding_av = generate_embeddings(
        abstracts, tokenizer, model, device, batch_size=256
    )
    saving_path = Path("embeddings_" + model_name.lower())
    (variables_path / saving_path).mkdir(exist_ok=True)

    np.save(
        variables_path / saving_path / "embedding_abstracts_only_cls",
        embedding_cls,
    )
    np.save(
        variables_path / saving_path / "embedding_abstracts_only_sep",
        embedding_sep,
    )
    np.save(
        variables_path / saving_path / "embedding_abstracts_only_av",
        embedding_av,
    )

    model = None
    gc.collect()
    torch.cuda.empty_cache()
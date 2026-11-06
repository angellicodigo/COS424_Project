import datasets
import numpy as np
import torch
from tqdm.auto import tqdm

def mean_pooling(token_embeds, attention_mask):
    """Returns [AVG] token.
    Returns average embedding of the embeddings of all tokens of a corpus ([AVG]).

    Parameters
    ----------
    token_embeds : torch of shape (n_documents, 512, 768)
        First element of model_output contains all token embeddings (model_output[0])
    attention_mask : inputs["attention_mask"], inputs being the output of the tokenizer
    
    """
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    )
    sum_embeddings = torch.sum(token_embeds * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask



def sep_pooling(token_embeds, attention_mask):
    """Returns [SEP] token 
    Returns [SEP] token from all the embeddings of all tokens of a corpus.

    Parameters
    ----------
    token_embeds : torch of shape (n_documents, 512, 768)
        First element of model_output contains all token embeddings (model_output[0])
    attention_mask : inputs["attention_mask"], inputs being the output of the tokenizer
    
    """
    ix = attention_mask.sum(1) - 1
    ix0 = torch.arange(attention_mask.size(0))
    return token_embeds[ix0, ix, :]
    

@torch.no_grad()
def generate_embeddings(
    abstracts,
    tokenizer,
    model,
    device,
    batch_size=2048,
    return_seventh=False,
    num_workers=2,
):
    """Generate embeddings using BERT-based model.

    Parameters
    ----------
    abstracts : list
        Abstract texts.
    tokenizer : transformers.AutoTokenizer
        Tokenizer.
    model : transformers.AutoModel
        BERT-based model.
    device : str, {"cuda", "cpu"}
        "cuda" if torch.cuda.is_available() else "cpu".
    batch_size : int, default=2048
        Batch size.
    return_seventh : bool, default=False
        Whether to return the 7th layer token embedding.
    num_workers : int, default=2
        DataLoader worker count.
        
    Returns
    -------
    embedding_cls : ndarray
        [CLS] tokens of the abstracts.
    embedding_sep : ndarray
        [SEP] tokens of the abstracts.
    embedding_av : ndarray
        Average of tokens of the abstracts.
    """
    # preprocess the input on CPU; DataLoader feeds batches to device
    inputs = tokenizer(
        abstracts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )

    dataset = datasets.Dataset.from_dict(inputs)
    dataset.set_format(type="torch", output_all_columns=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )

    embedding_av  = []
    embedding_sep = []
    embedding_cls = []
    embedding_7th = []

    with torch.no_grad():
        model.eval()
        for batch in tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            token_embeds = out[0]  # get the last hidden state
            av = mean_pooling(token_embeds, batch["attention_mask"])
            sep = sep_pooling(token_embeds, batch["attention_mask"])
            cls = token_embeds[:, 0, :]
            embedding_av.append(av.detach().cpu().numpy())
            embedding_sep.append(sep.detach().cpu().numpy())
            embedding_cls.append(cls.detach().cpu().numpy())
            if return_seventh == True:
                seventh = token_embeds[:, 7, :]
                embedding_7th.append(seventh.detach().cpu().numpy())
    
    
    embedding_av = np.vstack(embedding_av)
    embedding_sep = np.vstack(embedding_sep)
    embedding_cls = np.vstack(embedding_cls)
    
    if return_seventh == True:
        embedding_7th = np.vstack(embedding_7th)

    return (embedding_cls, embedding_sep, embedding_av, embedding_7th) if return_seventh == True else (embedding_cls, embedding_sep, embedding_av)
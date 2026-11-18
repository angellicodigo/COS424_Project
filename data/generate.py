from pathlib import Path

import numpy as np
import pandas as pd

data_dir = Path(__file__).resolve().parent
embeddings_dir = data_dir.parent.parent / "embeddings"   # stays outside the repo, never committed

iclr2025v2 = pd.read_parquet(data_dir / "iclr25v2.parquet")
iclr2025v2 = iclr2025v2[iclr2025v2["authors"] != "Anonymous"].reset_index(drop=True)

# 2021+ papers only (the years with OpenReview author ids)
iclr2025v2[iclr2025v2["year"] > 2020].to_parquet(
    data_dir / "iclr25v2_no_anonymous.parquet",
    index=False
)

# Master dataframe: every paper (2017-2025) with its embedding, last author, and outcome
# The SBERT embeddings were computed on iclr25v2 without "Anonymous", so row i of the
# embeddings is row i of the dataframe above
embedding_av = np.load(embeddings_dir / "embedding_abstracts_only_av.npy")
assert len(embedding_av) == len(iclr2025v2)

# Papers with the same abstract must have the same embedding
duplicates = iclr2025v2[iclr2025v2["abstract"].duplicated(keep=False)].groupby("abstract").groups
for indices in duplicates.values():
    assert np.allclose(embedding_av[indices[0]], embedding_av[indices[1]])

master = iclr2025v2.copy()
master.insert(0, "paper_idx", master.index)   # row of the paper in the embeddings

authors = master["authors"].str.split(",").apply(lambda names: [a.strip() for a in names if a.strip()])
author_ids = master["author_ids"].str.split(",").apply(lambda ids: [a.strip() for a in ids if a.strip()])

master["n_authors"] = authors.str.len()
master["last_author"] = authors.str[-1]
master["last_author_id"] = author_ids.str[-1]   # OpenReview ids only exist from 2021 (NaN before)
master["has_author_ids"] = master["year"] > 2020

# Accept vs. reject; workshop invites (2017-18) count as reject
master["decision_group"] = np.select(
    [
        master["decision"].str.startswith("Accept"),
        master["decision"].isin(["Reject", "Invite to Workshop Track"]),
        master["decision"] == "Withdrawn",
        master["decision"] == "Desk rejected",
    ],
    ["accept", "reject", "withdrawn", "desk_rejected"],
    default="other"
)
master["accepted"] = master["decision_group"] == "accept"
master["mean_score"] = master["scores"].apply(lambda s: np.mean(s) if len(s) > 0 else np.nan)

# Raw SBERT embedding (not normalized); load with np.stack(master["embedding"])
master["embedding"] = list(embedding_av)

master.to_parquet(embeddings_dir / "master.parquet", index=False)

print(f"Master dataframe: {len(master)} papers, {master['year'].min()}-{master['year'].max()}")
print(f"Saved to {embeddings_dir / 'master.parquet'}")

import pandas as pd

iclr2025v2 = pd.read_parquet("iclr25v2.parquet")

iclr2025v2 = iclr2025v2[
    (iclr2025v2["authors"] != "Anonymous") &
    (iclr2025v2["year"] > 2020)
]

iclr2025v2.to_parquet(
    "iclr25v2_no_anonymous.parquet",
    index=False
)
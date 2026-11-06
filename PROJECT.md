## Repository Structure

```
COS424_Project/
├── data/                               # Datasets, model weights, and precomputed embeddings
│
├── scripts/                            # Core modular Python library
│   ├── __init__.py                     # Package marker
│   ├── extract_dataset.py              # Author parsing, PI extraction, and metadata aggregation
│   ├── embeddings.py                   # Token pooling (mean/sep) and batched SBERT embedding pipeline
│   └── iclr_analysis.py                # Model downloader, embedding generator, and t-SNE wrapper
│
├── initial_analysis.ipynb              # Jupyter notebook export of initial analysis
├── environment.yml                     # Conda environment definition (`iclr`)
├── README.md                           # Brief project description and citation
└── PROJECT.md                          # Project onboarding, architecture, and usage guide
```

---

## 1. Setup the Anaconda environment

### If you are working on the HPC clusters (Adroit)
```bash
ml purge
ml anaconda3/2026.7 
conda env create -f environment.yml
conda activate iclr
```

### If you are working locally
```bash
conda env create -f environment.yml
conda activate iclr
```

### Updating the environment
If there are changes to the `environment.yml`, update your Anaconda environment:
```bash
conda env update -f environment.yml --prune
```

---

## 2. Codebase Walkthrough & Module Guide

### `scripts/extract_dataset.py`
Provides data wrangling and aggregation utilities for paper records:
- `parse_authors(raw_authors)`: Parses comma-separated author strings into clean lists.
- `extract_last_author(raw_authors)`: Extracts the final author listed on each paper, operating on the academic convention that the last author is typically the Principal Investigator / senior lab director.
- `extract_collaborators(raw_authors, pi)`: Isolates co-authors on each submission excluding the PI.
- `group_by_last_author(papers_df, ...)`: Groups the entire dataset by PI and aggregates their publication history (IDs, titles, abstracts, collaborators, decisions, review scores, keywords, labels, and paper count).

### `scripts/embeddings.py`
Implements PyTorch and Hugging Face Transformers embedding routines:
- `mean_pooling(token_embeds, attention_mask)`: Calculates attention-weighted average embeddings across token positions.
- `sep_pooling(token_embeds, attention_mask)`: Extracts representations at the `[SEP]` boundary token.
- `generate_embeddings(abstracts, tokenizer, model, device, batch_size)`: Batches and encodes paper abstracts using PyTorch `DataLoader` with GPU acceleration when available.

### `scripts/iclr_analysis.py`
High-level orchestrator for model caching, representation generation, and projection:
- `download_autotokenizer(model_name, target_dir)`: Downloads Hugging Face weights and tokenizers locally to `data/autotokenizer/` to allow offline cluster execution.
- `compute_text_embeddings_SBERT(...)`: Computes or loads precomputed SBERT representations for all paper abstracts (`_cls`, `_sep`, `_av`).
- `compute_tsne(...)`: Projects high-dimensional text embeddings into 2D coordinates using `openTSNE`. Can also be executed directly via command line:
  ```bash
  python -m scripts.iclr_analysis --model_name sentence-transformers/all-mpnet-base-v2 --target_dir data/autotokenizer
  ```

---

## 3. Analysis Pipeline in `initial_analysis.py`

The main notebook (`initial_analysis.py`) performs the following exploratory analysis steps:
1. **Dataset Loading**: Reads `data/iclr26v1.parquet` (or earlier years) containing papers and metadata.
2. **PI Aggregation**: Uses `group_by_last_author` to construct author-level profiles.
3. **CCDF Analysis**: Visualizes the Complementary Cumulative Distribution Function of papers per author to choose an informed threshold for active researchers ($k \ge 5$).
4. **Solo vs. Collaborative Analysis**: Analyzes independent authors versus collaborative PIs.
5. **Distribution of Papers per Author**: Plots linear and log-log distribution curves examining publication concentration.
6. **Abstract Embeddings & t-SNE Projections**: Computes or loads SBERT average embeddings and fits 2D t-SNE representations.
7. **Researcher Subplot Visualizations**: Highlights individual researcher portfolios (top authors by paper count) against the overall conference landscape, annotating their publication volume and acceptance rate.

---

## 4. Running on HPC (Slurm)

For computing heavy embeddings or training models on Princeton's Adroit cluster, submit jobs using the Slurm script:
```bash
sbatch jobs/job.slurm
```
This requests a GPU node (`--gres=gpu:1`), activates the `iclr` conda environment, and executes the analysis script non-interactively.

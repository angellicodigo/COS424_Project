TODO: update this .md file. This .md file will serve as a guide on how to understand the code

# 1. Setup the Anaconda environment

If you are working on the HPC clusters (Adroit)
```bash
ml purge
ml anaconda3/2026.7 
conda env create -f environment.yml
conda activate cos424
```
Else you can just do
```bash
conda env create -f environment.yml
conda activate cos424
```

If there are changes to the environment.yml, you need to update your Anaconda environment
```bash
conda env update -f environment.yml --prune
```


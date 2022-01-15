#! /bin/bash

#SBATCH -n 5
#SBATCH -A irel
#SBATCH --time=1-00:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

which python
cd ~/indic_wikibot/copernicus/KELM_WITA

python run.py bn

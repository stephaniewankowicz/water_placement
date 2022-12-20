#!/bin/bash
#$ -l h_vmem=20G
#$ -l mem_free=20G
#$ -t 1-1
#$ -l h_rt=40:00:00
#$ -pe smp 2
#$ -R yes
#$ -V

export PATH="/wynton/home/fraserlab/swankowicz/anaconda3/bin:$PATH"
#source=/wynton/home/fraserlab/swankowicz/anaconda3/envs/
source activate wat2

cd /wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/analysis
python KDE_full_protein.py 

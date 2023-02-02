#!/bin/bash
#$ -l h_vmem=10G
#$ -l mem_free=10G
#$ -t 1-15
#$ -l h_rt=20:00:00
#$ -pe smp 8
#$ -R yes
#$ -V

export PATH="/wynton/home/fraserlab/swankowicz/anaconda3/bin:$PATH"
source activate wat2

#cd /wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/analysis
PDB_file=/wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/analysis/test_proteins/pdb.txt
base_dir='/wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/analysis/test_proteins/'
bandwidth=/wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/analysis/bandwidths.txt

pdb=$(cat $PDB_file | awk '{ print $1 }' |head -n $SGE_TASK_ID | tail -n 1)
echo 'PDB:' ${pdb}
cd $base_dir
cd $pdb

#run intermediate output
/wynton/home/fraserlab/swankowicz/anaconda3/envs/wat2/bin/python /wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/analysis/place_all_full_protein.py --pdb $pdb

#cd /wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/analysis
for i in {1..15}; do
    band=$(cat $bandwidth | awk '{ print $1 }' |head -n $i | tail -n 1)
    /wynton/home/fraserlab/swankowicz/anaconda3/envs/wat2/bin/python /wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/analysis/KDE_full_protein.py --pdb $pdb --band $band -d /wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/analysis
done 

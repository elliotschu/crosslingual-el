#!/bin/bash
#$ -cwd
#$ -j y -o /home/eschumac/concept-linker/logs/linker_conda2.log
#$ -m beas
#$ -M eschumac@cs.jhu.edu
#$ -l mem_free=15G,ram_free=15G,gpu=1,hostname=!b0[123456789]*&!b10*&!c20*
#$ -pe smp 1
#$ -V
#$ -q g.q

#export PATH="/home/eschumac/anaconda3/bin:$PATH"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate linker

echo $PATH
export PYTHONPATH=${PYTHONPATH}:.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/NVIDIA/cuda-9.0/lib64/
python -V
CUDA_VISIBLE_DEVICES=`free-gpu -n 1` python codebase/linker.py --embedding elmo

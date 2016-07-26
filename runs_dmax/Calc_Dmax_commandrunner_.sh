#! /bin/bash
#$ -l h_rt=24:00:00
#$ -cwd
#$ -t 1-30
#$ -V
bash calc_Dmax__$SGE_TASK_ID.sh

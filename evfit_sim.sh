#!/bin/bash

# import required environment variables such as PYTHONPATH
#$ -v PYTHONPATH=/research/astro/gama/loveday/git/PhD_Luminosity_function
#$ -o /mnt/lustre/scratch/astro/loveday
# Combine error and output files
#$ -j y
# Job class (test.long = 1 week)
#$ -jc test.long
#$ -m eas
#$ -M J.Loveday@sussex.ac.uk
# Tell the SGE that this is an array job, with "tasks" to be
# numbered 1-10 - NB: emails are sent for every task!
#$ -t 1-10
# When a single command in the array job is sent to a compute
# node, it’s task number is stored in the variable SGE_TASK_ID,
# use the value of that variable to get the results you want.
#
# specify the queue with optional architecture spec following @@
#$ -q smp.q
# estimated runtime
##$ -l d_rt=08:00:00
# catch kill and suspend signals
#$ -notify
cd /research/astro/gama/loveday/Data/gama/
conda activate jon
python -V
python <<EOF
import os
taskid = os.environ['SGE_TASK_ID']
import evfit
evfit.ev_fit_sim(taskid-1)
EOF

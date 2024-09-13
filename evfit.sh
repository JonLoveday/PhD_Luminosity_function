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
#
# specify the queue with optional architecture spec following @@
#$ -q smp.q
#$ -l m_mem_free=4G
# estimated runtime
##$ -l d_rt=08:00:00
# catch kill and suspend signals
#$ -notify
cd /research/astro/gama/loveday/Data/gama/
module load Anaconda3
conda activate jon
python -V
python <<EOF
import evfit
evfit.ev_fit_II()
EOF

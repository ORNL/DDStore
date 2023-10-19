#!/bin/bash
#BSUB -env "all"
#BSUB -J HydraGNN
#BSUB -W 60
#BSUB -nnodes 1
#BSUB -o job-%J.out
#BSUB -e job-%J.out

[ -z $JOBID ] && JOBID=$LSB_JOBID
[ -z $JOBSIZE ] && JOBSIZE=$(((LSB_DJOB_NUMPROC-1)/42))
NP=$((JOBSIZE*6))

## Prepar
jskill all
\rm -f run0.log run1.log
jsrun -n$JOBSIZE bash -c "rm -f /dev/mqueue/*"
sleep 3

## For single run:
#time jsrun -n$NN -g1 -c6 -brs python -u vae-ddp.py --epochs 3 2>&1 | tee run1.log 

## For producer-consumer run:
time jsrun -n$NN -g0 -c1 -brs python -u vae-ddp-service.py --mq --producer 2>&1 > run0.log &
sleep 3
#time jsrun -n$NN -g1 -c6 -brs python -u vae-ddp-service.py --mq --consumer 2>&1 | tee run1.log 
time jsrun -n$NN -g1 -c6 -brs python -u vae-ddp.py --epochs 3 --mq --consumer 2>&1 | tee run1.log 

## Cleanup
sleep 3
jskill all


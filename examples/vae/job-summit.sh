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

cleanup() {
    jskill all
    \rm -f run0.log run1.log
    jsrun -n$JOBSIZE bash -c "rm -f /dev/mqueue/*"
    jsrun -n$JOBSIZE bash -c "ls /dev/mqueue/*"
}

export DDSTORE_VERBOSE=1
## For single run:
cleanup
(time jsrun -n$NP -g1 -c6 -brs python -u vae-ddp.py) 2>&1 | tee run-v0.log 

## For producer-consumer run:
cleanup
MASTER_PORT=8889 jsrun -n$NP -g0 -c1 -brs python -u vae-ddp-service.py --mq --producer 2>&1 > run-v1-role0.log &
sleep 3
(time MASTER_PORT=8890 jsrun -n$NP -g1 -c6 -brs python -u vae-ddp.py --mq --consumer) 2>&1 | tee run-v1-role1.log 

## For producer-consumer stream run:
cleanup
MASTER_PORT=8889 jsrun -n$NP -g0 -c1 -brs python -u vae-ddp-service.py --mq --producer --stream 2>&1 > run-v2-role0.log &
sleep 3
(time MASTER_PORT=8890 jsrun -n$NP -g1 -c6 -brs python -u vae-ddp.py --mq --consumer --stream) 2>&1 | tee run-v2-role1.log 
## Cleanup
cleanup


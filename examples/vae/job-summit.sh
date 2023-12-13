#!/bin/bash
#BSUB -env "all"
#BSUB -J HydraGNN
#BSUB -W 30
#BSUB -nnodes 4
#BSUB -o job-%J.out
#BSUB -e job-%J.out

[ -z $JOBID ] && JOBID=$LSB_JOBID
[ -z $JOBSIZE ] && JOBSIZE=$(((LSB_DJOB_NUMPROC-1)/42))
NP=$((JOBSIZE*6))

cleanup () {
    jskill all
    sleep 3
    jsrun -n$JOBSIZE -g0 -c1 -r1 -brs bash -c "rm -f /dev/mqueue/*"
    jsrun -n$JOBSIZE -g0 -c1 -r1 -brs bash -c "ls /dev/mqueue/*"
    sleep 3
}

set -x
export DDSTORE_VERBOSE=1

## For single run:
cleanup
time jsrun -n$NP -g1 -c7 -r6 -brs python -u vae-ddp.py --epochs=3 2>&1 | tee run-v0.log 

## For producer-consumer run:
cleanup
wait
MASTER_PORT=8889 jsrun -n$NP -g0 -c1 -r6 -brs python -u vae-ddp-service.py --epochs=3 --mq --producer 2>&1 > run-v1-role0.log &
sleep 3
time MASTER_PORT=8890 jsrun -n$NP -g1 -c6 -r6 -brs python -u vae-ddp.py --epochs=3 --mq --consumer 2>&1 | tee run-v1-role1.log 

## For producer-consumer stream run:
cleanup
MASTER_PORT=8889 jsrun -n$NP -g0 -c1 -r6 -brs python -u vae-ddp-service.py --epochs=3 --mq --producer --stream 2>&1 > run-v2-role0.log &
sleep 3
time MASTER_PORT=8890 jsrun -n$NP -g1 -c6 -r6 -brs python -u vae-ddp.py --epochs=3 --mq --consumer --stream 2>&1 | tee run-v2-role1.log 
## Cleanup
cleanup


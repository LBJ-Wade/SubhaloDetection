#!/bin/bash
COUNTER=1
while [  $COUNTER -lt 31 ]; do
    echo qsub runs_dmax/Calc_Dmax_commandrunner_$COUNTER.sh
    let COUNTER=COUNTER+1
done
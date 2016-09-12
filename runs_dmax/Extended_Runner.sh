#!/bin/bash
COUNTER=1
while [  $COUNTER -lt 31 ]; do
    qsub Calc_Dmax_commandrunner_$COUNTER.sh
    let COUNTER=COUNTER+1
done
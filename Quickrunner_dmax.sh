cd runs_dmax
COUNTER=1
while [  $COUNTER -lt 31 ]; do
    sh ./calc_Dmax__$COUNTER.sh
    let COUNTER=COUNTER+1
done
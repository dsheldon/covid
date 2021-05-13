#!/bin/bash
MODEL_NAME=SEIRD_renewal_ablation2
DATE=2021-05-09
DIR=/mnt/nfs/work1/sheldon/gcgibson/$MODEL_NAME/$DATE
#DIR=results
PLACES=${2:-../vis/countries.js}
echo $DIR
echo $PLACES

find -L $DIR -name "vis" -exec cp ../vis/index.html {} \; -exec cp $PLACES {}/places.js \;

rsync -avz --exclude="*samples*" $DIR/ doppler:/var/www/html/covid/results_$MODEL_NAME/

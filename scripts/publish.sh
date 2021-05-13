#!/bin/bash

DIR=/mnt/nfs/work1/sheldon/gcgibson/SEIRD_renewal/
PLACES=${2:-../vis/states.js}
echo $DIR
echo $PLACES

find -L $DIR -name "vis" -exec cp ../vis/index.html {} \; -exec cp $PLACES {}/places.js \;

rsync -avz --exclude="*samples*" $DIR/ doppler:/var/www/html/covid/ablation_renewal/

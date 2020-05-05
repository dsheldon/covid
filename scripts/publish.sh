#!/bin/bash

DIR=${1:-results}
echo $DIR

find -L $DIR -name "vis" -exec cp ../vis/index.html {} \;

rsync -avz --exclude="*samples*" $DIR/ doppler:/var/www/html/covid/$DIR/

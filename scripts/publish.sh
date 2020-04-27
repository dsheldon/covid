#!/bin/bash

DIR=${1:-vis}
echo $DIR

cp ../vis/index.html $DIR/

rsync -avz $DIR/ doppler:/var/www/html/covid/$DIR/

#!/bin/bash

MODEL=${1:-SEIRD/2020-05-03}
SRC=${2:-swarm2:covid/scripts/results}
DST=${3:-results}

rsync -avz --relative $SRC/./$MODEL/ $DST/

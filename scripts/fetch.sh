#!/bin/bash

DIR=${1:-out}

rsync -avz swarm2:covid/scripts/$DIR/ $DIR/

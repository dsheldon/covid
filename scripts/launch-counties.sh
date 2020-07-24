#!/bin/bash

EXTRA_ARGS=$@

root=results1
start='2020-03-04'
configs="counties"
forecast_dates="2020-05-31 2020-06-07 2020-06-14 2020-06-21 2020-06-28 2020-07-05"
forecast_dates="2020-05-10 2020-05-17 2020-05-24"
forecast_dates="2020-06-07 2020-07-05 2020-07-12"
forecast_dates="2020-07-19"
forecast_dates="2020-07-26"

max_places=500

for config in $configs; do
    for forecast_date in $forecast_dates; do
	prefix=$root/$config/$forecast_date
	echo "prefix is $prefix"

	logdir=log/$config/$forecast_date
	[ -d $logdir ] || mkdir -p $logdir

	i=0
	while IFS= read -r place && [ $i -lt $max_places ]; do

	    name=$place-$forecast_date-$config

	    echo "launching $name"

	    sbatch --job-name="$name" \
	    	--output="$logdir/$place.out" \
	    	--error="$logdir/$place.err" \
	    	--nodes=1 \
	    	--ntasks=1 \
	    	--mem=4000 \
	    	--partition=defq \
	    	./run_sir.sh "$place" --start $start --end $forecast_date --config $config --prefix $prefix $EXTRA_ARGS

	    ((i++))

	    sleep 0.1
	done < counties.txt
    done
done

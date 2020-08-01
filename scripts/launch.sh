#!/bin/bash

EXTRA_ARGS=$@

root=results1
start='2020-03-04'
configs="resample_80_last_10"
forecast_dates="2020-06-28 2020-07-05 2020-07-12 2020-07-19"
forecast_dates="2020-07-26"


states="US AS GU MP PR VI AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY"

for config in $configs; do
    for forecast_date in $forecast_dates; do
	prefix=$root/$config/$forecast_date
	echo "prefix is $prefix"

	for state in $states; do
	    name=$state-$forecast_date-$config

	    logdir=log/$config/$forecast_date
	    [ -d $logdir ] || mkdir -p $logdir

	    echo "launching $name"

	    sbatch --job-name=$name \
		--output=$logdir/$state.out \
		--error=$logdir/$state.err \
		--nodes=1 \
		--ntasks=1 \
		--mem=4000 \
		--partition=defq \
		./run_sir.sh $state --start $start --end $forecast_date --config $config --prefix $prefix $EXTRA_ARGS

	    sleep 0.1
	done
    done
done

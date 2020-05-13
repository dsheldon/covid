#!/bin/bash


root=results
start='2020-03-15'
configs="SEIRD_incident"
#forecast_dates="2020-04-04 2020-04-11  2020-04-18 2020-04-25"   # validation
forecast_dates="2020-05-03" # this week's submission


#states="AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY"
states="AL"
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
		--mem=5000 \
		--partition=defq \
		./run_sir.sh $state --start $start --end $forecast_date --config $config --prefix $prefix

	    sleep 0.1
	done
    done
done

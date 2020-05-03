#!/bin/bash


states="AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY"

for state in $states; do
    echo "launching $state"
    sbatch --job-name=$state \
      --output=log/result-$state.out \
      --error=log/result-$state.err \
      --nodes=1 \
      --ntasks=1 \
      --mem=5000 \
      --partition=defq \
       ./run_validation.sh $state
done

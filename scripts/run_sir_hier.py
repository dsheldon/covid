import numpyro
numpyro.enable_x64()

import sys
import argparse
import covid.util as util
import configs
import numpy as onp
import pandas as pd
import covid
if __name__ == "__main__":


    data =  util.load_data()
        # MI doesn't report on Sundays

    model_type=covid.models.SEIRD.SEIRD,
    start = '2020-03-04'
    end = None
    save = True
    num_warmup = 1000
    num_samples = 1000
    num_chains = 4
    num_prior_samples = 0            
    T_future=4*7
    prefix = "results"
    resample_low=0
    resample_high=100
    save_fields=['beta0', 'beta', 'sigma', 'gamma', 'dy0', 'dy', 'dy_future', 'dz0', 'dz', 'dz_future','dz_full','dy_full', 'y0', 'y', 'y_future', 'z0', 'z', 'z_future','det_prob','time_to_death' ]


    numpyro.enable_x64()
    #print(f"Running {place} (start={start}, end={end})")
    place_data = take(32, d.iteritems())

    


    print (place_data)
    sys.exit()

    
    print(" * running MCMC")
    mcmc_samples = model.infer(num_warmup=num_warmup, 
                               num_samples=num_samples)
    if resample_low > 0 or resample_high < 100:
        print(" * resampling")
        mcmc_samples = model.resample(low=resample_low, high=resample_high, **kwargs)

    # Prior samples
    prior_samples = None
    if num_prior_samples > 0:
        print(" * collecting prior samples")
        prior_samples = model.prior(num_samples=num_prior_samples)

    # In-sample posterior predictive samples (don't condition on observations)
    print(" * collecting in-sample predictive samples")
    post_pred_samples = model.predictive()

    # Forecasting posterior predictive (do condition on observations)
    print(" * collecting forecast samples")
    forecast_samples = model.forecast(T_future=T_future,T_old=T)
    
    if save:

        # Save samples
        path = Path(prefix) / 'samples'
        path.mkdir(parents=True, exist_ok=True)
        filename = path / f'{place}.npz'
        
        save_samples(filename,
                     prior_samples,
                     mcmc_samples, 
                     post_pred_samples,
                     forecast_samples,
                     save_fields=save_fields)
        
        path = Path(prefix) / 'summary'
        path.mkdir(parents=True, exist_ok=True)
        filename = path / f'{place}.txt'
        
        write_summary(filename, model.mcmc)

        
def save_samples(filename, 
                 prior_samples,
                 mcmc_samples, 
                 post_pred_samples,
                 forecast_samples,
                 save_fields=None):
    

    def trim(d):
        if d is not None:
            d = {k : v for k, v in d.items() if k in save_fields}
        return d
    onp.savez_compressed(filename, 
                         prior_samples = trim(prior_samples),
                         mcmc_samples = trim(mcmc_samples),
                         post_pred_samples = trim(post_pred_samples),
                         forecast_samples = trim(forecast_samples))



    # RI, CT, GU don't report on Saturdays/Sundays
    #   Oct 19 -- add WI (removed Oct 25)
    #   Oct 18 -- add KS
    
   # util.gen_forecasts(data,
    #                   args.place,
     #                  start=args.start,
      #                 prefix=args.prefix,
       #                model_type=config['model'],
        #               show=False)

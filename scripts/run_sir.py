import sys
import covid.util as util

if __name__ == "__main__": 
    place = sys.argv[1]
    data = util.load_state_data()
    util.run_place(data, place)
    util.gen_forecasts(data, place, show=False)

from sklearn.model_selection import ParameterSampler
import pandas as pd
import numpy as np
from HA import HA
from numpy.random import dirichlet
from time import time
#PARAMETERS:
validationMaxTimeAgo = 0.1 #fraction of overall time since the latest timestamp to the oldest to randomly sample from
validationSetSize = 6
noCells = 32
noSlots = 10
possibleSlots = [48, 49, 50, 51, 52, 72, 73, 96, 97, 120, 144, 168, 192, 216, 240, 264, 312, 336, 337, 360, 504]
n_iter = 1
bbox = { "latitude":[51.503349, 51.563349], "longitude": [-0.177061, -0.085061] }
random = False

# TO DO FROM SCRATCH. V
# df = load_gps_data("C:/Lanterne/predicio.csv").sort_values("local")
# # start = datetime.now()
# dfForValidationYPrep, validation_timestamps = prepare_df(df, noCells=noCells, bbox=bbox)
# validation_timestamps = np.random.choice(np.partition(validation_timestamps, int(len(validation_timestamps)*(1-validationMaxTimeAgo)))[int(len(validation_timestamps)*(1-validationMaxTimeAgo)):], 20, replace=False)
# validation_timestamps = pd.to_datetime(validation_timestamps)
# validationDataset = prepare_dataset(dfForValidationYPrep, validation_timestamps, noCells)

#

# TO DO FROM PREPREPARED DATA FROM CROWDLIVE_PREPROCESSING.PY. V
validationDataset = pd.read_pickle("C:/Lanterne/validationDataset01234.pickle").iloc[6:18]
# exit()
validation_timestamps = validationDataset.index
df = pd.read_csv("C:/Lanterne/smallerPrepreparedPredicio01234.csv")
df.local = pd.to_datetime(df.local)

#

#Automated check to see if data we have stretches back far enough
oldestTimestamp = min(validation_timestamps)
target_date = oldestTimestamp - pd.Timedelta(max(possibleSlots), "hours")
df_slice = df[
    (df.local.dt.hour == target_date.hour) & (df.local.dt.date == target_date.date())
]
if len(df_slice.index) == 0:
    print("ERROR: DATA DOES NOT STRETCH BACK {} HOURS FROM OLDEST VALIDATION TIMESTAMP: {}!".format(max(possibleSlots), oldestTimestamp))
    exit()


# optimal ratios so far, not completed analysis though yet as not enough data[0.41535052, 0.18150571, 0.17760467, 0.0936657 , 0.01253069,
        #   0.04342188, 0.06557791, 0.01034292]

Y = validationDataset.data

# Try for noSlots slots
param_grid = {
    "sample_proportions" : [dirichlet(np.ones(noSlots)) for _ in range(30)],
    "daysToSampleFrom" : [np.random.choice(possibleSlots, noSlots, replace=False) for _ in range(30)], #replace = true will allow for occasional number of slots below 5 
    "validation_timestamps" : [validation_timestamps], 
    "noCells" : [noCells],
    "bbox" : [bbox]
}

param_grids = [{ #grids for choosing days
    "sample_proportions" : [1/noSlots for _ in range(noSlots)],
    "daysToSampleFrom" : i,
    "validation_timestamps" : validation_timestamps, 
    "noCells" : noCells
    # "bbox" : bbox
}
for i in [np.random.choice(possibleSlots, noSlots, replace=False) for _ in range(n_iter)]
]

param_grids = [{ #grids for choosing samples and thus also dropping days
    "sample_proportions" : i,
    "daysToSampleFrom" : [48, 50, 52, 72, 73, 168, 169, 336, 504, 672],
    # "daysToSampleFrom" : list(np.arange(-7,-2)*-24),
    "validation_timestamps" : validation_timestamps, 
    "noCells" : noCells
    # "bbox" : bbox
}
for i in [ 
    dirichlet(np.ones(noSlots)) for _ in range(4)
    # Use above function for testing, below arrays were the final results of all tests organised by best value on test dataset (I took top 4, and also average of top 10, and 20, then validated them on a diff, larger dataset)

    # [0.11766342, 0.05807056, 0.02653818, 0.09557644, 0.08668869,
    #    0.08692645, 0.07926713, 0.28794903, 0.12663206, 0.03468805], #av of top 10best by far on other actual "validation" set of [6:18] :56.61
    #    [0.06338937, 0.01151326, 0.01811608, 0.01082097, 0.14479482,
    #    0.02037302, 0.16629781, 0.43726657, 0.05820452, 0.06922359], #1: 58.06
    #    [0.01721322, 0.06982554, 0.01286012, 0.2577425 , 0.04749347,
    #    0.16375124, 0.00272113, 0.2591271 , 0.14663074, 0.02263493], #2: 58.34
    #    [0.09331018, 0.12338302, 0.00555739, 0.07828642, 0.16200038,
    #    0.0458417 , 0.10712622, 0.30964388, 0.02243736, 0.05241345], #3: 57.71
    #    [0.00600888, 0.01720298, 0.07766654, 0.06694007, 0.13029272,
    #    0.06664688, 0.06008051, 0.34701461, 0.18733115, 0.04081565], #4: 57.34
    # [0.09549164, 0.06152503, 0.02715414, 0.11392028, 0.08212526,
    #    0.07341311, 0.07658173, 0.29534222, 0.12709614, 0.04735046] #av of top 20: 57.22

    #simple HA:
    # [0.4, 0.03, 0.07, 0.2, 0.3]
]]

# import pickle
# with open("tinkeringresults.pickle","rb") as f:
#     results = pickle.load(f)

start = time()
results = []
if random:
    for parameters in ParameterSampler(param_grid, n_iter):
        model = HA(**parameters)
        model.fit(df, Y)
        print(time() - start, "fit")
        start = time()
        newScore = model.score(df, Y)
        print(time() - start, "score")
        results.append([newScore, parameters]) # could only keep max value, but I want to see top 5, so am keeping array instead and then sorting
else:
    for parameters in param_grids:
        model = HA(**parameters)
        model.fit(df, Y)
        # print(time() - start, "fit")
        start = time()
        newScore = model.score(df, Y)
        print(time() - start, "score")
        results.append([newScore, parameters])
        
resultsNP = np.array(results)
topN = resultsNP[resultsNP[:,0].argsort()][:5] # N=5 here
for result in topN:
    print(result)
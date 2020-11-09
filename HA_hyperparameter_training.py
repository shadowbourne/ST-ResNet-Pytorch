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
noSlots = 8
possibleSlots = [48, 49, 50, 51, 52, 72, 73, 96, 97, 120, 144, 168, 192, 216, 240, 264, 312, 336, 337, 360, 504]
n_iter = 1
bbox = { "latitude":[51.503349, 51.563349], "longitude": [-0.177061, -0.085061] }
random = False

# TO DO FROM SCRATCH. V
# df = load_gps_data("C:/Lanterne/predicio.csv").sort_values("local")
# # start = datetime.now()
# dfForValidationYPrep, validation_timestamps = prepare_df(df, noCells=noCells, bbox=bbox)
# validation_timestamps = np.random.choice(np.partition(validation_timestamps, int(len(validation_timestamps)*validationMaxTimeAgo))[:int(len(validation_timestamps)*validationMaxTimeAgo)], validationSetSize, replace=False)
# validation_timestamps = pd.to_datetime(validation_timestamps)
# # validation_timestamps = df.local.nlargest(int(len(df.index)*validationMaxTimeAgo)).sample(validationSetSize)
# # makeValidationSet = HA(validation_timestamps=validation_timestamps)
# validationDataset = prepare_dataset(dfForValidationYPrep, validation_timestamps, noCells)

#

# TO DO FROM PREPREPARED DATA FROM CROWDLIVE_PREPROCESSING.PY. V
validationDataset = pd.read_pickle("C:/Lanterne/validationDataset2.pickle").iloc[0:6]
# print(validationDataset)
# exit()
validation_timestamps = validationDataset.index
df = pd.read_csv("C:/Lanterne/smallerPrepreparedPredicio2.csv")
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

param_grids = [{
    "sample_proportions" : [1/noSlots for _ in range(noSlots)],
    "daysToSampleFrom" : i,
    "validation_timestamps" : validation_timestamps, 
    "noCells" : noCells
    # "bbox" : bbox
}
for i in [np.random.choice(possibleSlots, noSlots, replace=False) for _ in range(n_iter)]
]
param_grids = [{
    "sample_proportions" : i,
    "daysToSampleFrom" : [48, 50, 52, 72, 73, 168, 169, 336],
    "validation_timestamps" : validation_timestamps, 
    "noCells" : noCells
    # "bbox" : bbox
}
for i in [ 
    dirichlet(np.ones(noSlots)) for _ in range(10)
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












    


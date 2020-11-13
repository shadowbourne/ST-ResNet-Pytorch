import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from crowdlive_preprocessing import prepare_df, get_matrix
from datetime import datetime
import numpy as np

class HA():
    def __init__(self, sample_proportions, daysToSampleFrom, validation_timestamps, noCells=32, bbox=None):
        self.sample_proportions = sample_proportions
        self.daysToSampleFrom = daysToSampleFrom
        self.validation_timestamps = validation_timestamps
        self.noCells = noCells
        self.bbox = bbox
        # self.criterion = criterion

    
    def fit(self, X, y):

        out_dfs = []
        # print(self.validation_timestamps)
        for dt in self.validation_timestamps:
            start = datetime.now()
            predicted_df = self.gps_forecast(dt, X)
            print(datetime.now() - start, "forecast")
            start = datetime.now()
            if self.bbox:
                predicted_df = prepare_df(predicted_df, for_benchmark=True, noCells=self.noCells, bbox=self.bbox)
            else:
                predicted_df = prepare_df(predicted_df, for_benchmark=True, noCells=self.noCells)
            out_dfs.append(get_matrix(predicted_df, noCells=self.noCells))
            print(datetime.now() - start, "matrix")
        self.Y_pred = np.array(out_dfs)
        #up until this comment is the same code that would be used for prediction
        return self
    
    def score(self, X=None, y="REQUIRED", sample_weight=None): 
        """
        y should be the matrices for the associated validation set timestamps
        X is not used
        returns the score from the following function
        #2d idea of wasserstein (earth movers distance) see for explanation: 
        # https://stackoverflow.com/questions/57562613/python-earth-mover-distance-of-2d-arrays
        takes: 2 arrays of 2d arrays
        returns: mean of the EMD for these arrays
        """


        values = []
        for i,j in zip(self.Y_pred, y):
            d = cdist(i,j)
            assignment = linear_sum_assignment(d)
            values.append(d[assignment].sum())
        return np.array(values).mean()/len(self.Y_pred[0])

        


    def gps_forecast(self, dt, df): #edited version from forecast.py
        
        out_df = pd.DataFrame(columns=list(df.columns))
        for i, hour_delta in enumerate(self.daysToSampleFrom):
            target_date = dt - pd.Timedelta(hour_delta, "hours")
            df_slice = df[
                (df.local.dt.hour == target_date.hour) & (df.local.dt.date == target_date.date())
            ]
            df_slice_sample = df_slice.sample(frac=self.sample_proportions[i])
            out_df = out_df.append(df_slice_sample, ignore_index=True)

        return out_df



import numpy as np
from numpy import histogram2d
import pandas as pd
from datetime import datetime

noCells = 32

def round_and_process_timestamp(dt):
    return datetime.utcfromtimestamp(int(dt)).replace(second=0, microsecond=0, minute=0) + pd.Timedelta(1, "h")

def process_timestamp(dt):
    return datetime.utcfromtimestamp(int(dt)) + pd.Timedelta(1, "h")

def load_gps_data(path):

    df = pd.read_csv(path)
    df["utc"] = df["timestamp"].apply(datetime.utcfromtimestamp)
    df["local"] = df["utc"] + pd.Timedelta(1, "h")
    df = df.drop(["horizontal_accuracy", "utc", "timestamp", "device_aid"], axis=1)
    return df

def simple_gps_forecast(dt, df): #edited version from forecast.py
    """
    A simple function that forecasts the mobile GPS coordinates for a
    given day and hour.

    Keyword arguments:
    dt -- datetime
    df -- mobile gps data to forecast with
    
    Output:
    out_df -- dataframe of predicted gps coords
    real_df -- actual gps coords (if available)
    """
    day = dt.date()

    out_df = pd.DataFrame(columns=list(df.columns))
    
    # the weights for the weighted average
    sample_proportions = [0.4, 0.03, 0.07, 0.2, 0.3]

    for i, day_delta in enumerate(np.arange(-7, -2)):
        target_date = day + pd.Timedelta(day_delta, "days")
        df_slice = df[
            (df.local.dt.hour == dt.hour) & (df.local.dt.date == target_date)
        ]
        df_slice_sample = df_slice.sample(frac=sample_proportions[i])
        out_df = out_df.append(df_slice_sample, ignore_index=True)

    # real_df = df[(df.local.dt.hour == hour.hour) & (df.local.dt.date == target_date)]
    return out_df

def prepare_df(df, for_benchmark=False, noCells=noCells, bbox={ "latitude":[51.448953, 51.546925], "longitude": [-0.259661, 0.027820] }):
    #NOTE BBOX CAN BE IMPLEMENTED AS AN ARRAY INSTEAD AND INSERTED INTO histogram2d as range=[[],[]]
    # bbox = { "latitude":[51.448953, 51.546925], "longitude": [-0.259661, 0.027820] } #lat[lower,upper] lng[lower,upper] 
    df = df[
        (df.latitude > bbox["latitude"][0]) & (df.latitude < bbox["latitude"][1]) & (df.longitude > bbox["longitude"][0]) & (df.longitude < bbox["longitude"][1])
    ]
    #old system, 60% slower
    # cellDeltaLat = ( bbox["latitude"][1] - bbox["latitude"][0] ) / noCells
    # cellDeltaLng = ( bbox["longitude"][1] - bbox["longitude"][0] ) / noCells
    # lat_bin = lambda x: np.floor((x - bbox["latitude"][0]) / cellDeltaLat)
    # lng_bin = lambda x: np.floor((x - bbox["longitude"][0]) / cellDeltaLng)
    # df["latitude"] = df.latitude.map(lat_bin)
    # df["longitude"] = df.longitude.map(lng_bin)

    if for_benchmark:
        return df

    df["localHour"] = df.local.map(lambda x: x.replace(second=0, microsecond=0, minute=0)) # very slow but needed to get timestamps
    timestamps = df.localHour[(df.localHour.dt.hour > 7) & (df.localHour.dt.hour < 19)].sort_values().unique()
    timestamps = df.localHour.sort_values().unique()

    return df, timestamps

def prepare_dataset(df, timestamps, noCells=noCells):
    dataset = pd.DataFrame(columns=["data"])
    dataset.index.name = "date"
    for dt in timestamps:
        dataset.loc[dt, "data"] = get_matrix(df, noCells, dt)
    return dataset

def get_matrix(df, noCells=noCells, dt=None):
    if dt:
        df = df[
                (df.localHour == dt)
            ]
    return histogram2d(df.latitude, df.longitude, bins=noCells)[0]

    # old system, 60% slower
    # df = df.groupby(["latitude", "longitude"]).size()
    # df = df.fillna(0)
    # matrix = np.zeros((noCells, noCells))
    # for index in df.index:
    #     matrix[int(index[1])][int(index[0])] = df.loc[index]
    # # print(datetime.now()-start)
    # return matrix

from time import time
if __name__ == '__main__':
    #FOR HA TUNING. V
    start = time()
    df = load_gps_data("C:/Lanterne/predicio.csv").sort_values("local")
    print(start - time())
    start = time()
    # start = datetime.now() 51.485174, -0.095863
    bbox = { "latitude":[51.455174, 51.515174], "longitude": [-0.141863, -0.049863] } #selected so a 32 by 32 grid will be 200m each
    dfForValidationYPrep, validation_timestamps = prepare_df(df, noCells=32, bbox=bbox)
    dfForValidationYPrep.to_csv("smallerPrepreparedPredicio2.csv")
    print(start - time())
    start = time()
    validation_timestamps = np.random.choice(np.partition(validation_timestamps, int(len(validation_timestamps)*0.1))[:int(len(validation_timestamps)*0.1)], 20, replace=False)
    validation_timestamps = pd.to_datetime(validation_timestamps)
    dataset = prepare_dataset(dfForValidationYPrep, validation_timestamps, noCells=32)
    print(start - time())
    start = time()
    dataset.to_pickle("validationDataset2.pickle")
    print(start - time())

    exit() 
    #FOR CNN. V
    df = load_gps_data("C:/Users/shadow/Downloads/predicio.csv")
    df, timestamps = prepare_df(df)
    dataset = prepare_dataset(df, timestamps, noCells)
    dataset.to_csv("predicio_dataset.csv")
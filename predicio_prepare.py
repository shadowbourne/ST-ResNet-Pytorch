import numpy as np
import pandas as pd
from datetime import datetime
def load_gps_data(path):
    df = pd.read_csv(path)
    df["utc"] = df["timestamp"].apply(datetime.utcfromtimestamp)
    df["local"] = df["utc"] + pd.Timedelta(1, "h")
    df = df.drop(["horizontal_accuracy", "utc"], axis=1)
    return df



def prepare_df(df):
    bbox = { "latitude":[51.448953, 51.546925], "longitude": [-0.259661, 0.027820] } #lat[lower,upper] lng[lower,upper]
    df = df[
        (df.latitude > bbox["latitude"][0]) & (df.latitude < bbox["latitude"][1]) & (df.longitude > bbox["longitude"][0]) & (df.longitude < bbox["longitude"][1])
    ]
    df["localHour"] = df.local.map(lambda x: pd.Timestamp(x.replace(second=0, microsecond=0, minute=0)))
    df = df.drop(["timestamp", "device_aid", "local"], axis=1)
    cellDeltaLat = ( bbox["latitude"][1] - bbox["latitude"][0] ) / noCells
    cellDeltaLng = ( bbox["longitude"][1] - bbox["longitude"][0] ) / noCells
    lat_bin = lambda x: int(np.floor((x - bbox["latitude"][0]) / cellDeltaLat))
    lng_bin = lambda x: int(np.floor((x - bbox["longitude"][0]) / cellDeltaLng))
    # df = df.apply(collatedLambda, axis=1, args=(bbox,cellDeltaLat,cellDeltaLng,))
    df["latitude"] = df.latitude.map(lat_bin)
    df["longitude"] = df.longitude.map(lng_bin)
    timestamps = df.localHour.sort_values().unique()
    # df = df.groupby(["latitude", "longitude"])
    return df, timestamps

def prepare_dataset(df, timestamps, noCells):
    dataset = pd.DataFrame(columns=["data"])
    dataset.index.name = "date"
    for dt in timestamps:
        dataset.loc[dt, "data"] = get_matrix(df, dt, noCells)
    return dataset

def get_matrix(df, dt, noCells):
    # try:
    df = df[
            (df.localHour == dt)
        ]
    # except:
    #     print(dt)
    #     print(df.localHour == dt)
    #     print(df.loc[:,"localHour"] == dt)
    df = df.groupby(["latitude", "longitude"]).count()
    # df_indices = df.index
    matrix = np.zeros((noCells, noCells))
    for index in df.index:
        matrix[index[1]][index[0]] = df.loc[index]
    # for y in range(noCells):
    #     for x in range(noCells):
    #         matrix[y][x] = len(df[ (df.latitude==y) & (df.longitude==x) ].index)
    return matrix

noCells = 32
df = load_gps_data("C:/Users/shadow/Downloads/predicio.csv")
df, timestamps = prepare_df(df)
dataset = prepare_dataset(df, timestamps, noCells)
dataset.to_pickle("londonPredicio.pickle")
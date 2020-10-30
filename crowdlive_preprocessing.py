
# from helpers import load_gps_data
# import sys
# noCells = 32 
# datetime(2020,9,11,13)


#Old system
# cellCenter = noCells // 2
# cellSize = 0.1 # in km
# # Estimates for how large the cells should be
# cellDeltaLat = cellSize / 70
# cellDeltaLng = cellSize / 110

# bbox = [[51.448953, 51.546925], [-0.259661, 0.027820]]
# cellDeltaLat = ( bbox[0][1] - bbox[0][0] ) / noCells
# cellDeltaLng = ( bbox[1][1] - bbox[1][0] ) / noCells

# halfMatrixDeltaLat = cellDeltaLat * noCells / 2
# halfMatrixDeltaLng = cellDeltaLng * noCells / 2

# 51.546925, -0.259661
# 51.448953, 0.027820
# def collatedLambda(row, bbox, cellDeltaLat, cellDeltaLng):
#     row["localHour"] = row.local.replace(second=0, microsecond=0, minute=0)
#     row.latitude = np.floor((row.latitude - bbox["latitude"][0]) / cellDeltaLat)
#     row.longitude = np.floor((row.longitude - bbox["longitude"][0]) / cellDeltaLng)
#     return row


import numpy as np
import pandas as pd
from datetime import datetime


noCells = 32



def load_gps_data(path):
    df = pd.read_csv(path)
    df["utc"] = df["timestamp"].apply(datetime.utcfromtimestamp)
    df["local"] = df["utc"] + pd.Timedelta(1, "h")
    df = df.drop(["horizontal_accuracy", "utc"], axis=1)
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

def prepare_df(df, for_benchmark=False, noCells=noCells):
    bbox = { "latitude":[51.448953, 51.546925], "longitude": [-0.259661, 0.027820] } #lat[lower,upper] lng[lower,upper]
    df = df[
        (df.latitude > bbox["latitude"][0]) & (df.latitude < bbox["latitude"][1]) & (df.longitude > bbox["longitude"][0]) & (df.longitude < bbox["longitude"][1])
    ]
    df = df.drop(["timestamp", "device_aid", "local"], axis=1)
    cellDeltaLat = ( bbox["latitude"][1] - bbox["latitude"][0] ) / noCells
    cellDeltaLng = ( bbox["longitude"][1] - bbox["longitude"][0] ) / noCells
    lat_bin = lambda x: np.floor((x - bbox["latitude"][0]) / cellDeltaLat)
    lng_bin = lambda x: np.floor((x - bbox["longitude"][0]) / cellDeltaLng)
    # df = df.apply(collatedLambda, axis=1, args=(bbox,cellDeltaLat,cellDeltaLng,))
    df["latitude"] = df.latitude.map(lat_bin)
    df["longitude"] = df.longitude.map(lng_bin)
    if for_benchmark:
        return df

    df["localHour"] = df.local.map(lambda x: x.replace(second=0, microsecond=0, minute=0))
    timestamps = df.localHour.sort_values().unique()
    # df = df.groupby(["latitude", "longitude"])
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
    df = df.groupby(["latitude", "longitude"]).size()
    df = df.fillna(0)
    matrix = np.zeros((noCells, noCells))
    for index in df.index:
        matrix[int(index[1])][int(index[0])] = df.loc[index]
    return matrix


if __name__ == '__main__':
    df = load_gps_data("C:/Users/shadow/Downloads/predicio.csv")
    df, timestamps = prepare_df(df)
    dataset = prepare_dataset(df, timestamps, noCells)
    dataset.to_csv("predicio_dataset.csv")


#datetime(x.dt.year, x.dt.day, x.dt.day, x.dt.hour)

# def forecast(df, dt, lat, lng):
#     trainingDf = pd.DataFrame(columns=["crowdFlowWindows", "y"]) #possible to add weather here
    
#     for 

## For https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8526506 implementation (not the one we are now doing)

# def prepare_matrix(df, dt, lowerLat, lowerLng):
#     df_slice = df[
#         (df.localHour == dt)
#     ]
#     matrix = np.empty((noCells, noCells))
#     #lng = x, lat = y
#     for y in range(noCells):
#         for x in range(noCells):
#             matrix[y][x] = len(df[
#                 (df.latitude > lowerLat + y*cellDeltaLat) & (df.latitude < lowerLat + (y+1)*cellDeltaLat) & (df.longitude > lowerLng + x*cellDeltaLng) & (df.longitude < lowerLng + (x+1)*cellDeltaLng)
#             ].index)
#     return matrix

# print(prepare_windows(df, datetime(2020,9,11,13), 51.906640, -0.934136))
# sys.exit(1)

#For ST-ResNet
# step = 0.2
# to_bin = lambda x: np.floor(x / step) * step
# df["latbin"] = df.Latitude.map(to_bin)
# df["lonbin"] = df.Longitude.map(to_bin)
# groups = df.groupby(("latbin", "lonbin"))








## For https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8526506 implementation (not the one we are now doing)

# def prepare_windows(df, dt, lat, lng):
#     lowerLat = lat - halfMatrixDeltaLat
#     lowerLng = lng - halfMatrixDeltaLng
#     df_slice = df[
#         (df.latitude > lowerLat) & (df.latitude < lat + halfMatrixDeltaLat) & (df.longitude > lowerLng) & (df.longitude < lng + halfMatrixDeltaLng)
#     ]
#     windowIntToCheck = [1,2,24,168] #Add whatever windows of interest
#     windowDatetimes = [dt - timedelta(hours=window) for window in windowIntToCheck]
#     windowsArray = []
#     for windowDt in windowDatetimes:
#         windowsArray.append(prepare_matrix(df_slice, windowDt, lowerLat, lowerLng))
#     return windowsArray

# def prepare_matrix(df, dt, lowerLat, lowerLng):
#     df_slice = df[
#         (df.localHour == dt)
#     ]
#     matrix = np.empty((noCells, noCells))
#     #lng = x, lat = y
#     for y in range(noCells):
#         for x in range(noCells):
#             matrix[y][x] = len(df[
#                 (df.latitude > lowerLat + y*cellDeltaLat) & (df.latitude < lowerLat + (y+1)*cellDeltaLat) & (df.longitude > lowerLng + x*cellDeltaLng) & (df.longitude < lowerLng + (x+1)*cellDeltaLng)
#             ].index)
#     return matrix

# def optimize_SARIMA(parameters_list, d, D, s, exog):
#     """
#         Return dataframe with parameters, corresponding AIC and SSE
        
#         parameters_list - list with (p, q, P, Q) tuples
#         d - integration order
#         D - seasonal integration order
#         s - length of season
#         exog - the exogenous variable
#     """
    
#     results = []
    
#     for param in tqdm_notebook(parameters_list):
#         try: 
#             model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
#         except:
#             continue
            
#         aic = model.aic
#         results.append([param, aic])
        
#     result_df = pd.DataFrame(results)
#     result_df.columns = ['(p,q)x(P,Q)', 'AIC']
#     #Sort in ascending order, lower AIC is better
#     result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
#     return result_df

    
# p = range(0, 4, 1)
# d = 1
# q = range(0, 4, 1)
# P = range(0, 4, 1)
# D = 1
# Q = range(0, 4, 1)
# s = 4
# parameters = product(p, q, P, Q)
# parameters_list = list(parameters)
# print(len(parameters_list))

# result_df = optimize_SARIMA(parameters_list, 1, 1, 4, data['data'])
# print(result_df)

# # best_model = SARIMAX(data['data'], order=(0, 1, 2), seasonal_order=(0, 1, 2, 4)).fit(dis=-1)
# # print(best_model.summary())

# # best_model.plot_diagnostics(figsize=(15,12));

# # data['arima_model'] = best_model.fittedvalues
# # data['arima_model'][:4+1] = np.NaN
# # forecast = best_model.predict(start=data.shape[0], end=data.shape[0] + 8)
# # forecast = data['arima_model'].append(forecast)
# # plt.figure(figsize=(15, 7.5))
# # plt.plot(forecast, color='r', label='model')
# # plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
# # plt.plot(data['data'], label='actual')
# # plt.legend()
# # plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import sys
import time
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
DATA_PATH = '../data/'
INPUT_PATH = '../data/input/'
SUBMIT_PATH = '../data/submit/'
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def timestamp_p(df):
    df['year'] = df['timestamp'].dt.year
    df['year'] = df['year'].map({2015:0, 2016:1, 2017:2, 2018:3}).astype(np.int8)
    df['hour'] = df['timestamp'].dt.hour.astype(np.int8)
    df['month'] = df['timestamp'].dt.month.astype(np.int8)
    df['day'] = df['timestamp'].dt.day.astype(np.int8)
    df['dayofweek'] = df['timestamp'].dt.dayofweek.astype(np.int8)
    return df

def model_input():
    train = pd.read_csv(DATA_PATH + 'train.csv', parse_dates=['timestamp'])  # 2016-01-01 ： 2016-12-31 20216100
    train = train.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')# 19869988 # 去除异常数据
    test = pd.read_csv(DATA_PATH + 'test.csv', parse_dates=['timestamp'])  # 2017-01-01 ：2018-12-31  41697600
    building_metadata = pd.read_csv(DATA_PATH + 'building_metadata.csv')
    weather_train = pd.read_csv(DATA_PATH + 'weather_train.csv', parse_dates=['timestamp'])
    weather_test = pd.read_csv(DATA_PATH + 'weather_test.csv', parse_dates=['timestamp'])

    # 对齐时间戳
    weather = pd.concat([weather_train, weather_test], ignore_index=True)
    weather_key = ['site_id', 'timestamp']
    temp_skeleton = weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()
    temp_skeleton['temp_rank'] = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.date])['air_temperature'].rank('average')
    df_2d = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.hour])['temp_rank'].mean().unstack(level=1)

    # Subtract the columnID of temperature peak by 14, getting the timestamp alignment gap.
    # 获取每个地区温度最高的时间点,假设最高温度出现在当地14点，进行时间点的对齐
    site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)
    site_ids_offsets.index.name = 'site_id'

    def timestamp_align(df):
        df['offset'] = df.site_id.map(site_ids_offsets)
        df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))
        df['timestamp'] = df['timestamp_aligned']
        del df['timestamp_aligned']
        return df
    weather_train = timestamp_align(weather_train)
    weather_test = timestamp_align(weather_test)
    #填充缺失值,使用插值方法
    #weather_train = weather_train.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
    #weather_test = weather_test.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))


    train = timestamp_p(train)
    test = timestamp_p(test)

    train = train.merge(building_metadata, how='left', on=['building_id'])
    test = test.merge(building_metadata, how='left', on=['building_id'])
    train = train.merge(weather_train, how='left', on=['site_id', 'timestamp'])
    test = test.merge(weather_test, how='left', on=['site_id', 'timestamp'])

    train['site_id'] = train['site_id'].astype(np.int8)
    train['meter'] = train['meter'].astype(np.int8)
    train['building_id'] = train['building_id'].astype(np.int16)
    test['site_id'] = test['site_id'].astype(np.int8)
    test['meter'] = test['meter'].astype(np.int8)
    test['building_id'] = test['building_id'].astype(np.int16)

    train['is_vacation_month'] = np.int8(0)
    train.loc[(train['primary_use'] == 'Education') & (train['month'] >= 6) & (train['month'] <= 8), 'is_vacation_month'] = np.int8(1)
    test['is_vacation_month'] = np.int8(0)
    test.loc[(test['primary_use'] == 'Education') & (test['month'] >= 6) & (test['month'] <= 8), 'is_vacation_month'] = np.int8(1)
    beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9),
                (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]
    for item in beaufort:
        train.loc[(train['wind_speed'] >= item[1]) & (train['wind_speed'] < item[2]), 'beaufort_scale'] = item[0]
        test.loc[(test['wind_speed'] >= item[1]) & (test['wind_speed'] < item[2]), 'beaufort_scale'] = item[0]

    data = pd.concat([train, test], ignore_index=True)
    data['primary_use'] = data['primary_use'].map(dict(zip(data['primary_use'].unique(), range(data['primary_use'].nunique()))))
    data['square_feet'] = np.log(data['square_feet'])

    train = data[data['year'] <= 1]
    test = data[data['year'] > 1]


    dates_range = pd.date_range(start='2015-12-31', end='2019-01-01')
    us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
    train['is_holiday'] = (train['timestamp'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)
    test['is_holiday'] = (test['timestamp'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)
    train.loc[(train['dayofweek'] == 5) | (train['dayofweek'] == 6), 'is_holiday'] = 1
    test.loc[(test['dayofweek']) == 5 | (test['dayofweek'] == 6), 'is_holiday'] = 1

    #填充缺失值
    # def mean_without_overflow_fast(col):
    #     col /= len(col)
    #     return col.mean() * len(col)
    # missing_values = (100 - train.count() / len(train) * 100).sort_values(ascending=False)
    # missing_features = train.loc[:, missing_values > 0.0]
    # missing_features = missing_features.apply(mean_without_overflow_fast)
    # for key in train.loc[:, missing_values > 0.0].keys():
    #     if key == 'year_built' or key == 'floor_count':
    #         train[key].fillna(math.floor(missing_features[key]), inplace=True)
    #         test[key].fillna(math.floor(missing_features[key]), inplace=True)
    #     else:
    #         train[key].fillna(missing_features[key], inplace=True)
    #         test[key].fillna(missing_features[key], inplace=True)


    del train['timestamp'], train['offset'], train['row_id']
    del test['timestamp'], test['offset'], test['meter_reading']
    test['row_id'] = test['row_id'].astype(int)
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    print(f'use memory {train.memory_usage().sum() / 1024**2}',train.info())
    print(f'use memory {test.memory_usage().sum() / 1024**2}',test.info())
    with open(INPUT_PATH + 'train.pk', 'wb') as train_f:
        pickle.dump(train, train_f)
    with open(INPUT_PATH + 'test.pk', 'wb') as test_f:
        pickle.dump(test, test_f)

def data_processing():
    train = pd.read_pickle(INPUT_PATH + 'train.pk')
    test = pd.read_pickle(INPUT_PATH + 'test.pk')

    print(train.info())
    print(test.info())
if __name__ == "__main__":
    model_input()
    #data_processing()
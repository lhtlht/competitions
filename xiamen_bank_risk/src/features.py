import pandas as pd
import numpy as np
import pickle
import sys
import time
import warnings
import time
import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
P_DATA = "./data/"
P_SUBMIT = "./submit/"

def feature_labelRate(train, test, labelName, featureName):
    feature_labelRate_df = train.groupby([featureName], as_index=False)[labelName].agg({
        f'{featureName}_labelRate': 'mean',
        f'{featureName}_labelCount': 'sum',
        f'{featureName}_count': 'size'})
    feature_labelRate_df = feature_labelRate_df[feature_labelRate_df[f'{featureName}_count']>300]
    train = train.merge(feature_labelRate_df, how='left', on=featureName)
    test = test.merge(feature_labelRate_df, how='left', on=featureName)

    return train, test

if __name__ == "__main__":
    train = pd.read_csv(P_DATA + 'train.csv', encoding='utf-8')
    train_target = pd.read_csv(P_DATA + 'train_target.csv', encoding='utf-8')
    train_data = train.merge(train_target, how='left', on='id')
    test_data = pd.read_csv(P_DATA + 'test.csv', encoding='utf-8')
    del train_data['id']
    train_data.drop_duplicates(inplace=True)
    data = pd.concat([train_data, test_data], ignore_index=True)


    # 特征处理
    age_mean = int(data['age'].mean())
    data.loc[(data['age'] == 117), 'age'] = age_mean

    for i in range(79):
        data.loc[(data[f'x_{i}'] == -999), f'x_{i}'] = -1
    data['x_sum'] = data[[f'x_{i}' for i in range(79)]].sum(axis=1)
    data['dist'] = data['dist'].astype(str)
    data['dist_len'] = data['dist'].apply(lambda x: len(x))
    data['province'] = data['dist'].apply(lambda x: x[0:2] if len(x) != 7 else x)
    data['post'] = data['dist'].apply(lambda x: x[2:3] if len(x) != 7 else x)
    data['city'] = data['dist'].apply(lambda x: x[3:4] if len(x) != 7 else x)
    data['delivery'] = data['dist'].apply(lambda x: x[4:6] if len(x) != 7 else x)
    data['beginDate'] = data['certValidBegin'].apply(lambda x: time.strftime("%Y-%m-%d", time.localtime(x)))
    data['stopDate'] = data['certValidStop'].apply(lambda x: time.strftime("%Y-%m-%d", time.localtime(x)) if x < 10000000000 else 'null')

    data['certDayDiff'] = data.apply(lambda x:
                                     (datetime.datetime.strptime(x['stopDate'], '%Y-%m-%d') - datetime.datetime.strptime(x['beginDate'], '%Y-%m-%d')).days if x['stopDate']!='null' else -1, axis=1)
    data['certMonthDiff'] = data.apply(lambda x:
                                     (datetime.datetime.strptime(x['stopDate'], '%Y-%m-%d') - datetime.datetime.strptime(x['beginDate'], '%Y-%m-%d')).days//30 if x['stopDate'] != 'null' else -1, axis=1)
    # 是否少数民族
    data['is_han'] = data['ethnic'].apply(lambda x: 1 if x == 0 else 0)
    # 是否周末
    data['weekend'] = data['weekday'].apply(lambda x: 1 if x >= 6 else 0)
    # 组合特征
    data['job_edu'] = str(data['job']) + '_' + str(data['highestEdu'])

    train = data[~data.target.isnull()]
    test = data[data.target.isnull()]
    with open(P_DATA + 'train.pk', 'wb') as train_f:
        pickle.dump(train, train_f)
    with open(P_DATA + 'test.pk', 'wb') as test_f:
        pickle.dump(test, test_f)
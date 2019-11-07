from feature import *
from model import *
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import gc
import zipfile
import seaborn as sns
from tqdm import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import warnings
warnings.filterwarnings("ignore")

def evel_model(eval_df):
    eval_df['y'] = np.log(eval_df['y'] + 1)
    eval_df['p'] = np.log(eval_df['p'] + 1)
    return np.sqrt(np.sum((eval_df['p'] - eval_df['y']) * (eval_df['p'] - eval_df['y'])) / eval_df.shape[0])

offline = False
train = pd.read_pickle(INPUT_PATH + 'train.pk')
if offline == True:
    train_model = train[train['month']<12]
    test_model = train[train['month']==12]
else:
    test = pd.read_pickle(INPUT_PATH + 'test.pk')
    train_model = train
    test_model = test
train_model['meter_reading_log'] = np.log1p(train_model['meter_reading'])
# 训练
numerical_features = ['year', 'month', 'hour', 'dayofweek', 'day', 'is_holiday',
                      'square_feet', 'year_built', 'floor_count',
                      'air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed'
                      ]
category_features = ['building_id', 'site_id', 'meter', 'primary_use'
                      ]
label_name = "meter_reading_log"
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.1,
    'min_child_samples': 5,
    'min_child_weight': 0.01,
    'subsample_freq': 1,
    'num_leaves': 31,
    'max_depth': -1,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'reg_alpha': 0,
    'reg_lambda': 5,
    'verbose': -1,
    'random_state': 4590,
    'n_jobs': 6,
}

train_model.reset_index(inplace=True,drop=True)
test_model.reset_index(inplace=True,drop=True)
features = category_features + numerical_features
train_x = train_model[features]
train_y = train_model[label_name]
test_x = test_model[features]

n_fold = 5
count_fold = 0
preds_list = list()
#oof = np.zeros(train_x.shape[0])
kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=2019)
kfold = kfolder.split(train_x, train_y)
for train_index, vali_index in kfold:
    print("training......fold",count_fold)
    count_fold = count_fold + 1
    k_x_train = train_x.loc[train_index]
    k_y_train = train_y.loc[train_index]
    k_x_vali = train_x.loc[vali_index]
    k_y_vali = train_y.loc[vali_index]

    dtrain = lgb.Dataset(k_x_train, k_y_train)
    dvalid = lgb.Dataset(k_x_vali, k_y_vali, reference=dtrain)
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    if 'sample_weight' in train.columns:
        lgb_model = lgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_vali, k_y_vali)],
                                  early_stopping_rounds=200, verbose=False, eval_metric="mae",
                                  sample_weight=train.loc[train_index]['sample_weight'],
                                  categorical_feature=category_features)
    else:
        lgb_model = lgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_vali, k_y_vali)],
                                  early_stopping_rounds=200, verbose=False, eval_metric="mae",
                                  categorical_feature=category_features)
    #k_pred = lgb_model.predict(k_x_vali, num_iteration=lgb_model.best_iteration_)
    pred = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration_)

    preds_list.append(pred)
    #oof[vali_index] = k_pred

preds_array = np.array(preds_list)
preds = list(preds_array.mean(axis=0))

# 评测
if offline:
    eval_df = pd.DataFrame()
    eval_df['y'] = test_model['meter_reading']
    eval_df['p'] = preds
    eval_df['p'] = np.expm1(eval_df['p'])
    eval_df.loc[eval_df['p'] < 0, 'p'] = 0
    eval_df['p'] = eval_df['p'].round(4)

    score = evel_model(eval_df)
    print(f'test-score:{score}')
else:
    submit_df = pd.DataFrame()
    submit_df['row_id'] = test_model['row_id'].astype(np.int32)
    submit_df['meter_reading'] = preds
    submit_df['meter_reading'] = np.expm1(submit_df['meter_reading'])
    submit_df.loc[submit_df['meter_reading'] < 0, 'meter_reading'] = 0
    submit_df['meter_reading'] = submit_df['meter_reading'].round(4)

    file_name  = 'submit1107_1'
    submit_df.to_csv(SUBMIT_PATH + f'{file_name}.csv', index=False)

    z = zipfile.ZipFile(SUBMIT_PATH + f'{file_name}.zip', 'w')
    z.write(SUBMIT_PATH + f'{file_name}.csv')
    z.close()


'''
test-score:1.0509966598281046 线上：1.17
test-score:1.034806138868842
test-score:1.0293598378013888
test-score:1.010172550892753

'''
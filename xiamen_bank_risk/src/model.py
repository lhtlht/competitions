import pandas as pd
import numpy as np
import sys
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from scipy import sparse
from scipy.sparse import csr_matrix
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'n_estimators': 500,
    'metric': 'mae',
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'min_child_weight': 0.01,
    'subsample_freq': 1,
    'num_leaves': 31,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'reg_alpha': 0,
    'reg_lambda': 5,
    'verbose': -1,
    'random_state': 4590,
    'n_jobs': -1,

}
xgb_params = {
        'booster': 'gbtree',
        'learning_rate': 0.01,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'objective': 'reg:linear',
        'n_estimators': 10000,
        'min_child_weight': 3,
        'gamma': 0,
        'silent': True,
        'n_jobs': -1,
        'random_state': 4590,
        'reg_alpha': 2,
        'reg_lambda': 0.1,
        'alpha': 1,
        'verbose': 1
    }


from sklearn.metrics import f1_score

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

def multi_column_LabelEncoder(df,columns,rename=True):
    le = LabelEncoder()
    for column in columns:
        #print(column,"LabelEncoder......")
        le.fit(df[column])
        df[column+"_index"] = le.transform(df[column])
        if rename:
            df.drop([column], axis=1, inplace=True)
            df.rename(columns={column+"_index":column}, inplace=True)
    print('LabelEncoder Successfully!')
    return df

def reg_model(train, test, label_name, model_type, numerical_features, category_features1, category_features2, seed):
    import lightgbm as lgb
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    train.reset_index(inplace=True,drop=True)
    test.reset_index(inplace=True,drop=True)
    if model_type == 'rf':
        train.fillna(0, inplace=True)

    combine = pd.concat([train, test], axis=0)
    combine = multi_column_LabelEncoder(combine, category_features1, rename=True)
    combine[category_features1+category_features2] = combine[category_features1+category_features2].astype('category')
    train = combine[:train.shape[0]]
    test = combine[train.shape[0]:]

    features = category_features1+category_features2 + numerical_features
    train_x = train[features]
    train_y = train[label_name]
    test_x = test[features]

    n_fold = 5
    count_fold = 0
    preds_list = list()
    oof = np.zeros(train_x.shape[0])
    kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    kfold = kfolder.split(train_x, train_y)
    for train_index, vali_index in kfold:
        #print("training......fold",count_fold)
        count_fold = count_fold + 1
        k_x_train = train_x.loc[train_index]
        k_y_train = train_y.loc[train_index]
        k_x_vali = train_x.loc[vali_index]
        k_y_vali = train_y.loc[vali_index]
        if model_type == 'lgb':
            dtrain = lgb.Dataset(k_x_train, k_y_train)
            dvalid = lgb.Dataset(k_x_vali, k_y_vali, reference=dtrain)
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            if 'sample_weight' in train.columns:
                lgb_model = lgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_vali, k_y_vali)],
                                          early_stopping_rounds=200, verbose=False, eval_metric="mae",
                                          sample_weight=train.loc[train_index]['sample_weight'],
                                          )
            else:
                lgb_model = lgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_vali, k_y_vali)],
                                      early_stopping_rounds=200, verbose=False, eval_metric="mae")
            k_pred = lgb_model.predict(k_x_vali, num_iteration=lgb_model.best_iteration_)
            pred = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration_)
        elif model_type == 'xgb':
            xgb_model = XGBRegressor(**xgb_params)
            xgb_model = xgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                                      early_stopping_rounds=200, verbose=False)
            k_pred = xgb_model.predict(k_x_vali)
            pred = xgb_model.predict(test_x)
        elif model_type == 'rf':
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, criterion="mae",n_jobs=-1,random_state=2019)
            model = rf_model.fit(k_x_train, k_y_train)
            k_pred = rf_model.predict(k_x_vali)
            pred = rf_model.predict(test_x)
        elif model_type == 'cat':
            ctb_params = {
                'n_estimators': 1000,
                'learning_rate': 0.02,
                'random_seed': 4590,
                'reg_lambda': 0.08,
                'subsample': 0.7,
                'bootstrap_type': 'Bernoulli',
                'boosting_type': 'Plain',
                'one_hot_max_size': 100,
                'rsm': 0.5,
                'leaf_estimation_iterations': 5,
                'use_best_model': True,
                'max_depth': 5,
                'verbose': -1,
                'thread_count': 4,
                'cat_features':category_features1+category_features2
            }

            cat_model = cat.CatBoostRegressor(**ctb_params)
            cat_model.fit(k_x_train, k_y_train, verbose=False, use_best_model=True, eval_set=[(k_x_vali, k_y_vali)])
            k_pred = cat_model.predict(k_x_vali)
            pred = cat_model.predict(test_x)
        preds_list.append(pred)
        oof[vali_index] = k_pred

    # if model_type == 'lgb':
    #     print(pd.DataFrame({
    #         'column': features,
    #         'importance': lgb_model.feature_importances_,
    #     }).sort_values(by='importance'))

    preds_columns = ['preds_{id}'.format(id=i) for i in range(n_fold)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds = list(preds_df.mean(axis=1))

    return preds, oof


def reg_model_v2(train, test, label_name, model_type, numerical_features, category_features1, category_features2, split_info, bset_iter):
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    combine = pd.concat([train, test], axis=0)
    combine = multi_column_LabelEncoder(combine, category_features1, rename=True)
    #combine[category_features1 + category_features2] = combine[category_features1 + category_features2].astype(
    #    'category')
    train = combine[:train.shape[0]]
    test = combine[train.shape[0]:]

    features = category_features1 + category_features2 + numerical_features

    split_name = split_info[0]
    train_ind = split_info[1]
    valid_ind = split_info[2]

    train_x = train[(train[split_name]>=train_ind[0]) & (train[split_name]<=train_ind[1])][features]
    train_y = train[(train[split_name]>=train_ind[0]) & (train[split_name]<=train_ind[1])][label_name]
    valid_x = train[(train[split_name] >= valid_ind[0]) & (train[split_name] <= valid_ind[1])][features]
    valid_y = train[(train[split_name] >= valid_ind[0]) & (train[split_name] <= valid_ind[1])][label_name]
    test_x = test[features]

    print(train_x.shape, valid_x.shape)
    model = get_model_type(train_x, train_y, valid_x, valid_y, model_type, category_features1+category_features2)
    valid_pre = model.predict(valid_x)

    valid_df = pd.DataFrame()
    valid_df['y'] = valid_y
    valid_df['y_hat'] = valid_pre
    valid_df['model'] = valid_x['model']
    valid_df['model_adcode_ym'] = train[(train[split_name] >= valid_ind[0]) & (train[split_name] <= valid_ind[1])]['model_adcode_ym']
    print('model.best_iteration_:',model.best_iteration_)
    model.n_estimators = model.best_iteration_ + 100
    #model.n_estimators = bset_iter + 100
    model.fit(train[features], train[label_name], categorical_feature=category_features1+category_features2)

    test_pre = model.predict(test_x)

    return model, valid_df, test_pre


def get_model_type(train_x,train_y,valid_x,valid_y,m_type='lgb', cat_features=None):
    if m_type == 'lgb':
        model = lgb.LGBMRegressor(
                                num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='mse',
                                max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=2019,
                                n_estimators=2000, subsample=0.9, colsample_bytree=0.7,
                                )
        model.fit(train_x, train_y,
              eval_set=[(train_x, train_y),(valid_x, valid_y)],
              early_stopping_rounds=100, verbose=False, categorical_feature=cat_features)
    elif m_type == 'xgb':
        model = xgb.XGBRegressor(
                                max_depth=5 , learning_rate=0.05, n_estimators=2000,
                                objective='reg:gamma', tree_method = 'hist',subsample=0.9,
                                colsample_bytree=0.7, min_child_samples=5,eval_metric = 'rmse'
                                )
        model.fit(train_x, train_y,
              eval_set=[(train_x, train_y),(valid_x, valid_y)],
              early_stopping_rounds=100, verbose=100)
    else:
        model = 1
    return model

def class_model(train, test, features_map, model_type='lgb', class_num=2, cv=True):
    label = features_map['label']
    category_features = features_map['category_features']
    numerical_features = features_map['numerical_features']
    combine = pd.concat([train, test], axis=0)
    #combine = multi_column_LabelEncoder(combine, category_features, rename=True)
    combine.reset_index(inplace=True)
    combine[category_features] = combine[category_features].astype('category')

    train_df = combine.loc[:train.shape[0]-1]
    test_df = combine.loc[train.shape[0]:]
    train_df[label] = train_df[label].astype(np.int)


    features = category_features + numerical_features
    train_y = train_df[[label]]
    train_x = train_df[features]
    test_x = test_df[features]



    #模型训练
    lgb_params = {
        'application': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'max_depth': -1,
        'num_leaves': 31,
        'verbosity': -1,
        'data_random_seed': 2019,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.6,
        'nthread': 4,
        'lambda_l1': 1,
        'lambda_l2': 5,
        'device':'cpu'
    }
    cat_model = cat.CatBoostClassifier(iterations=1000, depth=8, cat_features=features, learning_rate=0.05, custom_metric='F1',
                               eval_metric='F1', random_seed=2019,
                               l2_leaf_reg=5.0, logging_level='Silent')
    # clf = lgb.LGBMClassifier(
    #     objective='binary',
    #     learning_rate=0.02,
    #     n_estimators=1000,
    #     max_depth=-1,
    #     num_leaves=31,
    #     subsample=0.8,
    #     subsample_freq=1,
    #     colsample_bytree=0.8,
    #     random_state=2019,
    #     reg_alpha=1,
    #     reg_lambda=5,
    #     n_jobs=6
    # )
    cxgb = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=1000,
        subsample=0.8,
        random_state=2019,
        n_jobs=6
    )
    if cv:
        n_fold = 5
        print(train.shape[0])
        result = np.zeros((test.shape[0],))
        oof = np.zeros((train.shape[0],))
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=2019)
        kfold = skf.split(train_x, train_y)
        count_fold = 0
        for train_index, vali_index in kfold:
            print("training......fold",count_fold)
            count_fold = count_fold + 1
            k_x_train = train_x.loc[train_index]
            k_y_train = train_y.loc[train_index]
            k_x_vali = train_x.loc[vali_index]
            k_y_vali = train_y.loc[vali_index]

            if model_type == 'lgb':
                trn = lgb.Dataset(k_x_train, k_y_train)
                val = lgb.Dataset(k_x_vali, k_y_vali)
                lgb_model = lgb.train(lgb_params, train_set=trn, valid_sets=[trn, val],
                              num_boost_round=5000,early_stopping_rounds=200, verbose_eval=False)
                test_pred_proba = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration)
                val_pred_proba = lgb_model.predict(k_x_vali, num_iteration=lgb_model.best_iteration)
                # clf.fit(k_x_train, k_y_train,eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],early_stopping_rounds=200, verbose=False)
                # test_pred_proba = clf.predict_proba(test_x, num_iteration=clf.best_iteration_)
                # val_pred_proba = clf.predict_proba(k_x_vali, num_iteration=clf.best_iteration_)
            elif model_type == 'xgb':
                cxgb.fit(k_x_train, k_y_train,eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],early_stopping_rounds=200, verbose=False)
                test_pred_proba = cxgb.predict_proba(test_x)
                val_pred_proba = cxgb.predict_proba(k_x_vali)
            elif model_type == 'cat':
                cat_model.fit(k_x_train,k_y_train)
                test_pred_proba = cat_model.predict(test_x)
                val_pred_proba = cat_model.predict(k_x_vali)
            result = result + test_pred_proba
            oof[vali_index] = val_pred_proba
        result = result/n_fold
    else:

        print(train_x.shape, train_y.shape)
        if model_type == 'cat':
            cat_model.fit(train[features], train[label])
            test_pred_proba = cat_model.predict(test[features])
            train_pred_proba = cat_model.predict(train[features])
        else:
            lgb_df = lgb.Dataset(train_x, train_y)
            lgb_model = lgb.train(lgb_params, train_set=lgb_df, categorical_feature=category_features,
                                  num_boost_round=1500,)

            test_pred_proba = lgb_model.predict(test_x)
            train_pred_proba = lgb_model.predict(train_x)
            feat_imp = lgb_model.feature_importance(importance_type='gain')
            feat_nam = lgb_model.feature_name()
            for fn, fi in zip(feat_nam, feat_imp):
                print(fn,fi)
        # clf.fit(train_x, train_y, categorical_feature=category_features)
        # #test_pred = clf.predict(test_x)
        # test_pred_proba = clf.predict_proba(test_x, num_iteration=clf.best_iteration_)
        # train_pred_proba = clf.predict_proba(train_x, num_iteration=clf.best_iteration_)


        result = test_pred_proba
        oof = train_pred_proba

    return oof,result


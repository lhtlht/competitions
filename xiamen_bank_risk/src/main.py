from features import *
from model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,mean_absolute_error,mean_squared_error,roc_auc_score


if __name__ == "__main__":
    train_data = pd.read_pickle(P_DATA + 'train.pk') #132029 104
    train_target = train_data['target']
    del train_data['target']
    test_data = pd.read_pickle(P_DATA + 'test.pk') #23561


    offline = False
    if offline:
        train, test, train_y, test_y = train_test_split(train_data,train_target,test_size=0.2, random_state=0) # 105623 26406
    else:
        train = train_data
        train_y = train_target
        test = test_data
    train = pd.concat([train, train_y], axis=1)
    #特征处理
    train, test = feature_labelRate(train, test, 'target', 'certId') # gender_labelRate
    train, test = feature_labelRate(train, test, 'target', 'bankCard')  # gender_labelRate


    for i in train_data.columns:
        print(i)

    x_features = ['x_0', 'x_12','x_14','x_16','x_20','x_25','x_26','x_27','x_28','x_29','x_30','x_31','x_32','x_33','x_34',
                                               'x_35', 'x_39', 'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_50',
                                               'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66',
                                               'x_67', 'x_68', 'x_69', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76',]
    #np.linalg.svd(train[x_features].values)
    credit_features = ['ncloseCreditCard','unpayIndvLoan','unpayOtherLoan','unpayNormalLoan','5yearBadloan']
    #dist 衍生特征 测试没好结果
    dist_features = ['delivery','city','post','province']

    user_features = ['edu', 'job', 'highestEdu', 'gender', 'is_han',] + ['certId_labelRate', 'bankCard_labelRate']  #用户基本属性
    load_features = ['loanProduct', 'setupHour', 'weekday', 'linkRela', 'basicLevel']
    features_map = {
        'label':'target',
        'numerical_features':['age', 'lmt'] + x_features,
        'category_features': user_features + load_features + credit_features

    }
    oof,result = class_model(train, test, features_map, )

    if offline:
        train_auc = roc_auc_score(train_y, oof)
        test_auc = roc_auc_score(test_y, result)
        print(train_auc, test_auc)
    else:
        train_auc = roc_auc_score(train_y, oof)
        print(train_auc)

        test['target'] = result
        test[['id','target']].to_csv(P_SUBMIT + 'submit_20191013_2.csv', encoding='utf-8', index=False)
'''
0.7087464926185002 0.7185855388604988  --76703

0.708560414149213 0.7263235812767077
0.7096689738896833 0.7306906818237742



'''
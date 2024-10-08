import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt



df_member = pd.read_csv("D:\Kkbox\kkbox_churn_prediction_challenge\members_v3.csv\members_v3.csv")
df_train = pd.read_csv("D:\Kkbox\kkbox_churn_prediction_challenge\\train.csv\\train.csv")
df_train2 = pd.read_csv("D:\Kkbox\kkbox_churn_prediction_challenge\\train_v2.csv\data\churn_comp_refresh\\train_v2.csv")
df_test = pd.read_csv("D:\Kkbox\kkbox_churn_prediction_challenge\sample_submission_v2.csv\data\churn_comp_refresh\sample_submission_v2.csv")
#df_user = pd.read_csv("D:\Kkbox\kkbox_churn_prediction_challenge\user_logs\user_logs.csv")
#df_user2 = pd.read_csv("D:\Kkbox\kkbox_churn_prediction_challenge\user_logs_v2.csv\data\churn_comp_refresh\user_logs_v2.csv")
df_transaction = pd.read_csv("D:\Kkbox\kkbox_churn_prediction_challenge\\transactions.csv\\transactions.csv")
df_transaction2 = pd.read_csv("D:\Kkbox\kkbox_churn_prediction_challenge\\transactions_v2.csv\data\churn_comp_refresh\\transactions_v2.csv")

#l_m = len(df_member['bd'])
#l_train = len(df_train)
#l_train2 = len(df_train2)
#l_test = len(df_test)
#判斷'bd'能不能用

#處理 member 資料型態
'''
df_member = df_member.dropna(subset=['city', 'registered_via'])

df_member = df_member.dropna(subset=['registration_init_time'])
df_member['registration_init_time'] = df_member['registration_init_time'].astype(int).astype(str)
df_member['registration_init_time'] = pd.to_datetime(df_member['registration_init_time'], format='%Y%m%d', errors='coerce')

train_end_date = pd.to_datetime('2017-03-31')

df_member['date'] = (train_end_date - df_member['registration_init_time']).dt.days
dl_member = df_member.drop(columns = ['bd', 'gender','registration_init_time'])

#print(dl_member.dtypes)
'''
'''
merged_df = df_train2.merge(df_member, on='msno', how='left')
valid_ages = merged_df[(merged_df['bd'] > 12) & (merged_df['bd'] < 85)]
total = len(valid_ages)
result = total / l_train2

print(result)
'''

#整理資料型態

###train
merged_train_df = df_train.merge(df_member, on='msno', how='left')
merged_train_df = pd.DataFrame(merged_train_df)

merged_train_df = merged_train_df.dropna(subset=['city', 'registered_via'])

merged_train_df = merged_train_df.dropna(subset=['registration_init_time'])
merged_train_df['registration_init_time'] = merged_train_df['registration_init_time'].astype(int).astype(str)
merged_train_df['registration_init_time'] = pd.to_datetime(merged_train_df['registration_init_time'], format='%Y%m%d', errors='coerce')

train_end_date = pd.to_datetime('2017-03-31')

merged_train_df['date'] = (train_end_date - merged_train_df['registration_init_time']).dt.days
merged_train_df = merged_train_df.drop(columns = ['bd', 'gender','registration_init_time'])

###train2
merged_train2_df = df_train2.merge(df_member, on='msno', how='left')
merged_train2_df = pd.DataFrame(merged_train2_df)

merged_train2_df = merged_train2_df.dropna(subset=['city', 'registered_via'])

merged_train2_df = merged_train2_df.dropna(subset=['registration_init_time'])
merged_train2_df['registration_init_time'] = merged_train2_df['registration_init_time'].astype(int).astype(str)
merged_train2_df['registration_init_time'] = pd.to_datetime(merged_train2_df['registration_init_time'], format='%Y%m%d', errors='coerce')

train2_end_date = pd.to_datetime('2017-11-13')
merged_train2_df['date'] = (train2_end_date - merged_train2_df['registration_init_time']).dt.days

merged_train2_df = merged_train2_df.drop(columns = ['bd', 'gender', 'registration_init_time'])


#合併train, train2
merged_trained_df = pd.concat([merged_train_df, merged_train2_df], ignore_index=True)


###合併transaction, transaction2
merged_transactioned_df = pd.concat([df_transaction, df_transaction2], ignore_index=True)
###transaction
#print(merged_transactioned_df['payment_method_id'].unique())   #41
merged_transactioned_df['transaction_date'] = merged_transactioned_df['transaction_date'].astype(int).astype(str)
merged_transactioned_df['transaction_date'] = pd.to_datetime(merged_transactioned_df['transaction_date'], format='%Y%m%d', errors='coerce')
merged_transactioned_df['membership_expire_date'] = merged_transactioned_df['membership_expire_date'].astype(int).astype(str)
merged_transactioned_df['membership_expire_date'] = pd.to_datetime(merged_transactioned_df['membership_expire_date'], format='%Y%m%d', errors='coerce')
merged_transactioned_df['plan_date'] = (merged_transactioned_df['membership_expire_date'] - merged_transactioned_df['transaction_date']).dt.days
merged_transactioned_df = merged_transactioned_df.drop(columns = ['transaction_date', 'membership_expire_date'])
#print(merged_transactioned_df.head())

merged_trained_df = merged_trained_df.merge(merged_transactioned_df, on='msno', how='left')
merged_trained_df = pd.DataFrame(merged_trained_df)

'''
###user
merged_user_df = df_user.merge(dl_member, on='msno', how='left')
merged_user_df = pd.DataFrame(merged_user_df)

###user2
merged_user2_df = df_user2.merge(dl_member, on='msno', how='left')
merged_user2_df = pd.DataFrame(merged_user2_df)
'''

###test

merged_test_df = df_test.merge(df_member, on='msno', how='left')
merged_test_df = pd.DataFrame(merged_test_df)

merged_test_df = merged_test_df.dropna(subset=['city', 'registered_via'])

merged_test_df = merged_test_df.dropna(subset=['registration_init_time'])
merged_test_df['registration_init_time'] = merged_test_df['registration_init_time'].astype(int).astype(str)
merged_test_df['registration_init_time'] = pd.to_datetime(merged_test_df['registration_init_time'], format='%Y%m%d', errors='coerce')

test_end_date = pd.to_datetime('2017-03-31')

merged_test_df['date'] = (train_end_date - merged_test_df['registration_init_time']).dt.days
merged_test_df = merged_test_df.drop(columns = ['bd', 'gender','registration_init_time'])
merged_test_df= merged_test_df.merge(merged_transactioned_df, on='msno', how='left')
merged_test_df = pd.DataFrame(merged_test_df)
#print(merged_test_df.isna().sum())
#print(merged_test_df.head())

#正規化date
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(merged_trained_df[['date']])
merged_trained_df['date_scaled'] = X_scaled
merged_trained_df = merged_trained_df.drop(columns = ['date'])
'''
'''
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
print("before lable:\n",merged_trained_df['city'])
merged_trained_df['city'] = labelencoder.fit_transform(merged_trained_df['city'])
merged_trained_df['registered_via'] = labelencoder.fit_transform(merged_trained_df['registered_via'])

print("after lable:\n",merged_trained_df['city'])
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(sparse_output=False)
one_hot_encoded =onehotencoder.fit_transform(merged_trained_df[['city']])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=onehotencoder.categories_[0])
merged_trained_df = merged_trained_df.drop(columns=['city'])
merged_trained_df = merged_trained_df.join(one_hot_df)
print("after ohn:\n",merged_trained_df)
print(merged_trained_df.shape)

print(merged_trained_df.dtypes)
'''
#切訓練集

from sklearn.model_selection import train_test_split
y = merged_trained_df["is_churn"].to_numpy()
X = merged_trained_df.drop(columns=["is_churn","msno"]).to_numpy()

print(len(X))
# 获取特征名称
feature_names = merged_trained_df.drop(columns=["is_churn", "msno"]).columns

# 切分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=61)

###################
###Random Forest###
###################
'''
forest = RandomForestClassifier(n_estimators=5,criterion="log_loss",n_jobs=-1)
forest.fit(X_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
#plt.title("feature")
#plt.bar(range(X_train.shape[1]),
#        importances[indices],
#        align="center")
#plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=0)
#plt.show()

# 验证模型在验证集上的表现
val_score = forest.score(X_val, y_val)
print(f"Validation Accuracy: {val_score:.2f}")
#Randomforst

#Predict
# 假设 X_new 是你想要进行预测的新数据集
y_predict = merged_test_df["is_churn"].to_numpy()
X_predict = merged_test_df.drop(columns=["msno", "is_churn"]).to_numpy()


# 使用训练好的模型进行预测
predictions = forest.predict(X_predict)

# 如果你想查看概率预测，可以使用 predict_proba
probabilities = forest.predict_proba(X_predict)

# 输出预测结果
print("Predicted labels:", predictions)
print("Predicted probabilities:", probabilities)
'''

############
##XGBoost###
############


from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
# xgbmodel = XGBClassifier(eta = 0.2,
#                          max_depth=5,
#                          n_estimators=100,
#                          use_label_encoder=False,
#                          eval_metric='logloss')
# xgbmodel.fit(X_train, y_train)

# val_score_xg = xgbmodel.score(X_val, y_val)
# print(f"Validation Accuracy of XGB: {val_score_xg:.2f}")
# y_pred_xgb = merged_test_df["is_churn"].to_numpy()
# X_pred_xgb = merged_test_df.drop(columns=["msno", "is_churn"]).to_numpy()

# predictions_xgb = xgbmodel.predict(X_pred_xgb)

# probabilities_xgb = xgbmodel.predict_proba(X_pred_xgb)

# print("Predicted labels of XGB:", predictions_xgb)
# print("Predicted probabilities of XGB:", probabilities_xgb)

# y_pred_proba = xgbmodel.predict_proba(X_val)
# loss = log_loss(y_val, y_pred_proba)
# print(f'Log Loss: {loss:.4f}')
# accuracy = accuracy_score(y_val, predictions_xgb)
# print(f'Accuracy: {accuracy:.4f}')


'''
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2], #0.2
    'max_depth': [3, 5, 7, 9], #9
    'n_estimators': [100, 200, 300, 400] #400
}

grid_search = GridSearchCV(estimator=XGBClassifier(use_label_encoder=False),
                           param_grid=param_grid,
                           scoring='neg_log_loss',
                           cv=5,
                           verbose=1)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best log loss: {grid_search.best_score_}")
'''
xgbmodel = XGBClassifier(eta = 0.2,
                          max_depth=9,
                          n_estimators=400,
                          use_label_encoder=False,
                          eval_metric='logloss')
xgbmodel.fit(X_train, y_train)

val_score_xg = xgbmodel.score(X_val, y_val)
print(f"Validation Accuracy of XGB: {val_score_xg:.2f}")
y_pred_xgb = merged_test_df["is_churn"].to_numpy()
X_pred_xgb = merged_test_df.drop(columns=["msno", "is_churn"]).to_numpy()

predictions_xgb = xgbmodel.predict(X_pred_xgb)

probabilities_xgb = xgbmodel.predict_proba(X_pred_xgb)
print("Predicted labels of XGB:", predictions_xgb)
print("Predicted probabilities of XGB:", probabilities_xgb)

y_pred_proba = xgbmodel.predict_proba(X_val)
loss = log_loss(y_val, y_pred_proba)
print(f'Log Loss: {loss:.4f}')
accuracy = accuracy_score(y_val, predictions_xgb)
print(f'Accuracy: {accuracy:.4f}')

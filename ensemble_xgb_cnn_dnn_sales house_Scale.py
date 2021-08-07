import sys
sys.path
sys.path.append('c:\\users\lab535\\.conda\envs\\machinelearning\\lib\\site-packages')
import pandas as pd
import numpy as np 
import os
from scipy import stats
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
os.getcwd()
house_df_train=pd.read_csv('ntut-ml-2020-regression/ntut-ml-regression-2020/train-v3.csv')
house_df_valid=pd.read_csv('ntut-ml-2020-regression/ntut-ml-regression-2020/valid-v3.csv')
combine=pd.DataFrame()
combine=house_df_train.append(house_df_valid)
combine.reset_index(drop=True,inplace=True)
final_combine=combine
def mae_value(y_true, y_pred):
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred))/n
    return mae

house_feature_list=list(house_df_train)
house_feature_dict_1={
#'id':'id',
# 'price':'price',
# 'sale_yr':'year',
# 'sale_month':'月份',
# 'sale_day':'日期',
 'bedrooms':'房間',
'bathrooms':'Number of Bedrooms/House',
'sqft_living':'square footage of the home',
'sqft_lot':'square footage of the lot',
'floors':'Total floors (levels) in house'}
house_feature_dict_2={
# 'price':'price',
'waterfront':'House which has a view to a waterfront',
'view':'Has been viewed',
 'condition':'How good the condition is ( Overall )',
'grade':'overall grade given to the housing unit',
'sqft_above':'square footage of house apart from basement',  
'sqft_basement':'square footage of the basement',
'yr_built':'Built Year',
}
house_feature_dict_3={
# 'price':'price',
'yr_renovated':'Year when house was renovated',
#  'zipcode':'zip',
'lat':'Latitude coordinate',
'long':'Longitude coordinate',
'sqft_living15':'Living room area',
# 'sqft_lot15':'lotSize area'
 }

#數值篩選
house_feature_list=list(combine)
# #刪除離群值
# # mask=final_combine['bedrooms']>11
# # final_combine[mask]
# final_combine.drop(final_combine.loc[final_combine['bedrooms']>11].index, inplace=True)
# final_combine.loc[final_combine['bedrooms']>10].index
# # mask=combine['sqft_lot15']>600000
# # combine[mask]

final_combine.drop(final_combine.loc[final_combine['sqft_lot15']>800000].index, inplace=True)
# final_combine['sqft_living'].describe()
# # mask=final_combine['sqft_living']>10000

# final_combine.drop(final_combine.loc[final_combine['sqft_above']>10000].index, inplace=True)
# final_combine.drop(final_combine.loc[final_combine['sqft_above']>8000].index, inplace=True)
final_combine.reset_index(drop=True,inplace=True)

final_combine_label=final_combine['price']
del final_combine['id']
del final_combine['price']
scaler = preprocessing.StandardScaler().fit(final_combine)
train_scaler = scaler.transform(final_combine)

#DNN
# 在 Keras 裡面我們可以很簡單的使用 Sequential 的方法建建立一個 Model
model = Sequential()
# 加入 hidden layer-1 of 512 neurons 並指定 input_dim 為 784
model.add(Dense(80, input_dim=len(list(final_combine)), kernel_initializer='normal', activation='relu'))
# 加入 hidden layer-2 of 256 neurons
model.add(Dense(100, activation='relu', kernel_initializer='normal'))
model.add(Dense(200, activation='relu', kernel_initializer='normal'))
model.add(Dense(400, activation='relu', kernel_initializer='normal'))
model.add(Dense(200, activation='relu', kernel_initializer='normal'))
# 加入 hidden layer-3 of 128 neurons
model.add(Dense(100, activation='relu', kernel_initializer='normal'))
# 使用 'relu' 當作 activation function
# model.add(Dense(len(list(combine)*16),activation='relu', kernel_initializer='normal'))
# 使用 'relu' 當作 activation function
model.add(Dense(units=1,activation='relu', kernel_initializer='normal'))
# 定義訓練方式  
model.compile(loss='MAE', optimizer='adam')
model.summary()

# 開始訓練  
train_history = model.fit(x=train_scaler,  
                          y=final_combine_label, validation_split=0.1,  
                          epochs=200, batch_size=32, verbose=1)
#XGB
def grid_search():
    xgr=xgb.XGBRegressor(max_depth=8,min_child_weight=7, gamma=0)
    xgr.fit(train_scaler,final_combine_label)

#     parameters=[{'max_depth':[8,9,10,11,12],'min_child_weight':[4,5,6,7,8]}]
#     parameters=[{'gamma':[i/10.0 for i in range(0,5)]}]
#     parameters=[{'subsample':[i/10.0 for i in range(6,11)],'colsample_bytree':[i/10.0 for i in range(6,11)]}]
    parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:squarederror'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}
    grid_search= GridSearchCV(estimator=xgr, param_grid=parameters, cv=5,n_jobs=-1)

    print (1)
    grid_search=grid_search.fit(train_scaler,final_combine_label)
    print (2)
    best_accuracy=grid_search.best_score_
    best_parameters=grid_search.best_params_
    print (best_accuracy)
    print (best_parameters)
grid_search()

# xgr=xgb.XGBRegressor(max_depth=8,min_child_weight=7, gamma=0, colsample_bytree=1.0,subsample=0.7)
xgr=xgb.XGBRegressor(colsample_bytree=0.7, learning_rate=0.05, max_depth=6, min_child_weight=4, n_estimators=500, nthread=4, silent= 1, subsample=0.7)
xgr.fit(train_scaler,final_combine_label)

#XGB資料驗證
ahouse_df_test=pd.read_csv('ntut-ml-2020-regression/ntut-ml-regression-2020/test-v3.csv')
del ahouse_df_test['id']
# df7
scaler_test=scaler.transform(ahouse_df_test)
xgb_predictions=xgr.predict(scaler_test)
xgb_predictions=xgb_predictions.tolist()
xgb_prediction=[]
for i,k in enumerate(xgb_predictions):
    xgb_prediction.append([])
    xgb_prediction[i].append(k)   
xgb_prediction

for i in range(len(xgb_prediction)):
    xgb_prediction[i].insert(0,i+1)

final_df_xgb = pd.DataFrame(xgb_prediction,columns=['id','price'])
final_df_xgb.to_csv('xgb_output.csv',index=False)

#DNN test資料驗證 
dnn_predictions_array = model.predict(scaler_test)
dnn_predictions=dnn_predictions_array.tolist()
dnn_predictions

for i in range(len(dnn_predictions)):
    dnn_predictions[i].insert(0,i+1)

final_df = pd.DataFrame(dnn_predictions,columns=['id','price'])
final_df.to_csv('dnn_output_3.csv',index=False)

#CNN
train_reshape = train_scaler.reshape(-1, 1,len(list(final_combine)),1)
train_reshape.shape[1:]
vaild_reshape = scaler_test.reshape(-1, 1,len(list(final_combine)),1)
vaild_reshape.shape[1:]


# create model
model_cnn = Sequential()
# Adds a densely-connected layer with 64 units to the model:
model_cnn.add(Conv2D(64,(1,2), activation = 'relu', input_shape = train_reshape.shape[1:]))
model_cnn.add(MaxPooling2D(pool_size = (1,2)))
# Add another:
model_cnn.add(Conv2D(64,(1,2), activation = 'relu'))
model_cnn.add(Conv2D(64,(1,2), activation = 'relu'))
model_cnn.add(Conv2D(64,(1,2), activation = 'relu'))
model_cnn.add(Flatten())
model_cnn.add(Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model_cnn.add(Dense(1, activation='relu'))
model_cnn.compile(optimizer="adam",loss='MAE')


model_cnn.fit(train_reshape, final_combine_label, epochs=200, batch_size=32,verbose=1 )

display(model_cnn.summary())

# # 使用CNN方式
predictions = model_cnn.predict(vaild_reshape)
# predicted_val_cnn = [int(round(p[0])) for p in predictions]
# mae_value(df_valid_label,predicted_val_cnn)

# # 使用CNN方式
cnn_predictions = model_cnn.predict(vaild_reshape)

#使用集成學習
sample = np.array([dnn_predictions_array,cnn_predictions,xgb_prediction])
res = sample.mean(axis=0)
# mae_value(df_valid_label,list(res))
res
res=res.tolist()
res

for i in range(len(res)):
    res[i].insert(0,i+1)
final_df_res = pd.DataFrame(res,columns=['id','price'])
final_df_res.to_csv('emsemble_output_3.csv',index=False)

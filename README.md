# House Sales regression

## Step 1 使用環境

```python=
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

```

## Step 2 特徵值挑選(2-1特徵挑選 /2-2 不進行特徵挑選)

```python=
house_df_train=pd.read_csv('ntut-ml-2020-regression/ntut-ml-regression-2020/train-v3.csv')
house_df_valid=pd.read_csv('ntut-ml-2020-regression/ntut-ml-regression-2020/valid-v3.csv')
#合併valid和train
combine=pd.DataFrame()
combine=house_df_train.append(house_df_valid)
combine.reset_index(drop=True,inplace=True)
final_combine=combine
#特徵值list
house_feature_list=list(house_df_train)

```
### step 2-1 透過相關性進行特徵挑選
``` python 
house_feature_dict_1={
#'id':'id',
'price':'price',
'sale_yr':'year',
'sale_month':'月份',
'sale_day':'日期',
 'bedrooms':'房間',
'bathrooms':'Number of Bedrooms/House',
'sqft_living':'square footage of the home',
'sqft_lot':'square footage of the lot',
'floors':'Total floors (levels) in house'}
house_feature_dict_2={
'price':'price',
'waterfront':'House which has a view to a waterfront',
'view':'Has been viewed',
 'condition':'How good the condition is ( Overall )',
'grade':'overall grade given to the housing unit',
'sqft_above':'square footage of house apart from basement',  
'sqft_basement':'square footage of the basement',
'yr_built':'Built Year',
}
 house_feature_dict_3={
'price':'price',
'yr_renovated':'Year when house was renovated',
 'zipcode':'zip',
'lat':'Latitude coordinate',
'long':'Longitude coordinate',
'sqft_living15':'Living room area',
'sqft_lot15':'lotSize area'
 }
plot_want1 = list(house_feature_dict_1.keys())
plot_want2= list(house_feature_dict_2.keys())
plot_want3= list(house_feature_dict_3.keys())
display(sns.pairplot(house_df_train[plot_want1]))
display(sns.pairplot(house_df_train[plot_want2]))
display(sns.pairplot(house_df_train[plot_want3]))

```
![](https://i.imgur.com/feB00Vs.png)
![](https://i.imgur.com/xuxm6Hg.png)
![](https://i.imgur.com/wCETIUS.png)
### 最後挑選之特徵值 
##### 主要是挑選與price呈正相關的
##### 部分特徵值依常理而留存像是:建構年份後續進行特徵值的轉換
```
    'bedrooms':'bedroom',
    'bathrooms':'Number of Bedrooms/House',
    'sqft_living':'square footage of the home'
    'grade':'overall grade given to the housing unit',
    'sqft_above':'square footage of house apart from basement',  
    'sqft_basement':'square footage of the basement',
    'yr_built':'Built Year',
    'yr_renovated':'Year when house was renovated'
    'lat':'Latitude coordinate',
    'long':'Longitude coordinate',
    'sqft_living15':'Living room area'
    'zipcode':'zip'
```
## Step 3-1 特徵值數值轉換
###### one-hot encoding :zip、 bedrooms、bathrooms
###### 年份轉換:yr_built、yr_renovated
```python=
    
    df2=pd.get_dummies(house_df_train['zipcode'])
    df2=pd.get_dummies(house_df_train['bedrooms'])
    df2=pd.get_dummies(house_df_train['bathrooms'])
    house_df_train['yr_built']=2020-house_df_train['yr_built']
    mask=house_df_train['yr_renovated']>0
    house_df_train[mask]=2020-house_df_train[mask]

```

## Step 4-1 模型訓練
``` python 
X_df=house_df_train[final_feature]
valid_df=house_df_valid[final_feature]
combine=pd.concat([X_df,valid_df])
combine.reset_index(drop=True,inplace=True)
#標準化
scale = StandardScaler() #z-scaler物件
scale.fit(combine)
scaler_combine=scale.transform(combine)

#  Sequential 的方法建建立一個 Model
model = Sequential()
# 加入 hidden layer-1 of 80
model.add(Dense(80, input_dim=len(list(combine)), kernel_initializer='normal', activation='relu'))
# 加入 hidden layer-2 of 100 neurons
model.add(Dense(100, activation='relu', kernel_initializer='normal'))
# 加入 hidden layer-3 of 200 neurons
model.add(Dense(200, activation='relu', kernel_initializer='normal'))
# 加入 hidden layer-3 of 400 neurons
model.add(Dense(400, activation='relu', kernel_initializer='normal'))
# 加入 hidden layer-3 of 200 neurons
model.add(Dense(200, activation='relu', kernel_initializer='normal'))
# 加入 hidden layer-3 of 100 neurons
model.add(Dense(100, activation='relu', kernel_initializer='normal'))

# model.add(Dense(len(list(combine)*16),activation='relu', kernel_initializer='normal'))
# 使用 'relu' 當作 activation function
model.add(Dense(units=1,activation='relu', kernel_initializer='normal'))
# 定義訓練方式  
model.compile(loss='MAE', optimizer='adam')
model.summary()

```
##### 層數資訊

![](https://i.imgur.com/kzDdqnq.png)


#### 模型收斂狀況
![](https://i.imgur.com/ROTvTH0.png)

## Step 5-1 模型驗證
``` python
ahouse_df_test=pd.read_csv('ntut-ml-2020-regression/ntut-ml-regression-2020/test-v3.csv')


#數值前處理
df2=pd.get_dummies(ahouse_df_test['zipcode'])
df2=pd.get_dummies(ahouse_df_test['bedrooms'])
df2=pd.get_dummies(ahouse_df_test['bathrooms'])
ahouse_df_test['yr_built']=2020-ahouse_df_test['yr_built']
mask=ahouse_df_test['yr_renovated']>0
ahouse_df_test[mask]=2020-ahouse_df_test[mask]

test_df=ahouse_df_test[final_feature]


combine_list=list(combine)
combine_list

test_df=test_df[combine_list]


# 正規化數值轉換
scaler_test=scale.transform(test_df)
predictions = model.predict(scaler_test)
predictions=predictions.tolist()
predictions

#輸出預測值
for i in range(len(predictions)):
    predictions[i].insert(0,i+1)

final_df = pd.DataFrame(predictions,columns=['id','price'])
final_df.to_csv('fianlcsv',index=False)
```
#### 第一版挑選特徵值只進行DNN模型之訓練成績如下
![](https://i.imgur.com/wmHG6VW.png)



### step 2-2 不進行特徵挑選只處理outlier
##### 進行很多離群值淘汰 最後發現移除sqft_lot15大於80萬模型成效最好
```python=
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

```

### Step3-2 模型訓練
* DNN
* XGBOOST
* CNN
* 三種模型進行集成訓練

#### DNN
``` python=
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

```
#### DNN模型結果有點overfitting
![](https://i.imgur.com/RjfXNom.png)

#### DNN test資料預測
```python=
dnn_predictions_array = model.predict(scaler_test)
dnn_predictions=dnn_predictions_array.tolist()
dnn_predictions

```


### XGBOOST
#### 使用xgboost進行訓練，也進行模型的調參過程
```python=
def grid_search():
    xgr=xgb.XGBRegressor(max_depth=8,min_child_weight=5, gamma=0)
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
    grid_search= GridSearchCV(estimator=xgr, param_grid=parameters, cv=5,n_jobs=-1,scoring='neg_mean_absolute_error')
     
    grid_search=grid_search.fit(train_scaler,final_combine_label)
    best_accuracy=grid_search.best_score_
    best_parameters=grid_search.best_params_
    print (abs(best_accuracy))
    print (best_parameters)
    return(best_parameters)
best_parameters=grid_search()
```
##### 最佳模型參數輸出 MAE輸出為63764
![](https://i.imgur.com/TXIu4kO.png)

##### XGBOOST test資料預測
```python=
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
```


#### CNN 模型訓練
##### 卷積層 >池化層>卷積層>卷積層>卷積層>Flatten>DNN分類器
```python=
train_reshape = train_scaler.reshape(-1, 1,len(list(final_combine)),1)
train_reshape.shape[1:]
vaild_reshape = scaler_test.reshape(-1, 1,len(list(final_combine)),1)
vaild_reshape.shape[1:]
%%time
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
### prediction
cnn_predictions = model_cnn.predict(vaild_reshape)

```


#### 最後進行集成學習
###### 將三個預測結果取平均作為最後輸出
```python=
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
```

##### 最後分數輸出
![](https://i.imgur.com/ooAW3yF.png)


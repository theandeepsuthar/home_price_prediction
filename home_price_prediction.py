#!/usr/bin/env python
# coding: utf-8

# In[716]:


import numpy as np
import pandas as pd 


# In[717]:


data=pd.read_csv("C:\computer science engineering\ML\Bengaluru_House_Data.csv")


# In[718]:


data.head()


# In[719]:


data.shape


# In[720]:


data.info()


# In[721]:


for column in data.columns:
    print(data[column].value_counts())
    print('*'*20)
    


# In[722]:


data.isnull().sum()


# In[723]:


data.columns


# In[724]:


data.drop(columns=['area_type','availability','society','balcony'],inplace=True)


# In[725]:


data.shape


# In[ ]:





# In[726]:


data.describe()


# In[727]:


data.info()


# In[728]:


data['location'].value_counts()


# In[729]:


data['location']=data['location'].fillna('Sarjapur Road')


# In[730]:


data['size'].value_counts()


# In[731]:


data['size']=data['size'].fillna('2 BHK')


# In[732]:


data['bath']=data['bath'].fillna(data['bath'].median())


# In[733]:


data.info


# In[734]:


data.info


# In[735]:


data['bhk']=data['size'].str.split().str.get(0).astype(int)


# In[736]:


data[data.bhk>20]


# In[737]:


data['total_sqft'].unique()


# In[738]:


def convertrange(x):
    if isinstance(x, float):  # Check if x is a float
        x = str(x)  # Convert float to string

    temp = x.split('-')
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1])) / 2
    try:
        return float(x)
    except:
        return None


# In[739]:


data['total_sqft']=data['total_sqft'].apply(convertrange)


# In[740]:


data.head()


# price per  square feet
# 

# In[741]:


data['price_per_sqft']=data['price']*100000/data['total_sqft']


# In[742]:


data['price_per_sqft']


# In[743]:


data.describe()


# In[ ]:





# In[744]:


data['location'].value_counts()


# In[745]:


data['location']=data['location'].apply(lambda x: x.strip())


# In[746]:


location_count=data['location'].value_counts()


# In[747]:


location_count_less_10=location_count[location_count<=10]


# In[748]:


location_count_less_10


# In[749]:


data['location']=data['location'].apply(lambda x:'other' if x in location_count_less_10   else x)


# In[750]:


data['location'].value_counts()


# outlier detection and removal

# In[751]:


data.describe()


# In[752]:


(data['total_sqft']/data['bhk']).describe()


# In[753]:


data=data[((data['total_sqft']/data['bhk'])>=300)]
data.describe()


# In[754]:


data1.shape


# In[755]:


data.price_per_sqft.describe()


# In[756]:


def remove_outliers_sqft(df):
    df_output=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        
        gen_df=subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft <= (m+st))]
        
        df_output=pd.concat([df_output,gen_df],ignore_index=True)
    return df_output
data=remove_outliers_sqft(data)
data.describe()


# In[757]:


def bhk_outliers_remove(df):
    exclude_indices=np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices =np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
                


# In[758]:


data=bhk_outliers_remove(data)


# In[759]:


data.shape


# In[760]:


data


# In[761]:


data.drop(columns=['size','price_per_sqft'],inplace=True)


# In[762]:


data


# clened data 
# 

# In[763]:


data.head()


# In[764]:


data.to_csv("cleaned_data.csv")


# In[765]:


x=data.drop(columns=['price'])
y=data['price']


# In[766]:


x


# In[767]:


y


# In[768]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[769]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[770]:


categorical_columns = [0]  # Replace with the actual indices of categorical columns
numeric_columns = [1, 2, 3]  # Replace with the actual indices of numeric columns


# In[771]:


print(x_train.shape)


# 

# In[772]:


print(x_test.shape)


# Applying linear regression

# In[773]:


column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_columns),
    (StandardScaler(with_mean=False), numeric_columns),
    remainder="passthrough"
)


# In[ ]:





# In[774]:


lr=LinearRegression()


# In[775]:


pipe=make_pipeline(column_trans,lr)


# In[776]:


pipe


# In[777]:


pipe.fit(x_train,y_train)


# In[ ]:





# In[778]:


y_pred = pipe.predict(x_test)


# In[779]:


r2_score(y_test,y_pred)


# Applying lasso
# 

# In[784]:


column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_columns),
    (StandardScaler(with_mean=False), numeric_columns),
    remainder="passthrough"
)


# In[785]:


scaler = StandardScaler(with_mean=False) 


# In[786]:


lasso=Lasso()


# In[787]:


pipe=make_pipeline(column_trans,scaler,lasso)


# In[788]:


pipe.fit(x_train,y_train)


# In[797]:


y_pred_lasso=pipe.predict(x_test)


# In[798]:


r2_score(y_test,y_pred_lasso)


# applying ridge

# In[791]:


column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_columns),
    (StandardScaler(with_mean=False), numeric_columns),
    remainder="passthrough"
)


# In[792]:


ridge=Ridge()


# In[793]:


pipe=make_pipeline(column_trans,scaler,ridge)


# In[794]:


pipe.fit(x_train,y_train)


# In[795]:


y_pred_ridge=pipe.predict(x_test)


# In[796]:


r2_score(y_test,y_pred_ridge)


# In[799]:


print("no regularization:",r2_score(y_test,y_pred))
print("lasso:",r2_score(y_test,y_pred_lasso))
print("ridge:",r2_score(y_test,y_pred_ridge))


# In[800]:


import pickle


# In[801]:


pickle.dump(pipe,open('ridgeModel.pkl','wb'))


# In[ ]:





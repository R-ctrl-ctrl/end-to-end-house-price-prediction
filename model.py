import pandas as pd
df = pd.read_csv('Blr_data.csv')
df.drop(['area_type','availability','society','balcony','balcony'],axis=1,inplace=True)

import string
alphabet_l = string.ascii_lowercase
def rem_str(x):
    if str(x).find('-') != -1:
        les = str(x).split('-')
        avg = (float(les[0]) + float(les[1]))/2
        return avg
    for alpha in alphabet_l:
        if alpha in x:
            return 0
    return x

df.total_sqft = df.total_sqft.apply(rem_str)
df = df[df.total_sqft != 0]

df['size'].fillna('2 BHK')
df['size'] = df['size'].apply(lambda x:str(x).split(' ')[0])
df.bath = df.bath.fillna(df.bath.median())
df.dropna(inplace=True)
locations  =  df.location.value_counts()
locations_less_10 = locations[locations<=10]
df.location = df.location.apply(lambda x:x if x not in locations_less_10 else 'other')
df['total_sqft'] = df['total_sqft'].apply(lambda x:float(x))
df['price_per_sqft'] = df['price']*100000 / df['total_sqft']
df['size'] = df['size'].apply(lambda x:float(x))
df = df[df['total_sqft']/df['size'] >=300]

def rem_price_outlier(df):
    new_df = pd.DataFrame()
    for key,sub_df in df.groupby('location'):
        m = sub_df.price_per_sqft.mean()
        std = sub_df.price_per_sqft.std()
        gen_df = sub_df[(sub_df.price_per_sqft > (m-std)) & (sub_df.price_per_sqft < (m+std))]
        new_df =  pd.concat([new_df,gen_df],ignore_index=True)
    return new_df

new_df = rem_price_outlier(df)

import numpy as np
def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk,bhk_df in location_df.groupby('size'):
            bhk_stats[bhk] = {
                'mean' : bhk_df.price_per_sqft.mean(),
                'std' :  bhk_df.price_per_sqft.std(),
                'count' : bhk_df.shape[0]
            }
            
        for bhk,bhk_df in location_df.groupby('size'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft > (stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
            
new_df =bhk_outlier_remover(new_df)
new_df.drop(['price_per_sqft'],axis=1,inplace=True)

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X = new_df.drop(['price'],axis=1)
y = new_df.price
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
col_trans =make_column_transformer((OneHotEncoder(sparse=False),['location']),remainder='passthrough')
scaler = StandardScaler()
lr = LinearRegression()
pipe = make_pipeline(col_trans,scaler,lr)
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
print(r2_score(y_test,y_pred))

print(new_df.head(15))

import pickle
with open('pipe.pkl','wb') as f:
    pickle.dump(pipe,f)
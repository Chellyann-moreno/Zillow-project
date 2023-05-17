### IMPORTS
import pandas as pd 
import env
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler



# Acquire function:
url=env.get_db_url('zillow')
    
query= """Select prop.parcelid,bathroomcnt, bedroomcnt,calculatedfinishedsquarefeet
        ,yearbuilt,airconditioningtypeid,garagecarcnt, poolcnt,taxvaluedollarcnt,fips, pred.transactiondate
        from properties_2017 as prop
        left join predictions_2017 as pred using(parcelid) 
        where propertylandusetypeid in (261,279)"""
directory='/Users/chellyannmoreno/codeup-data-science/regression-project/'
filename=('zillow.csv')
def get_data():
    if os.path.exists(directory+filename):
        df=pd.read_csv(filename)
        df = df[df['transactiondate'].str.startswith("2017", na=False)]
        return df
    else:
        df=pd.read_sql(query,url)
        df.to_csv(filename,index=False)
        # cache data locally
        df.to_csv(filename, index=False)
        df = df[df['transactiondate'].str.startswith("2017", na=False)]
        return df

# Function to prep file

def wrangle_data(df):
  """ function to prep Zillow data set.
  """ 
  # renameing columns
  df = df.rename(columns={
        'bedroomcnt': 'bedrooms',
        'bathroomcnt': 'bathrooms',
        'calculatedfinishedsquarefeet': 'area',
        'taxvaluedollarcnt': 'taxvalue',
        'fips': 'county','airconditioningtypeid':'air_conditioning','poolcnt':'pool','garagecarcnt':'car_garage'})
 # Filter out rows with large area and filter out places with zero bathrooms and baths, and with more than 15.
  # Drop rows with missing values
  df=df[(df.bathrooms>0) & (df.bedrooms>0) & (df.area>0)]
  df=df[(df.bathrooms<11) & (df.bedrooms<9) &(df.area<18_000)]
  df.pool=df.pool.fillna(0)
  df.car_garage=df.car_garage.fillna(0)
  df=df[df.car_garage<5]
  df.air_conditioning=df.air_conditioning.fillna(0)
  df=df.dropna(subset=['yearbuilt','taxvalue'])
  df['car_garage']=df.car_garage*1
  df['air_conditioning']=df.air_conditioning.astype(bool)*1

    # Convert data types
  df[['bedrooms', 'area', 'taxvalue', 'yearbuilt']] = df[['bedrooms', 'area', 'taxvalue', 'yearbuilt']].astype(int)

    # Map county codes to names and 1/0
  county_map = {6037: 'LA', 6059: 'Orange', 6111: 'Ventura'}
  df.county = df.county.map(county_map)
  dummy_df = pd.get_dummies(df['county'])
  df = pd.concat( [df,dummy_df], axis=1 )
  dummy_df = pd.get_dummies(df['county'])
  df = pd.concat( [df,dummy_df], axis=1 )
  return df

# function to split data into train,validate and test
def split_data(df):
    " Function to split dataset"
    # Split into train_validate and test sets
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # Split into train and validate sets
    train, validate = train_test_split(train_validate, test_size=.25, random_state=123)

    return train, validate, test

# Function to scale features:
" Function to scale data"
def standard_scale_data(X_train, X_validate, X_test):
    # Initialize StandardScaler object
    scaler = StandardScaler()
    
    # Fit scaler object to training data
    scaler.fit(X_train)
    
    # Transform training and validation data
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)
    # Return scaled data
    return X_train_scaled, X_validate_scaled, X_test_scaled
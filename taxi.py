# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 10:10:04 2016

@author: brjohn
"""

import pandas as pd
import datetime
import random
import math
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler


in_path = 'C:/users/brjohn/desktop/datasets/green_tripdata_2015-09.csv'

data = []
errors = []

o = open(in_path)
header = o.readline().strip().split(',')
num_fields = len(header)

for line in o.readlines():
    line_data = line.strip().split(',')
    if len(line_data) == num_fields:
        data.append(line_data)
    else:
        errors.append(line_data)
        
o.close()

taxi_data = pd.DataFrame(data, columns = header)
taxi_data.info()

taxi_data['Fare_amount'] = taxi_data['Fare_amount'].astype(float)
taxi_data['Tip_amount'] = taxi_data['Tip_amount'].astype(float)

tip_data = taxi_data[taxi_data['Tip_amount'] > 0]
tip_data = tip_data[tip_data['Fare_amount'] > 0]

tip_data['Tip_percent'] = tip_data.apply(lambda x: x['Tip_amount'] / x['Fare_amount'], axis = 1)

#explore tip data:
print tip_data['Tip_percent'].describe()   #avg tip is pretty good (24%) max value = 150
                                            # std = .6, data has too much variance
print tip_data['Tip_percent'].quantile([.5, .9, .95, .99]) # 99% of tips are below 51%
#examine some outliers
print tip_data[tip_data['Tip_percent'] > 1][['Fare_amount', 'Tip_amount']][:10]  
#many of the tips which are greater than 100% seem like data errors, so we drop these records
tip_data = tip_data[tip_data['Tip_percent'] < .75]
print tip_data['Tip_percent'].hist(bins = 100) #looks good
print tip_data['Tip_percent'].describe() #avg tip = .2, std = .07, much better

def day_of_week(date):
    pickup = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    if (pickup.weekday() > 4 or date[:10] == '2015-09-07'): #counts labor day as a weekend
        return 'weekend'
    else:
        return 'weekday'

def time(date):
    pickup = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    if pickup.weekday() <= 4:
        if 7 <= pickup.hour <= 9:
            return 'morn_commute'
        elif 9< pickup.hour < 17:
            return 'daytime'
        elif 17<= pickup.hour <=19:
            return 'evening_commute'
    if (pickup.weekday() in [4,5] or date[:10] == '2015-09-06') and pickup.hour > 19:
            return 'going_out'
    if (pickup.weekday() in [5,6] or date[:10] == '2015-09-07') and pickup.hour < 4:
            return 'coming_home'
            
def dummy_data(DF, field):
    dummies = pd.get_dummies(DF[field], prefix=field)
    DF = DF.join(dummies)
    DF.drop(field, inplace=True, axis = 1)
    return DF
    
def RMSE(pred, actual):
    rmse = 0
    n = len(pred)
    for i, j in zip(pred, actual):
        rmse += (i-j)**2    
    return math.sqrt(rmse)/n
    
def create_stack_frame(train, scale):
    train_scale = scale.fit_transform(train)
    data = zip(rf.predict(train), ll.predict(train_scale))
    return pd.DataFrame(data, columns = ['rf', 'll'])

day_dict = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Sautrday', 6:'Sunday'}

tip_data['weekday'] = tip_data.apply(lambda x: day_of_week(x['lpep_pickup_datetime']), axis = 1)
tip_data['time'] = tip_data.apply(lambda x: time(x['lpep_pickup_datetime']), axis = 1)
tip_data['labor_day'] = tip_data.apply(lambda x: 1 if x['lpep_pickup_datetime'][:10] == '2015-09-07' else 0, axis = 1)
tip_data['day_of_week'] = tip_data.apply(lambda x: day_dict[datetime.datetime.strptime(x['lpep_pickup_datetime'], '%Y-%m-%d %H:%M:%S').weekday()], axis = 1)

vars_usd = ['Passenger_count', 'Fare_amount', 'Extra', 'day_of_week',
            'Tolls_amount', 'Trip_type', 'weekday','time','labor_day']

to_float = ['Passenger_count', 'Fare_amount', 'Extra', 'Tolls_amount']
to_dummy = ['Trip_type', 'weekday', 'time', 'day_of_week']

#splitting data into training and validation sets
tip_data['train'] = tip_data.apply(lambda x: 0 if random.random() < .3 else 1, axis = 1)

train_v = tip_data[vars_usd+['train', 'Tip_percent']]

for var in to_float:
    train_v[var] = train_v[var].astype(float)

for var in to_dummy:
    train_v = dummy_data(train_v, var)

train = train_v[train_v['train'] == 1]
validate = train_v[train_v['train'] == 0]

train_target = train['Tip_percent']
validate_target = validate['Tip_percent']

train.drop(['Tip_percent','train'], axis = 1, inplace=True)
validate.drop(['Tip_percent','train'], axis = 1, inplace=True)

train_mat = train.as_matrix()
validate_mat = validate.as_matrix()

#  models

# randomForest - many features or more or less categorical + randomForest generally performs well

param_grid = {'max_features' : ['auto', 'log2'],
              'max_depth' : [2, 6],
              'n_estimators' : [50, 100, 150]}
"""
rf_pre = RandomForestRegressor(n_jobs = -1, random_state=12)
clf = GridSearchCV(rf_pre, param_grid, cv = 5)
clf.fit(train_mat, train_target)

print clf.best_params_
rf = clf.best_estimator_
"""
rf = RandomForestRegressor(n_jobs = -1, n_estimators = 150, max_depth = 6)

rf.fit(train_mat, train_target)

a = zip(train.columns.values,rf.feature_importances_)
a.sort(key=lambda x: -x[1])

print "feature importances: \n"
for i in a:
    print i

score = rf.score(validate_mat, validate_target)
print score
pred = rf.predict(validate_mat)
print "RMSE: ", RMSE(pred, validate_target)

# lasso - simply linear model with regularization term

scale = StandardScaler()    #lasso is sensitive to variable scale 
train_mat_scale = scale.fit_transform(train_mat)
validate_mat_scale = scale.transform(validate_mat)

ll = linear_model.LassoCV(cv=5, eps=.001, n_alphas = 200, max_iter=2000)
ll.fit(train_mat_scale, train_target)

pred = ll.predict(validate_mat_scale)
print "RMSE: ", RMSE(pred, validate_target) #performance < RF performance

# stacking - trying to get an improvement in accuracy by combining the two models above

stack_frame = create_stack_frame(train_mat, scale)
stack_validate = create_stack_frame(validate_mat, scale)

lm = linear_model.LinearRegression()
lm.fit(stack_frame, train_target)

pred = lm.predict(stack_validate)
print "RMSE: ", RMSE(pred, validate_target)  # not much better than RF alone, but RMSE is about
                                             # 1 BP, so we'll take it.

#### Q5

#preprocessing
taxi_data['Pickup_longitude'] = taxi_data['Pickup_longitude'].astype(float)
taxi_data['Pickup_latitude'] = taxi_data['Pickup_latitude'].astype(float)

taxi_data['lat_bin'], lat_bins = pd.qcut(taxi_data['Pickup_latitude'], 200, 
                                        labels = range(1,201), retbins=True)
taxi_data['lon_bin'], lon_bins = pd.qcut(taxi_data['Pickup_longitude'], 200, 
                                        labels = range(1,201), retbins = True)

def find_bin(num, bins):
    i = 1
    for endpt in bins:
        if endpt > num:
            return i
        i += 1    

def find_nearest(df, loc, k, max_iter = 5):
    """
    loc is an array (lat, long), k is num_neighbors
    max_iter controls the number of times the boundries are extended to 
        find nbrs
    """
    lat, lon = loc    
    
    lat_bin = find_bin(lat, lat_bins)
    lon_bin = find_bin(lon, lon_bins)
    
    if abs(lat_bins[lat_bin-1] - lat) < abs(lat_bins[lat_bin-2] - lat):
        adj_lat = 1
    else:
        adj_lat = -1
    
    if abs(lon_bins[lon_bin-1] - lon) < abs(lon_bins[lon_bin-2] - lon):
        adj_lon = 1
    else:
        adj_lon = -1
    
    
    lat_check = [lat_bin, lat_bin+adj_lat]
    lon_check = [lat_bin, lon_bin+adj_lon]
    
    pn = df[df['lat_bin'].isin(lat_check)]
    pn = pn[pn['lon_bin'].isin(lon_check)]
    
    pn['dist'] = pn.apply(lambda x: math.sqrt(abs(lat-x['Pickup_latitude']) + abs(lon - x['Pickup_longitude'])), axis = 1)
    pn.sort(columns = 'dist', ascending = True, inplace=True)
    
    attempts = 1
    
    while len(pn) < k and attempts < max_iter:
            attempts +=1
            
            print "Looking for nbrs. Iter: ", attempts            
            
            lat_check.append(lat_bin - adj_lat)
            lon_check.append(lat_bin - adj_lon)
            adj_lat += adj_lat
            adj_lon += adj_lon
            
            lat_check = [i for i in lat_check if 0 <= i <= 200]
            lon_check = [i for i in lon_check if 0 <= i <= 200]
        
            pn = df[df['lat_bin'].isin(lat_check)]
            pn = pn[pn['lon_bin'].isin(lon_check)]
    
            pn['dist'] = pn.apply(lambda x: math.sqrt(abs(lat-x['Pickup_latitude']) + abs(lon - x['Pickup_longitude'])), axis = 1)
            pn.sort(columns = 'dist', ascending = True, inplace=True)        
        
    part_list = pn[:k]
    
    return part_list, attempts
    
def good_time(time1, time2):
    """
    Expects time1 in standard format: YYYY-MM-DD hh:mm:ss
    time2 with only time: hh:mm:ss
    """
    
    t1 = datetime.datetime.strptime(time1[11:], '%H:%M:%S')
    t2 = datetime.datetime.strptime(time2, '%H:%M:%S')
    
    d = t1 - t2
    
    if d.days == -1:
        d = t2 - t1
        
    return d.seconds <= 30.0*60
    

def find_nearest_with_time(df, loc, k, time, max_iter=5):
    pos_shares = df.apply(lambda x: good_time(x['lpep_pickup_datetime'], time), axis = 1)
    good_times = df[pos_shares]
    
    return find_nearest(good_times, loc, k, max_iter)
    
    
    
    
    







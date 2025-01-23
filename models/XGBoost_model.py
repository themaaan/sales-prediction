import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

def create_features(data):
    """
    Create tiem series features based on time series index.
    """
    data = data.copy()
    data['dayofweek'] = data.index.dayofweek
    data['hour'] = data.index.hour
    data['year'] = data.index.year
    data['quarter'] = data.index.quarter
    data['month'] = data.index.month
    data['dayofyear'] = data.index.dayofyear
    data['daysinmonth'] = data.index.day
    data['weekofyear'] = data.index.isocalendar().week
    return data

def lag_features(data):
    target_map = data["courier_partners_online"].to_dict()
    data["lag1"] = (data.index - pd.Timedelta("364 days")).map(target_map)
    data["lag2"] = (data.index - pd.Timedelta("728 days")).map(target_map)
    data["lag3"] = (data.index - pd.Timedelta("1092 days")).map(target_map)
    return data

def cross_validation(data, days, plot = False):
    tss = TimeSeriesSplit(n_splits=5, test_size=days)
    df = data.sort_index()
    fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)
    
    fold = 0
    preds = []
    scores = []
    for train_idx, val_idx in tss.split(df) :
        train = df.iloc[train_idx]
        test = df.iloc[val_idx]
        if plot:
            train['courier_partners_online'].\
                plot(ax=axs[fold],
                    label='Training Set',
                    title=f'Data Train/Test Split Fold {fold}')

            test[ 'courier_partners_online' ].\
                plot(ax=axs[fold],
                    label= 'Test Set')
            
            axs[fold].axvline(test.index.min(), color='black', ls ='--')
            fold += 1
            

        train = create_features(train)
        test = create_features(test)
        FEATURES = ['dayofyear',
                    'hour',
                    'dayofweek', 
                    'quarter', 
                    'month', 
                    'year',
                    'lag1', 
                    'lag2', 
                    'lag3' ]
        TARGET = 'courier_partners_online'
        
        X_train = train[ FEATURES]
        y_train = train [TARGET]

        X_test = test [FEATURES]
        y_test = test [TARGET]
        
        reg = xgb.XGBRegressor (base_score=0.5,
                                booster='gbtree',
                                n_estimators=1000, 
                                early_stopping_rounds=50, 
                                objective='reg:linear', 
                                max_depth=3, 
                                learning_rate=0.01)
        
        reg.fit(X_train, y_train,
                eval_set=[ (X_train, y_train), (X_test, y_test)],
                verbose=100)
        y_pred = reg.predict(X_test)
        preds.append(y_pred)
        score = np.sqrt(r2_score(y_test, y_pred))
        scores.append(score)
    plt.show()
    print("scores: ", scores)
    print("mean value of scores: ", np.mean(scores))
    return np.mean(scores)
  
def prediction(df, days = 14, plot = False):
    # DO the prediction:
    df = create_features(df)
    FEATURES = ['dayofyear',
                'hour',
                'dayofweek', 
                'quarter', 
                'month', 
                'year',
                'lag1', 
                'lag2', 
                'lag3' ]
    TARGET = 'courier_partners_online'
    
    X_all = df[FEATURES]
    y_all = df[TARGET]
    
    reg = xgb.XGBRegressor (base_score=0.5,
                            booster='gbtree',
                            n_estimators=1000, 
                            early_stopping_rounds=50, 
                            objective='reg:linear', 
                            max_depth=3, 
                            learning_rate=0.01)
    
    reg.fit(X_all, y_all,
            eval_set=[ (X_all, y_all)],
            verbose=100)
    
    # Create the future data
    max_date = df.index.max()           
    end_date = max_date + pd.Timedelta(14, 'D')  
    future = pd.date_range(start=max_date, end=end_date, freq='D')

    

    future_df = pd.DataFrame (index=future)
    future_df ['isFuture'] = True
    df[ 'isFuture'] = False
    df_and_future = pd.concat([df, future_df])
    df_and_future = create_features(df_and_future)
    df_and_future = lag_features(df_and_future)
    
    future_data = df_and_future.query("isFuture").copy()
    future_data['pred'] = reg.predict(future_data[FEATURES])

    future_data = future_data.reset_index().rename(columns={'index': 'date', 'pred': 'xgb_pred'})
    future_data = future_data[["date", "xgb_pred"]]
    
    return future_data

def forecast_courier_XGBoost(data, days = 14, plot = False):
    df = data.copy()
    df['date'] = pd.to_datetime(data['date'])
    df = df.set_index('date')
    df = df[['courier_partners_online']]

    df = lag_features(df)
    df = create_features(df)
    
    score = cross_validation(df, days = days, plot = plot)
    future_data = prediction(df, days = days, plot = plot)

    return future_data.tail(14).reset_index()[["date", "xgb_pred"]], score
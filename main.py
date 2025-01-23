import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.impute import KNNImputer
from models.XGBoost_model import *
from models.Prophet_model import *
#from pmdarima import *

def plot_histograms(data):
    """
    Plots a histogram for each numeric column in the dataset in a grid layout.
    We select only the numerical values, since there are 5 variabels, from which 1 is date and other and others are numerical
    """
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    cols = 2  
    rows = 2  

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  

    for i, column in enumerate(numeric_columns):
        ax = axes[i]
        data[column].hist(bins=99, ax=ax, edgecolor='black')
        ax.set_title(column)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    for i in range(len(numeric_columns), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_date(data):
    """
    Plots all numeric columns against a date column in a grid layout.
    """
    data['date'] = pd.to_datetime(data['date'])

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    cols = 2  
    rows = 2  

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten() 

    for i, column in enumerate(numeric_columns):
        ax = axes[i]
        ax.plot(data['date'], data[column], label=column, alpha=0.8)
        ax.set_title(column)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')
        
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))  # January and May
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    for i in range(len(numeric_columns), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

def data_cleaning(raw_data):

    raw_data.loc[raw_data['courier_partners_online'] > 150, 'courier_partners_online'] = np.nan
    
    numeric_data = raw_data.select_dtypes(include=['float64', 'int64'])
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(numeric_data)
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_data.columns, index=numeric_data.index)
    raw_data[numeric_data.columns] = imputed_df

    return raw_data

def inital_EDA(raw_data):
    
    print(raw_data.isnull().sum())
    plot_histograms(raw_data)
    plot_date(raw_data)

def post_processing(data, forecast_prophet, forecast_xgb):
    # print preds:
    df_orig = data.copy()
    df_orig = df_orig[["date", "courier_partners_online"]]
    df_orig = df_orig.rename(columns={"courier_partners_online": "courier"})
    print("PÄÄ: \n",df_orig)
    print("forecast_prophet: \n", forecast_prophet)
    print("forecast_xgb: \n", forecast_xgb)

    df_orig['date'] = pd.to_datetime(df_orig['date'])

    df_orig_extended = pd.concat([df_orig,
                             pd.DataFrame({'date': pd.date_range(df_orig['date'].max() + pd.Timedelta(days=1), periods=len(forecast_prophet)),
                                            'courier': np.nan})], ignore_index=True)

    df_xgb = pd.concat([forecast_xgb,
                             pd.DataFrame({'date': pd.date_range(end= forecast_xgb['date'].min() - pd.Timedelta(days=1), periods=len(df_orig)),
                                            'xgb_pred': np.nan})], ignore_index=True)
    
    df_prophet = pd.concat([forecast_prophet,
                             pd.DataFrame({'date': pd.date_range(end= forecast_prophet['date'].min() - pd.Timedelta(days=1), periods=len(df_orig)),
                                            'prophet_pred': np.nan})], ignore_index=True)
    

    final_df = (df_orig_extended.merge(df_xgb, on="date", how="inner").\
                merge(df_prophet, on="date", how="inner").\
                sort_values("date").reset_index(drop=True))
    return final_df

def modelling(data, plot = False):

    # Facebook's prophet model
    forecast_prophet, score_prophet = forecast_courier_prophet(data, plot = plot)
    
    # Auto_arima


    # XGBoost
    forecast_xgb, score_xgb = forecast_courier_XGBoost(data, days=14, plot = plot)

    final_df = post_processing(data, forecast_prophet, forecast_xgb)
    print(final_df)
    print("score_xgb: ", score_xgb)
    print("score_prophet: ", score_prophet)

    return final_df, score_prophet, score_xgb

def post_EDA(final_df, score_prophet, score_xgb):
    """
    Visualizes columns of interest from final_df over time.
    Assumes final_df has a 'date' column and numeric columns like 'courier', 'xgb_pred', 'prophet_pred'.
    """

    
    final_df['date'] = pd.to_datetime(final_df['date'])

    start_date = final_df.loc[final_df['courier'].isna(), 'date'].min() - pd.DateOffset(months=6)

    final_df = final_df[(final_df['date'] >= start_date)]

    
    columns_to_plot = {'courier': r'$\mathbf{Actual\ data}$',
                        'xgb_pred': r'$\mathbf{XGBoost\ (R^2\ =\ %.3f)}$' % score_xgb,
                        'prophet_pred': r'$\mathbf{Prophet\ (R^2\ =\ %.3f)}$' % score_prophet}

    plt.figure(figsize=(10, 6))
    for col, label in columns_to_plot.items(): 
        plt.plot(final_df['date'], final_df[col], label=label, alpha=0.8)

    plt.xlabel('Date')
    plt.ylabel('Amount of couriers')
    plt.title('Visualization of actual and predicted data.')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():

    plot = True
    # read the initial data
    raw_data = pd.DataFrame(pd.read_csv("data/daily_cp_activity_dataset.csv"))

    # Do the inital EDA
    if plot:
        inital_EDA(raw_data)
    
    # clean some outliers and fill the missing values.
    data = data_cleaning(raw_data)

    # Viewing the same plots after cleaning data
    if plot:
        inital_EDA(data)

    # Modelling and predicting the courier partners
    final_df, score_prophet, score_xgb = modelling(data, plot=plot)

    # Post EDA
    post_EDA(final_df, score_prophet, score_xgb)


if "__main__":
    main()
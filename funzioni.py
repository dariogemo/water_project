import numpy as np
import pandas as pd
import seaborn as sns

def outl_del(df, column, quantile_h, quantile_l = 0):
    '''
    function for deleting outliers
    '''
    outl_high = df[column].quantile(quantile_h)
    out_low = df[column].quantile(quantile_l)
    mask = (df[column] < outl_high) & (df[column] > out_low)
    df = df[mask]
    return df 

def median_col_in_target(df, column):
    '''
    function for median in target 0 and 1
    '''
    x = df[df['Potability'] == 0][column].median()
    y = df[df['Potability'] == 1][column].median()
    return x, y

def na_to_median_target(df, column):
    '''
    function for changing na's to median value, considering target
    '''
    df_0 = df[df['Potability'] == 0][column]
    df_1 = df[df['Potability'] == 1][column]
    df_0_median = df_0.median()
    df_1_median = df_1.median()
    df.loc[df['Potability'] == 0, column] = df.loc[df['Potability'] == 0, column].fillna(df_0_median)
    df.loc[df['Potability'] == 1, column] = df.loc[df['Potability'] == 1, column].fillna(df_1_median)
    
def na_to_median(df, column : int) -> int:
    '''
    function for changing na's to median value of all values in column
    '''
    return df[column].fillna(df[column].median(), inplace = True)
    
def na_to_norm_distr(df, column):
    '''
    function for changing na's in a column to sample from a normal distribution
    '''
    mean_value = df[column].mean()
    std_dev = df[column].std()
    random_values = np.random.normal(mean_value, std_dev, size=len(df))
    df[column] = df[column].fillna(pd.Series(random_values))

    
def na_to_unif_distr(df, column):
    '''
    function for changing na's in a column to sample from a uniform distribution
    '''
    high_unif = df[column].max()
    low_unif = df[column].min()
    random_values_unif = np.random.uniform(low_unif, high_unif, size=len(df))
    df[column] = df[column].fillna(pd.Series(random_values_unif))
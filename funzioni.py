import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def outl_del(df : pd.DataFrame, column, quantile_h : int, quantile_l : int = 0):
    '''
    Delete outliers of a given column using the high percentile and low percentile in input.\n
    Default low quantile is set to 0 and it's optional, since not all features need to delete outliers from the low.
    '''
    outl_high = df[column].quantile(quantile_h)
    out_low = df[column].quantile(quantile_l)
    mask = (df[column] < outl_high) & (df[column] > out_low)
    df = df[mask]
    return df 

def median_col_in_target(df : pd.DataFrame, column : str):
    '''
    Function for printing the medians of the values of a given column with Potability = 0 and Potability = 1
    '''
    x = df[df['Potability'] == 0][column].median()
    y = df[df['Potability'] == 1][column].median()
    return x, y

def na_to_median_target(df : pd.DataFrame, column : str):
    '''
    Fill NA's of the input column with the median of the values, considering the target variable.\n
    The rows with 'Potability' = 0 have their null values filled with the median of the rows with 'Potability' = 0.\n
    The rows with 'Potability' = 1 have their null values filled with the median of the rows with 'Potability' = 1.
    '''
    df_0 = df[df['Potability'] == 0][column]
    df_1 = df[df['Potability'] == 1][column]
    df_0_median = df_0.median()
    df_1_median = df_1.median()
    df.loc[df['Potability'] == 0, column] = df.loc[df['Potability'] == 0, column].fillna(df_0_median)
    df.loc[df['Potability'] == 1, column] = df.loc[df['Potability'] == 1, column].fillna(df_1_median)
    
def na_to_median(df : pd.DataFrame, column : str):
    '''
    Fill NA's with the median of the values inside the input column.
    '''
    return df[column].fillna(df[column].median(), inplace = True)
    
def na_to_norm_distr(df: pd.DataFrame, column : str):
    '''
    Fill NA's for a given column with random samples picked from a normal distribution.\n
    The normal distribution mean and standard deviation are chosen from the ones of the column.
    '''
    mean_value = df[column].mean()
    std_dev = df[column].std()
    random_values = np.random.normal(mean_value, std_dev, size=len(df))
    df[column] = df[column].fillna(pd.Series(random_values))

    
def na_to_unif_distr(df : pd.DataFrame, column : str):
    '''
    Fill NA's for a given column with random samples picked from a uniform distribution\n
    The uniform distribution interval is between the minimum and the maximum of the values inside the column.
    '''
    high_unif = df[column].max()
    low_unif = df[column].min()
    random_values_unif = np.random.uniform(low_unif, high_unif, size = len(df))
    df[column] = df[column].fillna(pd.Series(random_values_unif))
    
def k_fold_accuracy(X1, y1, lista : list):
    '''
    Ten-Fold cross validation using KFold, for a given list of numbers of decision trees. Each number of decision trees in "lista" is applied to the Random Forest Classifier and then the model is cross validated.\n
    It returns a list of scores containing the means of the accuracy scores for each group and a list of number of decision trees relative to the scores.
    '''
    scores = []
    n_est = []
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.3, random_state = 1)
    for x in lista:
        n_est.append(x)
        clf_svm = RandomForestClassifier(n_estimators = x, random_state = 1)
        clf_svm.fit(X_train, y_train)
        y_pred_svm = clf_svm.predict(X_test)
        kf = KFold(n_splits = 10, shuffle = True, random_state = 1)
        results = cross_val_score(clf_svm, X1, y1, cv = kf)
        results_mean = round(results.mean() * 100, 2)
        scores.append(results_mean)
    return scores, n_est
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import minmax_scale

def minmax_normalise():
    '''
    mix-max scale the data between 0 and 1
    '''
    df = pd.read_csv('data/final_data.csv')
    
    df['HP'] = minmax_scale(df['Mean of Median House Price'])
    df['SC'] = minmax_scale(df['Total School Score'])
    df['CR'] = minmax_scale(df['Crime Rate'])
    df['OS'] = minmax_scale(df['Total open space'])
    df['MP'] = minmax_scale(df['Medical Person Rate'])
    
    df_nor = df.loc[:, ['LGA','HP','SC','CR','OS','MP']].fillna(0)
    df_nor.to_csv('data/normalised final data.csv', index=False)
    return



# ===================================================================


def cal_pearson():
    '''
    calculate the pearson correlation coefficients
    for each feature and the median house price
    '''
    lst = []
    lst.append(pearson('SC'))
    lst.append(pearson('CR'))
    lst.append(pearson_os('OS'))
    lst.append(pearson('MP'))

    data = {'feature':['SC', 'CR', 'OS', 'MP'], 'pearson r':lst}
    dp = pd.DataFrame(data)
    dp.to_csv('calculation_result/pearson.csv')
    return


# =====================================================================


def cal_nmi():
    '''
    calculate the normalised mutual infomation between
    each feature and the median house price
    '''
    df = pd.read_csv('data/normalised final data.csv')
    df ['OS'] = df['OS'].replace(0, np.nan)
    
    # dicretise continuous data into 5 equal-length bins
    df['SC_bin']=bin_5(df, 'SC')
    df['CR_bin']=bin_5(df, 'CR')
    df['MP_bin']=bin_5(df, 'MP')
    df['HP_bin']=bin_5(df, 'HP')
    df['OS_bin']=bin_5(df, 'OS')

    lst = []
    lst.append(normalized_mutual_info_score(df['SC_bin'], df['HP_bin']))
    lst.append(normalized_mutual_info_score(df['CR_bin'], df['HP_bin']))
    lst.append(normalized_mutual_info_score(df['MP_bin'], df['HP_bin']))
    
    # remove rows that does not have OS data
    df2 = df[~df['OS_bin'].isnull()]
    lst.append(normalized_mutual_info_score(df2['OS_bin'], df2['HP_bin']))

    data = {'feature':['SC', 'CR', 'OS', 'MP'], 'nmi':lst}
    dp = pd.DataFrame(data)
    dp.to_csv('calculation_result/nmi.csv')
    return

# =============================================================

def cal_regression_old():
    '''
    build the regression model based on normalised data
    '''
    df = pd.read_csv('data/normalised final data.csv')
    cal_regression(df)
    return
    
# =============================================================

def cal_regression_new():
    '''
    an updated regression model which only consider house price
    data that lies between two standard deviations from the mean
    '''
    df = pd.read_csv('data/normalised final data.csv')
    # find the mean and standard deviation of HP data
    df_mean = df['HP'].mean()
    df_sd = df['HP'].std()
    upper_limit = df_mean + 2*df_sd
    df2 = df.loc[df['HP'] < upper_limit]

    cal_regression(df2)
    return

# helper functions -------------------------------------------------------------

def pearson(x):
    df = pd.read_csv('data/normalised final data.csv')
    return df[x].corr(df['HP'])

# ================================================================

def pearson_os(x):
    df = pd.read_csv('data/normalised final data.csv')
    df[x] = df[x].replace(0, np.nan)
    df = df.dropna()
    return df[x].corr(df['HP'])

# ===============================================================

# pre-processing: dicretise continuous data into bins
def bin_5(df, feature):
    label = ["low", "medium low", "medium", "medium high", "high"]
    min_v = df[feature].min()
    max_v = df[feature].max()
    interval = (max_v - min_v)/5
    return pd.cut(x=df[feature], bins=[min_v-1, min_v+interval, min_v+(2*interval),
                         min_v+(3*interval), max_v-interval, max_v], labels=label)

# =================================================================

def cal_regression(df):
    # create the design matrix using the variables
    X = df[['SC', 'CR', 'MP', 'OS']]
    y = df['HP']

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lm = LinearRegression()

    model = lm.fit(X_train, y_train)
    y_test_predictions = lm.predict(X_test)
    print('actual HP values of the first 5 test data:')
    print(y_test[0:5])
    print('')
    print('predicted HP values of the first 5 test data:')
    print(y_test_predictions[0:5])
    print('')

    # coefficient
    print('Coefficient: ', end = ' ')
    print(lm.coef_)
    print('')

    # intercept:
    print('Intercept: ', end = ' ')
    print(lm.intercept_)
    print('')

    # R^2
    r2_test = lm.score(X_test, y_test)
    r2_train = lm.score(X_train, y_train)

    print('Coefficient of determination (test): {0:.2f}'.format(r2_test))
    print('Coefficient of determination (training): {0:.2f}'.format(r2_train))
    return



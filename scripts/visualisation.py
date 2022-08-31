import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot():
    '''
    sketch a boxplot and four scatterplots used in the research
    '''
    df = pd.read_csv('data/normalised final data.csv')
    df['OS'] = df['OS'].replace(0,np.nan)
    
    # box plot showing distribution of median house price in Victoria
    sketch_box()
    
    # scatter plots compaing each feature and median house price
    sketch_scatter(df, 'MP', "Number of medical person vs Median House Price",\
           "total medical person per 100000", 'health workforce vs house price.png', 'red')
    
    sketch_scatter(df, 'SC', "Number of schools vs Median House Price",\
           "number of schools", 'school vs house price.png', 'green')
    
    sketch_scatter(df, 'OS', "Number of open space vs Median House Price",\
           "number of open space", 'open space vs hp 19.png', 'blue')
    
    sketch_scatter(df, 'CR', "Crime rate vs Median House Price",\
           "crime rate per 100000", 'crime vs hp 19.png', 'brown')
    return


# ==========================================================================


def sketch_scatter(df, coloumn, title, xlabel, figname, color):
    # sketch a scatterplot showing the correlation between median house price and number of schools
    
    plt.scatter(df[coloumn], df['HP'], color=color)
    plt.title(title, size = 14)
    plt.xlabel(xlabel, size = 12)
    plt.ylabel("house price", size = 12)
    plt.savefig("plots/" + figname, bbox_inches = "tight")
    plt.close()
    return


# ========================================================================


def sketch_box():
    df = pd.read_csv('data/final_data.csv')
    fig = plt.figure(figsize=(9, 5), dpi=100)
    plt.boxplot(df["Mean of Median House Price"], whis = 1.5, vert=False)
    plt.title('Median House Price in 2019', size = 14)
    plt.xlabel('Median House Price', size = 12)
    fig.savefig("plots/boxplot of hp.png")
    plt.close()
    return

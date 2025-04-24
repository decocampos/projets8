import pandas as pd

def get_summary(df):
    return df.describe(include='all').transpose()

def get_missing_values(df):
    return df.isnull().sum()

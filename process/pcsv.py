import pandas as pd
import numpy as np

def preprocess(filename):

    df = pd.read_csv(filename)
    columns = list(df.columns)

    columns.remove("num")

    missing_cols = []

    for column in columns:
        try :
            col = df[column]
            miss_val_count = col.value_counts()["?"]
            print(miss_val_count,"- missing values in column ",column)
            missing_cols.append(column)
        except : 
            continue

    for column in missing_cols:
        mode = df[column].mode()[0]
        df[column] = df[column].replace('?', mode).astype('int64')

    df_scaled = df.copy()

    columns.remove("sex")
    columns.remove("cp")
    columns.remove("fbs")
    columns.remove("restecg")
    columns.remove("exang")
    columns.remove("slope")
    columns.remove("ca")
    columns.remove("thal")

    for column in columns:
        col_min = df[column].min()
        col_max = df[column].max()
        df_scaled[column] = (df[column] - col_min) / (col_max - col_min)

        # mean = np.mean(df[column])
        # std = np.std(df[column])
        # df_scaled[column] = (df[column] - mean) / std

    df_scaled.to_csv("data/processed-cleveland.csv")

    return(df_scaled)
import pandas as pd

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

    for column in columns:
        col_min = df[column].min()
        col_max = df[column].max()
        df_scaled[column] = (df[column] - col_min) / (col_max - col_min)

    return(df_scaled)
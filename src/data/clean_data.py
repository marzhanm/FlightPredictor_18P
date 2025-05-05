import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    Q1 = df["Price"].quantile(0.25)
    Q3 = df["Price"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df["Price"] >= lower) & (df["Price"] <= upper)]

    df.to_csv('/Users/timurchiks/Desktop/flight_price_predictor/data/processed/cleaned_data.csv')
    return df

    

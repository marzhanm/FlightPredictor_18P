import pandas as pd
from sklearn.preprocessing import LabelEncoder

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df['curr_date'] = pd.to_datetime(df['curr_date'])
    df['departure_date'] = pd.to_datetime(df['departure_date'])

    df['days_until_flight'] = (df['departure_date'] - df['curr_date']).dt.days
    df['part_of_day'] = df['time_interval'].apply(get_part_of_day)
    df = df.drop(columns=['curr_date', 'departure_date', 'time_interval', 'arrival_date', 'is_weekends'])
    preprocessed_data = preprocess(df)
    x = preprocessed_data.drop(columns=['Price'])
    y = preprocessed_data['Price']
    df.to_csv('/Users/timurchiks/Desktop/flight_price_predictor/data/processed/built.csv')

    return x, y

def get_part_of_day(time_str):

    start_hour = int(time_str.split('-')[0].split(':')[0])
    
    if 5 <= start_hour < 12:
        return 'утро'
    elif 12 <= start_hour < 17:
        return 'день'
    elif 17 <= start_hour < 22:
        return 'вечер'
    else:
        return 'ночь'
    
def preprocess(df):
    df = df.copy()
    le = LabelEncoder()
    cat_features = ['From', 'To', 'avialine', 'part_of_day']

    for col in cat_features:
        df[col] = le.fit_transform(df[col])

    return df
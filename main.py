from src.data.load_data import load_data
from src.data.clean_data import clean_data
from src.features.build_features import build_features
from src.models.train_model import train_and_save_model
from src.models.evaluate_model import evaluate_model

df = load_data('/Users/timurchiks/Desktop/flight_price_predictor/data/processed/raw_data.csv')

print("data loaded")

df_clean = clean_data(df)

print('data cleaned')

x, y = build_features(df_clean)

print('features have built')

train_and_save_model(x, y)

print('model has trained')

evaluate_model(x, y)

print('end')
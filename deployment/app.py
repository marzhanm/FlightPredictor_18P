import streamlit as st
import pandas as pd
import joblib
import datetime
import holidays

model = joblib.load('/Users/timurchiks/Desktop/flight_price_predictor/models/joblib/gb_v2.joblib')
kz_holidays = holidays.KZ()
df = pd.read_csv('/Users/timurchiks/Desktop/flight_price_predictor/data/processed/built.csv')
le_avialine = joblib.load('/Users/timurchiks/Desktop/flight_price_predictor/models/encoder/le_avialine.pkl')
le_From = joblib.load('/Users/timurchiks/Desktop/flight_price_predictor/models/encoder/le_From.pkl')
le_To = joblib.load('/Users/timurchiks/Desktop/flight_price_predictor/models/encoder/le_To.pkl')
le_part_of_day = joblib.load('/Users/timurchiks/Desktop/flight_price_predictor/models/encoder/le_part_of_day.pkl')


airline = st.selectbox("Авиалиния", ['Air Astana', 'FlyArystan', 'Qazaq Air', 'SCAT'])
from_city = st.selectbox("Город вылета", ['Астана', 'Алматы', 'Шымкент'])
to_city = st.selectbox("Город прилета", ['Астана', 'Алматы', 'Шымкент'])
curr_date = datetime.date.today()
departure_date = st.date_input("Дата вылета", value=curr_date)
is_holiday = departure_date in kz_holidays
part_of_day = st.selectbox("Время вылета", ['утро', 'день', 'вечер', 'ночь'])

matching_rows = df[(df['From'] == from_city) & (df['To'] == to_city)]

duration = matching_rows['duration'].median()
days_until_flight = (departure_date - curr_date).days

x = pd.DataFrame()

x['From'] = le_From.transform([from_city])
x['To'] = le_To.transform([to_city])
x['is_holiday'] = is_holiday
x['avialine'] = le_avialine.transform([airline])
x['duration'] = duration
x['days_until_flight'] = days_until_flight
x['part_of_day'] = le_part_of_day.transform([part_of_day])

try:
    if st.button("Предсказать цену"):
        predict = model.predict(x)
        st.success(f"💰 Прогнозируемая цена: {int(predict[0])} тенге")
except:
    st.success('Выберите разные города')


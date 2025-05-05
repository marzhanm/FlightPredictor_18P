# FlightPredictor
Final project for Advanced Project Management course.

Introduction

Lots of people plan their travel through flight, and need a pragmatic and fast way to find a flight for a chosen date. Our team develops a flight prediction system that leverages machine learning to forecast flight and arrival times. The platform is fully developed, meaning that full front and back parts are written in Python. Streamlit library plays an essential role. The project consists of several models.


Problem Statement

Flight tickets vary in prices, influenced by several factors such as date of travel (weekend or holidays), demand and seasonality. Clients often struggle to find the best time to book flights at the lowest costs. Kazakhstan travelers risk overpaying without understanding price behaviour. This project addresses this problem by offering a prediction model with a flexible user interface that allows to foresee flight costs based on trends.

Objectives 

-To develop a software that uses user input to predict the cost of airline tickets.

-To train and evaluate machine learning models on historical flight fare data.

-To offer an intuitive and engaging user interface for querying predictions.

Technology Stack

Frontend & Backend: Streamlit. The project is fully written in Python.

ML training.

Installation Instructions

# 1. Clone the repository

git clone https://github.com/marzhanm/FlightPredictor_18P.git

# 2. Navigate into the project directory

cd FlightPredictor_18P

# 3. Install dependencies

pip install streamlit scikit-learn numpyl

# 4. Start the application

streamlit run app.py


Usage Guide

To run the project: streamlit run app.py in Terminal

Input travel details(date, origin and destination).

Click Predict Price to see the estimated tickets price. It must look like this:

![IMG_1546](https://github.com/user-attachments/assets/851ed3ee-907f-4a3c-bd99-6884ad333a2c)

Testing

It shows predicted price from different cities without any problem, also it doesn't show when user wants to flight from one city to another.

We tested cross-validation for the machine to train our model.

![photo_2025-05-05_19-48-21](https://github.com/user-attachments/assets/b758dc4a-521c-4503-b86c-7996c60b8007)

Tuning of hyperparameters. 

Limitations (Very Important!!!)

The code may not work on others’ computers because in code we mentioned one of coders filepath. In the perfect case, the path to files in 'src' and 'notebooks' packages needs to be changed to yours “/Users/username/Desktop/etc.”. 

References

Datasets in csv files based on flight information of Kazakhstani air companies (used to train the machine model).

Scikit-learn documentation: https://devdocs.io/scikit_learn/

Streamlit documentation: https://docs.streamlit.io/


Team Members:

Mutalova Marzhan, 220103354, 18P

Nurislam Galizhan, 220103263, 18P

Tuleubekova Togzhan, 220103197, 18P

Arim Askhatuly, 220103397, 18P

Azel Zhumagul, 220103265, 18P

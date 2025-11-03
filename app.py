import streamlit as st
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import urllib.request

st.set_page_config(page_title="🏠 Boston Real Estate Price Prediction", layout="centered")

st.title("🏠 Boston Real Estate Price Prediction")
st.markdown("### Predict median housing prices using neighborhood features")

MODEL_PATH = "boston_model.pkl"

@st.cache_resource
def load_boston_dataset():
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    return data, target

def train_and_save_model():
    X, y = load_boston_dataset()
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    pickle.dump(model, open(MODEL_PATH, "wb"))
    return model, score

if not os.path.exists(MODEL_PATH):
    model, acc = train_and_save_model()
else:
    model = pickle.load(open(MODEL_PATH, "rb"))
    acc = None

# Define feature names
feature_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

feature_descriptions = {
    "CRIM": "Per capita crime rate by town",
    "ZN": "Proportion of residential land zoned > 25,000 sqft",
    "INDUS": "Proportion of non-retail business acres",
    "CHAS": "Charles River dummy variable (1 = yes, 0 = no)",
    "NOX": "Nitric oxides concentration (parts per 10 million)",
    "RM": "Average number of rooms per dwelling",
    "AGE": "% of owner-occupied units built before 1940",
    "DIS": "Weighted distances to employment centers",
    "RAD": "Accessibility to radial highways index",
    "TAX": "Property tax rate per $10,000",
    "PTRATIO": "Pupil-teacher ratio by town",
    "B": "1000(Bk - 0.63)^2",
    "LSTAT": "% lower status of the population"
}

example_values = [0.1, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.90, 4.98]

st.markdown("#### Enter Feature Values:")
cols = st.columns(3)
input_values = []

for i, feature in enumerate(feature_names):
    col = cols[i % 3]
    input_val = col.number_input(
        f"{feature} ({feature_descriptions[feature]})",
        value=float(example_values[i]),
        step=0.01
    )
    input_values.append(input_val)

if st.button("Predict Price"):
    input_array = np.array(input_values).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f"🏡 Predicted Median House Price: **${prediction * 1000:.2f}**")

if acc is not None:
    st.caption(f"Model trained successfully. R² score: {acc:.2f}")
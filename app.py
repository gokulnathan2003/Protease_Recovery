import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import base64

# Set up the Streamlit app
st.set_page_config(page_title="Efficiency Predictor", layout="wide")

# Add background image using CSS
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        b64_str = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpg;base64,{b64_str}"

# Set background image
bg_image = get_base64_image("lab_image.jpg")
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{bg_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    h1, h2, h3, h4, h5, h6, p, label, span, div, .css-1cpxqw2, .css-10trblm, .stRadio label {{
        color: black !important;
        font-weight: bold !important;
    }}

    .stSlider .css-1cpxqw2, .stSlider label, .stSlider span, .stSlider div {{
        color: black !important;
        font-weight: bold !important;
    }}

    .stRadio label, .stRadio div {{
        color: black !important;
        font-weight: bold !important;
    }}
    .stButton button {{
        background-color: #4CAF50; /* Green */
        color: white;
        font-weight: bold;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# App title and image
st.markdown("<h1 style='text-align: center; font-size: 50px;'>🔬 Extraction Efficiency Predictor</h1>", unsafe_allow_html=True)


# Efficiency selection
eff_choice = st.radio("Select the Efficiency to Predict:", ["Efficiency 1", "Efficiency 2", "Efficiency 3"])

# Load and train model based on selection
def load_model_and_features(choice):
    if choice == "Efficiency 1":
        df = pd.read_csv("efficiency1.csv")
        X = df.drop(columns=["Extraction Efficiency 1", "Molar ratio", "Sample Ratio"])
        y = df["Extraction Efficiency 1"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_scaled, y)
        return model, scaler, X.columns, "efficiency1.csv"

    elif choice == "Efficiency 2":
        df = pd.read_csv("efficiency2.csv")
        X = df.drop(columns=["Extraction Efficiency 2", "Molar ratio", "Sample Ratio"])
        y = df["Extraction Efficiency 2"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_scaled, y)
        return model, scaler, X.columns, "efficiency2.csv"

    elif choice == "Efficiency 3":
        df = pd.read_csv("efficiency3.csv")
        df_clean = df.drop(columns=["Molar ratio", "Sample Ratio"])
        X = df_clean.drop(columns=["Extraction Efficiency 3"])
        y = df_clean["Extraction Efficiency 3"]
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X, y)
        return model, None, X.columns, "efficiency3.csv"

# Load model
model, scaler, feature_columns, data_file = load_model_and_features(eff_choice)

# Feature input section
st.markdown("### 🔧 Enter Input Features")
user_inputs = {}

# Load data for min/max
data_df = pd.read_csv(data_file)

for feature in feature_columns:
    if feature in data_df.columns:
        min_val = float(data_df[feature].min())
        max_val = float(data_df[feature].max())
    else:
        min_val = 0.0
        max_val = 100.0

    if min_val == max_val:
        st.warning(f"🔒 Feature `{feature}` has a constant value: {min_val}")
        user_inputs[feature] = min_val
    else:
        step_size = max((max_val - min_val) / 100, 0.01)
        mean_val = (min_val + max_val) / 2
        user_inputs[feature] = st.slider(
            f"{feature}",
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=step_size
        )

# Predict
if st.button("🚀 Predict"):
    input_df = pd.DataFrame([user_inputs])
    if scaler:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
    else:
        prediction = model.predict(input_df)

    st.success(f"✅ Predicted {eff_choice}: **{prediction[0]:.2f}**")
    st.markdown("#### 🔍 Input Summary")
    st.dataframe(input_df)

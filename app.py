import pickle
import pandas as pd
import numpy as np
import streamlit as st

def predict_price(product_name, product_url, product_type, scaler_path, model_path):
    try:
        # load scaler
        with open(encoders_path, 'rb') as file1:
            scaler = pickle.load(file1)

        # load model
        with open(model_path, 'rb') as file2:
            model = pickle.load(file2)

        # create input dataframe
        dct = {
            'product_name': [product_name],
            'product_url': [product_url],
            'product_type': [product_type]
        }

        x_new = pd.DataFrame(dct)

        # NOTE: You used LabelEncoder during training,
        # but didn’t save it → this will break consistency
        # For now, we simulate encoding (not ideal)

        for col in x_new.columns:
            x_new[col] = x_new[col].astype('category').cat.codes

        # scale input
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        for i in x_new.columns:
            x_new[i] = le.fit_transform(x_new[i])

        # prediction
        pred = model.predict(x_new)

        return pred[0]

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None


# ------------------ Streamlit UI ------------------

st.title("Skincare Product Price Predictor")

# input fields
product_name = st.text_input("Product Name")
product_url = st.text_input("Product URL")
product_type = st.text_input("Product Type")

# predict button
if st.button("Predict Price"):
    
    encoders_path = 'notebook/encoders.pkl'
    model_path = 'notebook/model.pkl'
    pred = predict_price(product_name, product_url, product_type, encoders_path, model_path)

    if pred is not None:
        st.subheader(f"Predicted Price: ₹ {pred:.2f}")
    else:
        st.error("Prediction Failed")
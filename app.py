import pickle
import pandas as pd
import numpy as np
import streamlit as st

def predict_price(product_name, product_url, product_type, scaler_path, model_path):
    try:
        # load scaler
        with open(scaler_path, 'rb') as file1:
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
        xnew_pre = scaler.transform(x_new)

        # prediction
        pred = model.predict(xnew_pre)

        return pred[0]

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None


# ------------------ Streamlit UI ------------------

st.title("Skincare Product Price Predictor")

# input fields
product_name = st.text_input("Product Name", "Face Cream")
product_url = st.text_input("Product URL", "example.com/product")
product_type = st.text_input("Product Type", "Moisturizer")

# predict button
if st.button("Predict Price"):
    
    scaler_path = 'scaler.pkl'
    model_path = 'model.pkl'
    pred = predict_price(product_name, product_url, product_type, scaler_path, model_path)

    if pred is not None:
        st.subheader(f"Predicted Price: ₹ {pred:.2f}")
    else:
        st.error("Prediction Failed")
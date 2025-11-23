import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('XGBmodel.pkl')

st.title("Success Predictor")

# Create inputs
user_input_1 = st.number_input("Funding Goal ($)")
user_input_2 = st.selectbox("Category", ["Tech", "Art", "Food"])
# ... add all your inputs

if st.button("Predict"):
    # Prepare data for model
    input_data = pd.DataFrame([[user_input_1, user_input_2]], columns=['Goal', 'Category'])
    
    # Get Prediction
    prediction = model.predict_proba(input_data)[0][1] # Probability of success
    st.metric("Chance of Success", f"{prediction:.2%}")

    # Logic to suggest improvement (simplified example)
    if user_input_1 > 10000:
        st.info("Tip: Lowering your goal below $10k usually increases success rates by 15%.")
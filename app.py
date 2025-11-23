import sklearn
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Page Config ---
st.set_page_config(page_title="Kickstarter Success Predictor", layout="wide")

# --- 2. Load Model & Preprocessor ---
@st.cache_resource
def load_assets():
    prep = None
    model = None
    try: 
        prep = joblib.load('preprocessor.pkl')
    except FileNotFoundError:
        st.error("⚠️ Critical Error: Preprocessor file not found.")
    try:
        model = joblib.load('XGBmodel.pkl')
    except FileNotFoundError:
        st.error("⚠️ Critical Error: Model files not found.")

    return prep, model

preprocessor, model = load_assets()

# --- 3. Extracted Data (From your CSV) ---
# Mapped from 'kickstarter_data_cleaned_revised.csv'
category_metrics = {
    'academic': 3.25, 
    'apps': 100.16, 
    'blues': 194.65, 
    'comedy': 29.67, 
    'experimental': 49.96, 
    'festivals': 36.12, 
    'flight': 162.39, 
    'gadgets': 379.41, 
    'hardware': 361.21, 
    'immersive': 54.75, 
    'makerspaces': 71.97, 
    'musical': 53.89, 
    'places': 5.95, 
    'plays': 43.02, 
    'robots': 210.86, 
    'shorts': 51.58, 
    'software': 87.18, 
    'sound': 351.55, 
    'spaces': 172.73, 
    'thrillers': 10.00, 
    'wearables': 422.07, 
    'web': 24.79, 
    'webseries': 11.00
}

valid_countries = sorted(['US', 'DE', 'GB', 'CA', 'AU', 'AT', 'CH', 'NZ', 'IT', 'MX', 'SE', 'ES', 'DK', 'NL', 'IE', 'NO', 'FR', 'BE', 'LU', 'SG', 'HK'])

# --- 4. Sidebar Inputs ---
st.sidebar.header("Campaign Details")

# Category Selection
default_category_index = sorted(category_metrics.keys()).index('apps')
cat_input = st.sidebar.selectbox("Category", options=sorted(category_metrics.keys()), index=default_category_index)
# Country Selection
default_country_index = valid_countries.index('US')
country_input = st.sidebar.selectbox("Country", options=valid_countries, index=default_country_index)

st.sidebar.subheader("Project Goals")
goal_input = st.sidebar.number_input("Goal Amount (USD)", min_value=1.0, value=5000.0, step=100.0)
duration_input = st.sidebar.slider("Campaign Duration (Days)", min_value=1, max_value=60, value=30)
pledge_input = st.sidebar.number_input("Avg Pledge per Backer ($)", min_value=1.0, value=15.0, step=5.0)

# Auto-lookup for the hidden feature
avg_backers_val = category_metrics[cat_input]
st.sidebar.info(f"ℹ️ Historical Avg Backers for '{cat_input}': **{avg_backers_val:.0f}**")

# Construct the input dictionary exactly as the model expects
input_data = {
    'country': country_input,
    'category': cat_input,
    'goal_usd_calc': goal_input,
    'campaign_duration_day': duration_input,
    'pledge_per_backer': pledge_input,
    'avg_backers_per_category': avg_backers_val  # Automatically filled
}

# --- 5. Main Dashboard Logic ---
st.title("Kickstarter Campaign Success Predictor ")
st.write("Adjust the inputs in the sidebar to see how they affect your success rate.")

if st.button("Predict & Optimize"):
    if preprocessor and model:
        
        # --- A. PREDICTION ---
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Transform data
        processed_data = preprocessor.transform(input_df)
        
        # Get Probability
        current_prob = model.predict_proba(processed_data)[0][1]
        
        # Display Big Metric
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Success Probability", f"{current_prob:.1%}")
        
        with col2:
            if current_prob > 0.75:
                st.success("✅ Strong Campaign! You are on track.")
            elif current_prob > 0.50:
                st.warning("⚠️ Moderate Risk. Consider the suggestions below.")
            else:
                st.error("❌ High Risk. Adjustments strongly recommended.")

        st.divider()

        # --- B. OPTIMIZATION ENGINE ---
        st.subheader("Parameter Recommendations")
        
        # We run 'What-If' scenarios to find the best tweak
        adjustments = [
             ('Goal', 'Increase Goal by 5%', 'goal_usd_calc', input_data['goal_usd_calc'] * 1.05),
            ('Goal', 'Increase Goal by 10%', 'goal_usd_calc', input_data['goal_usd_calc'] * 1.10),
            ('Goal', 'Reduce Goal by 10%', 'goal_usd_calc', input_data['goal_usd_calc'] * 0.90),
            ('Goal', 'Reduce Goal by 20%', 'goal_usd_calc', input_data['goal_usd_calc'] * 0.80),
            ('Duration', 'Increase Duration by 5 days', 'campaign_duration_day', max(7, input_data['campaign_duration_day'] + 5)),
            ('Duration', 'Increase Duration by 10 days', 'campaign_duration_day', max(7, input_data['campaign_duration_day'] + 10)),
            ('Duration', 'Decrease Duration by 5 days', 'campaign_duration_day', max(7, input_data['campaign_duration_day'] - 5)),
            ('Duration', 'Decrease Duration by 10 days', 'campaign_duration_day', max(7, input_data['campaign_duration_day'] - 10)),
             ('Pledge', 'Decrease Avg Pledge ($5)', 'pledge_per_backer', input_data['pledge_per_backer'] - 5),
            ('Pledge', 'Decrease Avg Pledge ($10)', 'pledge_per_backer', input_data['pledge_per_backer'] - 10),
            ('Pledge', 'Increase Avg Pledge ($5)', 'pledge_per_backer', input_data['pledge_per_backer'] + 5),
            ('Pledge', 'Increase Avg Pledge ($10)', 'pledge_per_backer', input_data['pledge_per_backer'] + 10),
        ]

        results = []

        for adj_type, adj_name, col_name, new_val in adjustments:
            # Create temp data copy
            temp_data = input_data.copy()
            temp_data[col_name] = new_val
            
            # Predict with adjustment
            temp_df = pd.DataFrame([temp_data])
            temp_processed = preprocessor.transform(temp_df)
            new_prob = model.predict_proba(temp_processed)[0][1]
            
            impact = new_prob - current_prob
            
            results.append({
                'Type': adj_type,
                'Suggestion': adj_name,
                'New Probability': new_prob,
                'Impact': impact
            })

        # Convert to DataFrame & Sort
        results_df = pd.DataFrame(results).sort_values(by='Impact', ascending=False)
        best_option = results_df.iloc[0]

        # Display Best Recommendation
        if best_option['Impact'] > 0.005:
            st.info(f"**Top Tip:** Try to **{best_option['Suggestion']}**.")
            st.write(f"This single change could boost your chance by **+{best_option['Impact']:.1%}** (up to {best_option['New Probability']:.1%}).")
        else:
            st.info("Minor adjustments to Goal or Duration won't change much. Your category/country selection is the main driver here.")

        # Show Comparison Table
        st.write("### Simulation Results")
        display_df = results_df.copy()
        display_df['New Probability'] = display_df['New Probability'].map('{:.1%}'.format)
        display_df['Impact'] = display_df['Impact'].map('{:+.1%}'.format)
        st.dataframe(display_df[['Type', 'Suggestion', 'New Probability', 'Impact']], use_container_width=True, hide_index=True)

    else:
        st.warning("Models not loaded. Please check your .pkl files.")
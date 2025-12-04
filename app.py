# =============================================================================
# IMPORTS & SETUP
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# The model (a Scikit-learn Pipeline) is loaded using Joblib for efficiency, and the 
# single-row DataFrame is essential to structure the user input with correct column 
# names and order, which is the exact format the Pipeline requires for prediction.

data_frame = pd.read_csv("final_cleaned_data.csv")
model = joblib.load("model.pkl")

# Applies the inverse of the log(1+x) transformation (np.expm1) to convert 
# a prediction from log-space back into the real currency scale (Euros).

def inverse_log_transform(y_pred_log):
    return np.expm1(y_pred_log)

# true price is expected to fall within a range 
# that is statistically +/- 0.1 units wide in the model's domain
LOG_MAE = 0.1

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Belgian Immo Price Predictor</h1>", unsafe_allow_html=True)
st.sidebar.header("Insert your features:")
image = Image.open("Immo_Eliza_Predictor.png")
st.image(image)

# Load the data for the selectbox options
province_opts = data_frame["province"].astype(str).unique()
type_opts = data_frame["type"].astype(str).unique()

# Making sure that missing values are assigned to 'Unknown' category in the drop down list.
building_status_processed = data_frame["state_of_building"].copy()
building_status_processed.fillna('Unknown', inplace=True)
building_status_opts = building_status_processed.astype(str).unique()

# Collects user-selected property features from the Streamlit sidebar widgets (selectboxes, sliders, and checkboxes). 
# The inputs are compiled into a single-row pandas DataFrame structured precisely to serve as input for the trained 
# machine learning model.Returns a pd.DataFrame - A single-row DataFrame matching the model's required input format.

def user_report():
    # Collect user input for each feature
    province = st.sidebar.selectbox("Select the Province", sorted(province_opts))
    type = st.sidebar.selectbox("Select the type", sorted(type_opts))
    type_filtered_data = data_frame[data_frame['type'] == type]

    # This code ensures that 'Apartment' is being removed from subtype category adn only stays in the type category.
    unique_subtypes = type_filtered_data['subtype'].unique()
    subtypes_to_exclude = {type} # Start with the selected type itself
    if type != 'Apartment':
        subtypes_to_exclude.add('Apartment')

    filtered_subtypes = [s for s in unique_subtypes if s not in subtypes_to_exclude]
    subtype = st.sidebar.selectbox("Select the subtype", sorted(filtered_subtypes))

    state_of_building = st.sidebar.selectbox("Select the building status", sorted(building_status_opts))

    # Numerical Inputs for Slider bar
    number_of_bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 20, 1)
    number_facades = st.sidebar.slider("Number of Facades", 1, 4, 1)
    living_area_m2 = st.sidebar.slider("Living Area (m²)", 10, 500, 10)
    terrace_area_m2 = st.sidebar.slider("Terrace Area (m²)", 0, 500, 1)

    # Binary Inputs (Checkboxes) 
    swimming_pool = 1 if st.sidebar.checkbox("Has Swimming Pool") else 0
    open_fire = 1 if st.sidebar.checkbox("Has Open Fireplace") else 0
    terrace = 1 if st.sidebar.checkbox("Has Terrace") else 0
    equiped_kitchen = 1 if st.sidebar.checkbox("Has Equiped Kitchen", value=True) else 0
    furnished = 1 if st.sidebar.checkbox("Is Furnished") else 0
    garden = 1 if st.sidebar.checkbox("Has Garden") else 0

    user_input_data = {
        "province": province,
        "type": type,
        "subtype": subtype,
        "state_of_building": state_of_building,

        # Numeric/Count features
        "number_of_bedrooms": number_of_bedrooms,
        "number_facades": number_facades,

        # Numeric Area features 
        "living_area (m²)": living_area_m2,
        "terrace_area (m²)": terrace_area_m2,

        # Binary features 
        "swimming_pool (yes:1, no:0)": swimming_pool, 
        "open_fire (yes:1, no:0)": open_fire, 
        "terrace (yes:1, no:0)": terrace,
        "equiped_kitchen (yes:1, no:0)": equiped_kitchen, 
        "furnished (yes:1, no:0)": furnished, 
        "garden (yes:1, no:0)": garden
    }
    return pd.DataFrame(user_input_data, index=[0])

# Collect user data
user_data = user_report()

# The model predicts the logarithm of the price to stabilize the distribution. The Log MAE (Mean Absolute Error: 0.1) 
# is used to calculate the confidence interval in log-space. All three values (prediction and bounds) 
# are then converted back to Euros using the inverse function (np.expm1) and formatted for display.

# NEW BUTTON IMPLEMENTATION
STYLISH_BUTTON_CSS = """
<style>
/* Centering Fix: Ensures the button is centered within its column */
div.stButton {
    width: fit-content; 
    margin: 0 auto;
}
div.stButton > button {
    background: linear-gradient(to right, #2196F3, #00BCD4);
    color: white; 
    padding: 14px 35px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 25px;
    border: none;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.4s ease;  
    /* === SHAPE/SIZE FIXES === */
    white-space: nowrap; /* Prevents text from wrapping onto multiple lines */
    min-width: 250px;    /* Ensures the button is wide enough for the text */
    /* ======================= */
}
div.stButton > button:hover {
    background: linear-gradient(to left, #2196F3, #00BCD4);
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
}
</style>"""

# --- Centering the Button ---
col1, col2, col3 , col4, col5, col6, col7, col8, col9 = st.columns(9)

with col1:
    pass
with col2:
    pass
with col5:
    pass
with col3:
    pass
with col6:
    pass
with col7:
    pass
with col8:
    pass
with col9:
    pass
with col4:
    prediction_button = st.button("Predict The Price")

if prediction_button:
    # ... Your prediction and output logic ...
    pass

st.markdown(STYLISH_BUTTON_CSS, unsafe_allow_html=True)

if prediction_button:
    try:
        price_log = model.predict(user_data)[0] 
        lower_bound_log = price_log - LOG_MAE
        upper_bound_log = price_log + LOG_MAE
        
        predicted_price_euros = inverse_log_transform(price_log)
        lower_bound_euros = inverse_log_transform(lower_bound_log)
        upper_bound_euros = inverse_log_transform(upper_bound_log)

        final_price = max(0, predicted_price_euros)
        final_lower = max(0, lower_bound_euros)
        final_upper = max(0, upper_bound_euros)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Estimated Property Price:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: #1f77b4;'>€{np.round(final_price):,.0f}</h2>", unsafe_allow_html=True) 
        
        st.markdown(f"<p style='text-align: center; font-size: small;'>Estimated Price Range in Euros (€):</p>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>€{np.round(final_lower):,.0f} - €{np.round(final_upper):,.0f}</h4>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Renders a collapsible Streamlit section to display a clear, formatted summary of all the property features the user 
# selected in the sidebar. This ensures transparency and allows the user to verify the inputs used for the prediction.

    with st.expander("View Selected Features"):
        data_dict = user_data.iloc[0].to_dict() # Convert the single-row DataFrame back to a dictionary for display
        
    
        for key, value in data_dict.items(): # Iterate and display each feature clearly
            display_key = key.replace(" (m²)", "").replace(" (yes:1, no:0)", "").replace("_", " ").title()
            
            display_value = value
            if value in [1, 0] and key.endswith("(yes:1, no:0)"): # Convert binary 1/0 values to Yes/No strings
                display_value = "Yes" if value == 1 else "No"

            st.markdown(f"- **{display_key}:** `{display_value}`")
  
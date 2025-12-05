import streamlit as st
import numpy as np
import joblib

# Load the deploy model (trained on 11 features)
model = joblib.load("predictive_maintenance_model.pkl")

st.title("🔧 Predictive Maintenance System")
st.write("Enter machine sensor values to check if failure is likely.")

# ===== User Inputs =====
air_temp = st.number_input("Air Temperature (K)", value=300.0)
process_temp = st.number_input("Process Temperature (K)", value=310.0)
speed = st.number_input("Rotational Speed (rpm)", value=1500)
torque = st.number_input("Torque (Nm)", value=40.0)
tool_wear = st.number_input("Tool Wear (minutes)", value=50)
machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

# Encode machine type (same mapping as LabelEncoder used earlier)
type_mapping = {"L": 1, "M": 2, "H": 0}
machine_type_num = type_mapping[machine_type]

# Assume no specific failure modes active (all zeros)
twf = 0  # Tool wear failure
hdf = 0  # Heat dissipation failure
pwf = 0  # Power failure
osf = 0  # Overstrain failure
rnf = 0  # Random failure

# ===== Prepare input in EXACT order used for training =====
# ['Type', 'Air temperature [K]', 'Process temperature [K]',
#  'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
#  'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

input_data = np.array([[
    machine_type_num,
    air_temp,
    process_temp,
    speed,
    torque,
    tool_wear,
    twf,
    hdf,
    pwf,
    osf,
    rnf
]])

# ===== Prediction =====
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob_failure = model.predict_proba(input_data)[0][1]

    if prediction == 0:
        st.success(f"✅ Machine is operating normally. (Failure probability: {prob_failure:.2%})")
    else:
        st.error(f"🚨 Warning: Machine failure likely! (Failure probability: {prob_failure:.2%})")

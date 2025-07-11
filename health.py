import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="Health Risk Predictor", layout="centered")

# Sidebar
st.sidebar.title("Health Risk Predictor")
st.sidebar.markdown("Built with ❤️ by Anshuman Sinha")

# Title
st.title("🩺 Health Risk Prediction Platform")
st.markdown("This simple health screening tool uses logistic regression to predict general health risk based on input vitals and metrics.")

# Expanders for health guidelines
with st.expander("📊 Ideal BMI Ranges by Age"):
    st.write("""
    - 18–24 yrs: 19–24
    - 25–34 yrs: 20–25
    - 35–44 yrs: 21–26
    - 45–54 yrs: 22–27
    - 55–64 yrs: 23–28
    - 65+ yrs: 24–29
    """)

with st.expander("🍬 Glucose Level Guidelines"):
    st.write("""
    - Normal: 70–99 mg/dL
    - Prediabetes: 100–125 mg/dL
    - Diabetes: 126+ mg/dL
    """)

with st.expander("🧠 Blood Pressure & Vitals"):
    st.write("""
    - Normal BP: 120/80 mmHg
    - Cholesterol Normal: < 200 mg/dL
    - Heart Rate Normal: 60–100 bpm
    """)

# Input fields
st.subheader("📥 Enter Patient Details")

age = st.slider("Age", 10, 100, 30)
gender = st.radio("Gender", ["Male", "Female"])
weight = st.number_input("Weight (kg)", 30.0, 150.0, 65.0)
height = st.number_input("Height (cm)", 120.0, 210.0, 170.0)
bmi = round(weight / ((height / 100) ** 2), 2)

glucose = st.slider("Glucose Level (mg/dL)", 50, 300, 90)
bp_systolic = st.slider("Systolic BP (mmHg)", 90, 200, 120)
bp_diastolic = st.slider("Diastolic BP (mmHg)", 60, 130, 80)
cholesterol = st.slider("Cholesterol Level (mg/dL)", 100, 400, 180)
heart_rate = st.slider("Heart Rate (bpm)", 40, 150, 75)
physical_activity = st.slider("Physical Activity (hours/week)", 0, 15, 3)
smoking = st.radio("Do you smoke?", ["No", "Yes"])
alcohol = st.radio("Do you consume alcohol?", ["No", "Yes"])
sleep_hours = st.slider("Sleep (hours/day)", 0, 12, 7)

# Binary Encoding
gender_encoded = 1 if gender == "Male" else 0
smoking_encoded = 1 if smoking == "Yes" else 0
alcohol_encoded = 1 if alcohol == "Yes" else 0

# Feature Vector
input_features = pd.DataFrame([[
    age, gender_encoded, weight, height, bmi, glucose, bp_systolic,
    bp_diastolic, cholesterol, heart_rate, physical_activity,
    smoking_encoded, alcohol_encoded, sleep_hours
]], columns=[
    'Age', 'Gender', 'Weight', 'Height', 'BMI', 'Glucose', 'SystolicBP', 'DiastolicBP',
    'Cholesterol', 'HeartRate', 'PhysicalActivity', 'Smoking', 'Alcohol', 'SleepHours'
])

# Dummy Dataset for Training
np.random.seed(42)
X_train = pd.DataFrame({
    'Age': np.random.randint(18, 80, 500),
    'Gender': np.random.randint(0, 2, 500),
    'Weight': np.random.randint(45, 110, 500),
    'Height': np.random.randint(150, 190, 500),
    'BMI': np.random.uniform(18, 35, 500),
    'Glucose': np.random.randint(70, 200, 500),
    'SystolicBP': np.random.randint(100, 180, 500),
    'DiastolicBP': np.random.randint(60, 110, 500),
    'Cholesterol': np.random.randint(150, 300, 500),
    'HeartRate': np.random.randint(55, 100, 500),
    'PhysicalActivity': np.random.randint(0, 10, 500),
    'Smoking': np.random.randint(0, 2, 500),
    'Alcohol': np.random.randint(0, 2, 500),
    'SleepHours': np.random.randint(4, 10, 500)
})
y_train = np.random.randint(0, 2, 500)

# Model Training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
model = LogisticRegression(max_iter=500)
model.fit(X_scaled, y_train)

# Prediction
input_scaled = scaler.transform(input_features)
risk_prob = model.predict_proba(input_scaled)[0][1]
prediction = model.predict(input_scaled)[0]

st.subheader("🔍 Risk Prediction Result")
st.markdown(f"**🧮 BMI:** `{bmi}`")
st.success(f"✅ Health Risk Probability: **{risk_prob * 100:.2f}%**")
if prediction == 1:
    st.error("⚠️ High Risk Detected! Please consult a doctor.")
else:
    st.success("🟢 Low Risk Detected. Keep maintaining your health!")

# Visualization
st.subheader("📈 Feature Importance")
coefficients = model.coef_[0]
features = X_train.columns
feat_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
}).sort_values(by="Coefficient", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=feat_importance, x='Coefficient', y='Feature', ax=ax, palette="Spectral", legend=False)
st.pyplot(fig)

# PDF Generation
def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Health Risk Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    for col in input_features.columns:
        val = input_features.iloc[0][col]
        pdf.cell(200, 10, txt=f"{col}: {val}", ln=True)

    pdf.cell(200, 10, txt=f"Risk Probability: {risk_prob * 100:.2f}%", ln=True)
    status = "High Risk" if prediction == 1 else "Low Risk"
    pdf.cell(200, 10, txt=f"Prediction: {status}", ln=True)

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

if st.button("📄 Generate Health Report (PDF)"):
    st.success("Report Generated Successfully!")
    pdf_file = generate_pdf()
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_file,
        file_name="health_report.pdf",
        mime="application/pdf"
    )

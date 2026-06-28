
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

model_path = hf_hub_download(
    repo_id="vidyaa2026/tourism-package-model",
    filename="best_tourism_model.joblib"
)

model = joblib.load(model_path)


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

st.markdown("""
<style>

/* Main page */
.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
    max-width:1200px;
}

/* Title */
h1{
    color:#0E6BA8;
    text-align:center;
    font-weight:700;
}

/* Buttons */
.stButton > button{
    width:100%;
    height:55px;
    border-radius:10px;
    font-size:18px;
    font-weight:600;
    background-color:#0E6BA8;
    color:white;
    border:none;
}

.stButton > button:hover{
    background-color:#09527E;
    color:white;
}

/* Metric Cards */
div[data-testid="stMetric"]{
    background:#F8F9FA;
    padding:15px;
    border-radius:12px;
    border:1px solid #EAEAEA;
}

/* Footer */
.footer{
    text-align:center;
    color:grey;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Logo
# ---------------------------------------------------------

logo_left, logo_mid, logo_right = st.columns([1,2,1])

with logo_mid:
    st.image(
        "https://i.postimg.cc/4NvPmtKX/image-82a08e46.png",
        width=300
    )

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("✈️ Tourism Package Purchase Prediction")

st.info("""
👋 **Welcome!**

Complete the customer information below and click **🎯 Predict Purchase Probability** to estimate the likelihood that a customer will purchase a tourism package.
""")


# ---------------------------------------------------------
# User Inputs
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=35
    )

    TypeofContact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

    CityTier = st.selectbox(
        "City Tier",
        [1,2,3]
    )

    DurationOfPitch = st.number_input(
        "Duration of Pitch",
        min_value=1,
        max_value=100,
        value=15
    )

    Occupation = st.selectbox(
        "Occupation",
        ['Salaried','Small Business','Large Business','Free Lancer']
    )

    Gender = st.selectbox(
        "Gender",
        ['Male','Female']
    )

    NumberOfPersonVisiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=10,
        value=2
    )

    NumberOfFollowups = st.number_input(
        "Number of Followups",
        min_value=0,
        max_value=10,
        value=3
    )

    ProductPitched = st.selectbox(
        "Product Pitched",
        ['Basic','Standard','Deluxe','Super Deluxe','King']
    )

with col2:

    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star",
        [3,4,5]
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        ['Single','Married','Divorced']
    )

    NumberOfTrips = st.number_input(
        "Number of Trips",
        min_value=0,
        max_value=30,
        value=3
    )

    Passport = st.selectbox(
        "Passport Available",
        ["No","Yes"]
    )

    Passport = 1 if Passport == "Yes" else 0

    PitchSatisfactionScore = st.slider(
        "Pitch Satisfaction Score",
        1,
        5,
        3
    )

    OwnCar = st.selectbox(
        "Own Car",
        ["No","Yes"]
    )

    OwnCar = 1 if OwnCar == "Yes" else 0

    NumberOfChildrenVisiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=5,
        value=1
    )

    Designation = st.selectbox(
        "Designation",
        ['AVP','VP','Manager','Senior Manager','Executive']
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=500000,
        value=25000
    )
# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------

input_data = pd.DataFrame([{

'Age':Age,
'TypeofContact':TypeofContact,
'CityTier':CityTier,
'DurationOfPitch':DurationOfPitch,
'Occupation':Occupation,
'Gender':Gender,
'NumberOfPersonVisiting':NumberOfPersonVisiting,
'NumberOfFollowups':NumberOfFollowups,
'ProductPitched':ProductPitched,
'PreferredPropertyStar':PreferredPropertyStar,
'MaritalStatus':MaritalStatus,
'NumberOfTrips':NumberOfTrips,
'Passport':Passport,
'PitchSatisfactionScore':PitchSatisfactionScore,
'OwnCar':OwnCar,
'NumberOfChildrenVisiting':NumberOfChildrenVisiting,
'Designation':Designation,
'MonthlyIncome':MonthlyIncome

}])

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

# Reserve space for results (prevents page jumping)
result = st.empty()

if st.button(
    "🎯 Predict Purchase Probability",
    use_container_width=True,
    type="primary"
):

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Prediction probability
    probability = model.predict_proba(input_data)[0][1]

    with result.container():

        st.divider()

        st.subheader("🎯 Prediction Result")

        if prediction == 1:
            st.success("✅ Customer is likely to purchase the tourism package.")
        else:
            st.error("❌ Customer is unlikely to purchase the tourism package.")

        # Probability
        st.metric(
            label="📊 Purchase Probability",
            value=f"{probability:.2%}"
        )

        st.progress(float(probability))

        # Purchase Potential
        if probability >= 0.80:
            st.success("🔥 High Purchase Potential")
        elif probability >= 0.50:
            st.info("⭐ Moderate Purchase Potential")
        else:
            st.warning("⚠️ Low Purchase Potential")

        st.divider()

        # Business Recommendation
        st.subheader("📌 Business Recommendation")

        if prediction == 1:

            st.markdown("""
**Recommended Actions**

- ✔ Prioritize this customer for immediate follow-up.
- ✔ Offer premium or customized travel packages.
- ✔ Provide limited-time promotional offers.
- ✔ Assign a senior travel consultant.
- ✔ Recommend loyalty or referral programs.
""")

        else:

            st.markdown("""
**Recommended Actions**

- • Continue nurturing the customer.
- • Share promotional discounts and seasonal offers.
- • Schedule another follow-up after a few weeks.
- • Recommend entry-level travel packages.
- • Include the customer in future marketing campaigns.
""")

        st.divider()

        # Model Information
        st.subheader("🤖 Model Information")

        st.info("""
**Model:** XGBoost pipeline with OneHotEncoder preprocessing loaded from Hugging Face.

**Deployment:** Hugging Face Spaces

**Purpose:** Predict whether a customer is likely to purchase a tourism package based on demographic, travel and financial information.
""")

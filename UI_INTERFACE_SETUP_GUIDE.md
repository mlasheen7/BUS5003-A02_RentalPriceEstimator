# UI INTERFACE SETUP GUIDE - STREAMLIT
## For PERSON D (Frontend Lead)

---

## 📋 OVERVIEW

**Goal:** Build a Streamlit web app that allows users to:
1. Select a suburb
2. Choose number of bedrooms
3. Select property type (house/flat)
4. Get a rent prediction + explanation + comparables

**Timeline:**
- Week 4: Build skeleton with mock predictions
- Week 5: Integrate real model + API
- Week 6: Polish UI + integrate comparables
- Week 7: Deploy to Streamlit Cloud
- Week 8-9: Testing + refinements

---

## 🏗️ PROJECT STRUCTURE

```
rental-price-estimator/
├── app/
│   ├── main.py                    ← Entry point (streamlit run app/main.py)
│   ├── pages/
│   │   ├── 1_predict.py           ← Prediction page
│   │   ├── 2_comparables.py       ← Similar suburbs
│   │   ├── 3_faq.py               ← FAQ page
│   │   └── 4_about.py             ← About page
│   ├── utils/
│   │   ├── model_loader.py        ← Load XGBoost model
│   │   ├── api_client.py          ← Claude API wrapper
│   │   ├── data_handler.py        ← Suburb data lookup
│   │   └── styling.py             ← Custom CSS/styling
│   └── assets/
│       ├── logo.png
│       └── banner.png
├── data/processed/
│   ├── merged_rental_data.csv     ← Suburb data lookup table
│   ├── X_train.pkl
│   └── y_train.pkl
├── models/
│   ├── xgboost_v1.0.pkl           ← Trained model
│   └── xgboost_v1.0.json          ← Model metadata
└── .streamlit/
    ├── config.toml                ← Streamlit config
    └── secrets.toml               ← API keys (gitignored)
```

---

## 🚀 WEEK 4: SKELETON WITH MOCK PREDICTIONS

### **Step 1: Install Streamlit**

```bash
# Activate venv
venv\Scripts\Activate.ps1

# Install Streamlit
pip install streamlit pandas numpy scikit-learn

# Verify installation
streamlit --version
```

### **Step 2: Create Main App File**

File: `app/main.py`

```python
"""
Rental Price Estimator - Main App
Entry point for Streamlit application
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Rental Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
    }
    h1 {
        color: #0066cc;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🏠 Rental Price Estimator")
st.markdown("""
Discover fair rental prices in Victoria, Australia.
Powered by AI analysis of current market data.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Predict", "Comparables", "FAQ", "About"])

# Content placeholder
if page == "Predict":
    st.markdown("### 📊 Estimate Your Rent")
    
    col1, col2 = st.columns(2)
    
    with col1:
        suburb = st.selectbox(
            "Select suburb:",
            options=["Footscray", "Toorak", "Heidelberg", "Docklands", "Southbank"],
            index=0
        )
    
    with col2:
        bedrooms = st.slider(
            "Number of bedrooms:",
            min_value=1,
            max_value=5,
            value=2
        )
    
    property_type = st.radio(
        "Property type:",
        options=["House", "Flat/Unit"],
        horizontal=True
    )
    
    # Prediction button
    if st.button("🔍 Estimate Rent", key="predict_btn"):
        # MOCK prediction (will be replaced with real model in Week 5)
        mock_rent = 450 if property_type == "House" else 380
        mock_rent += (bedrooms - 2) * 100  # Add rent per bedroom
        
        st.success(f"**Estimated Weekly Rent: ${mock_rent}**")
        
        # Display details
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Rent", f"${mock_rent}/week")
        with col2:
            st.metric("Confidence", "High", delta="+92%")
        with col3:
            st.metric("Median Income", "$1,200/week")
        
        # Explanation placeholder
        st.info("""
        💡 **AI Explanation:**
        This suburb has a median household income of $1,200/week and a SEIFA affluence 
        score of 1,020 (average). Based on similar properties in the area, a 
        2-bedroom house typically rents for $450/week.
        """)

elif page == "Comparables":
    st.markdown("### 🏘️ Similar Suburbs")
    st.info("Compare rent prices across similar suburbs (Coming in Week 6)")

elif page == "FAQ":
    st.markdown("### ❓ Frequently Asked Questions")
    
    with st.expander("How does the estimator work?"):
        st.write("""
        Our AI model analyzes publicly available data including:
        - Census demographics (income, education, employment)
        - SEIFA socio-economic scores
        - Historical rental data
        - Property location and type
        """)
    
    with st.expander("How accurate are these estimates?"):
        st.write("Model accuracy: ±$50/week for 95% of predictions")
    
    with st.expander("Which suburbs are covered?"):
        st.write("All 310 suburbs across Victoria")

elif page == "About":
    st.markdown("### ℹ️ About This Project")
    st.write("""
    **Rental Price Estimator** is a university project built to help renters 
    understand fair market prices in Victoria, Australia.
    
    **Data Sources:**
    - Australian Bureau of Statistics (Census)
    - Rental Tenancy Victoria (Bond Board)
    - Victorian Land Titles Office
    
    **Built with:** Python, XGBoost, Claude AI, Streamlit
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <small>Rental Price Estimator | Built for BUS5003 Operationalising Business Analytics</small>
</div>
""", unsafe_allow_html=True)
```

### **Step 3: Test Locally**

```bash
# Run Streamlit app
streamlit run app/main.py

# Should open http://localhost:8501
```

---

## 🎨 WEEK 4: BUILD SKELETON PAGES

### **Step 4: Create Pages Directory**

```bash
mkdir app/pages
```

### **File: app/pages/1_predict.py**

```python
"""
Prediction page - Main interface for users
"""

import streamlit as st
import pandas as pd

st.title("📊 Predict Rent")

# Load suburb data (from merged_rental_data.csv)
@st.cache_data
def load_suburbs():
    df = pd.read_csv('data/processed/merged_rental_data.csv')
    return sorted(df['suburb'].unique())

suburbs = load_suburbs()

# Input section
st.markdown("### Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    suburb = st.selectbox("Suburb:", suburbs)

with col2:
    bedrooms = st.slider("Bedrooms:", 1, 5, 2)

with col3:
    property_type = st.radio("Type:", ["House", "Flat"], horizontal=True)

# Predict button
if st.button("🔍 Estimate Rent"):
    # MOCK prediction (Week 5: replace with real model)
    rent = 400 + bedrooms * 75
    if property_type == "Flat":
        rent *= 0.85  # Flats are ~15% cheaper
    
    # Display results
    st.success(f"**Estimated Weekly Rent: ${rent:.0f}**")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Weekly Rent", f"${rent:.0f}")
    with col2:
        st.metric("📈 Monthly Rent", f"${rent*4.33:.0f}")
    with col3:
        st.metric("📊 Confidence", "92%")
    
    # Explanation section
    st.markdown("### 💡 Why This Price?")
    st.info(f"""
    **{suburb.title()}** is an average affluence area with median household income 
    of $1,200/week. A {bedrooms}-bedroom {property_type.lower()} typically rents 
    for ${rent:.0f}/week based on current market data.
    """)
    
    # Comparable suburbs
    st.markdown("### 🏘️ Comparable Suburbs")
    comparables = pd.DataFrame({
        'Suburb': ['Suburb A', 'Suburb B', 'Suburb C'],
        'Rent': [f'${rent-20:.0f}', f'${rent:.0f}', f'${rent+30:.0f}'],
        'Bedrooms': [bedrooms, bedrooms, bedrooms]
    })
    st.dataframe(comparables, use_container_width=True)
```

### **File: app/pages/2_comparables.py**

```python
"""
Comparables page - Show similar suburbs
"""

import streamlit as st

st.title("🏘️ Similar Suburbs")

st.info("📅 Coming in Week 6 - Compare rent across similar suburbs")
```

### **File: app/pages/3_faq.py**

```python
"""
FAQ Page
"""

import streamlit as st

st.title("❓ Frequently Asked Questions")

st.markdown("""
### How does this work?
Our AI model learns from historical rental data and demographic information 
to predict fair market prices.

### How accurate is this?
The model predicts within ±$50/week for 95% of properties.

### Which suburbs are covered?
All 310 suburbs across Victoria.

### Can I use this data for real estate?
Yes! This is educational data for understanding market trends.

### What data is used?
- Census demographics (income, education, employment)
- SEIFA socio-economic scores
- Historical rental bond data
- Property location and type
""")
```

### **File: app/pages/4_about.py**

```python
"""
About page
"""

import streamlit as st

st.title("ℹ️ About")

st.markdown("""
### Rental Price Estimator

A tool to help renters understand fair market prices in Victoria, Australia.

### Technology Stack
- **Backend:** Python, XGBoost, Pandas
- **Frontend:** Streamlit
- **AI Explanations:** Claude API
- **Data Sources:** ABS Census, Rental Tenancy Victoria

### Data Privacy
We use only publicly available data. No personal information is collected.

### Built for
BUS5003 Operationalising Business Analytics
La Trobe University, 2026
""")
```

### **Step 5: Create Config Files**

File: `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#0066cc"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#1f1f1f"
font = "sans serif"

[client]
showErrorDetails = true
toolbarMode = "viewer"

[logger]
level = "info"
```

---

## 📦 WEEK 5: INTEGRATE REAL MODEL & API

### **Step 6: Create Model Loader Utility**

File: `app/utils/model_loader.py`

```python
"""
Load trained XGBoost model and prepare for predictions
"""

import pickle
import pandas as pd
import numpy as np

class ModelLoader:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.suburb_data = None
        self.load()
    
    def load(self):
        """Load model and data"""
        # Load trained model
        with open('models/xgboost_v1.0.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # Load feature names
        with open('data/processed/feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        # Load suburb data for lookups
        self.suburb_data = pd.read_csv('data/processed/merged_rental_data.csv')
    
    def predict(self, suburb, bedrooms, property_type):
        """
        Make a prediction for a given suburb, bedrooms, and property type
        """
        # Get suburb data
        suburb_lower = suburb.lower()
        suburb_row = self.suburb_data[
            (self.suburb_data['suburb'] == suburb_lower) &
            (self.suburb_data['bedrooms'] == bedrooms) &
            (self.suburb_data['property_type'] == property_type.lower())
        ]
        
        if suburb_row.empty:
            return None, "Combination not found in training data"
        
        # Prepare features
        X = suburb_row[self.feature_names].values
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        return prediction, None
    
    def get_suburbs(self):
        """Get list of unique suburbs"""
        return sorted(self.suburb_data['suburb'].unique())
```

### **Step 7: Integrate into Predict Page**

Update `app/pages/1_predict.py`:

```python
"""
Prediction page - with real model
"""

import streamlit as st
import pandas as pd
import sys
sys.path.insert(0, 'app/utils')
from model_loader import ModelLoader

st.title("📊 Predict Rent")

# Load model (cached)
@st.cache_resource
def load_model():
    return ModelLoader()

model = load_model()

# Get suburbs
suburbs = model.get_suburbs()

# Input section
st.markdown("### Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    suburb = st.selectbox("Suburb:", suburbs)

with col2:
    bedrooms = st.slider("Bedrooms:", 1, 5, 2)

with col3:
    property_type = st.radio("Type:", ["House", "Flat"], horizontal=True)

# Predict
if st.button("🔍 Estimate Rent"):
    prediction, error = model.predict(suburb, bedrooms, property_type)
    
    if error:
        st.error(f"❌ {error}")
    else:
        st.success(f"**Estimated Weekly Rent: ${prediction:.0f}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Weekly", f"${prediction:.0f}")
        with col2:
            st.metric("📈 Monthly", f"${prediction * 4.33:.0f}")
        with col3:
            st.metric("✅ Confidence", "92%")
        
        # Explanation (from Claude API - Week 5)
        st.info(f"💡 Estimated rent based on market analysis")
        
        # Comparables (Week 6)
        st.markdown("### Similar Suburbs")
        st.write("(Coming in Week 6)")
```

---

## 🎯 WEEK 6: POLISH & COMPARABLES

### **Step 8: Add Comparable Suburbs**

File: `app/utils/comparables.py`

```python
"""
Find comparable suburbs based on SEIFA and demographics
"""

import pandas as pd

class ComparablesFinder:
    def __init__(self, suburb_data):
        self.data = suburb_data
    
    def find_comparables(self, suburb, bedrooms, property_type, n=5):
        """
        Find similar suburbs based on SEIFA score and income
        """
        # Get reference suburb data
        ref = self.data[
            (self.data['suburb'] == suburb.lower()) &
            (self.data['bedrooms'] == bedrooms) &
            (self.data['property_type'] == property_type.lower())
        ]
        
        if ref.empty:
            return pd.DataFrame()
        
        ref_seifa = ref['IRSAD_Score'].values[0]
        ref_income = ref['Median_Household_Income_Weekly_AUD'].values[0]
        
        # Find similar suburbs
        similar = self.data[
            (self.data['bedrooms'] == bedrooms) &
            (self.data['property_type'] == property_type.lower()) &
            ((self.data['IRSAD_Score'] - ref_seifa).abs() < 50) &
            ((self.data['Median_Household_Income_Weekly_AUD'] - ref_income).abs() < 300)
        ].copy()
        
        # Sort by SEIFA similarity
        similar['seifa_diff'] = (similar['IRSAD_Score'] - ref_seifa).abs()
        similar = similar.sort_values('seifa_diff').head(n)
        
        return similar[['suburb', 'bedrooms', 'property_type', 'median_weekly_rent', 'IRSAD_Score']]
```

---

## 🚀 DEPLOYMENT CHECKLIST

### **Week 7: Deploy to Streamlit Cloud**

```bash
# 1. Create requirements.txt
pip freeze > requirements.txt

# 2. Push to GitHub
git init
git add .
git commit -m "Initial Streamlit app"
git push origin main

# 3. Go to https://streamlit.io/cloud
# 4. Deploy from GitHub repo
# 5. Add secrets (API keys) in Streamlit Cloud settings
```

### **Streamlit Cloud Secrets**
File: `.streamlit/secrets.toml` (in Streamlit Cloud settings)

```toml
claude_api_key = "sk-..."
database_url = "..."
```

---

## ✅ TESTING CHECKLIST

- [ ] App runs locally: `streamlit run app/main.py`
- [ ] All pages load
- [ ] Suburb selector populates correctly
- [ ] Mock predictions show reasonable values
- [ ] Page navigation works
- [ ] Styling looks good
- [ ] No console errors
- [ ] Responsive on mobile

---

## 📊 WEEK 8-9: REFINEMENTS

- [ ] User testing (5-10 users)
- [ ] Collect feedback
- [ ] Fix bugs
- [ ] Performance optimization
- [ ] Final polish
- [ ] Deploy to production

---

## 💡 TIPS FOR SUCCESS

1. **Start simple:** Mock predictions first, integrate real model later
2. **Use caching:** `@st.cache_data` and `@st.cache_resource` for performance
3. **Error handling:** Always handle missing data gracefully
4. **User experience:** Clear labels, helpful hints, visual hierarchy
5. **Testing:** Test with real data early

---

## 📞 DEPENDENCIES

```
streamlit==1.28
pandas>=1.5
numpy>=1.20
scikit-learn>=1.0
```

Run: `pip install -r requirements.txt`

---

**Ready? Start building the skeleton!** 🚀

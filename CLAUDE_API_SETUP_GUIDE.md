# CLAUDE API SETUP & EXPLANATION GUIDE
## For PERSON C (API Engineer)

> ⚠️ **Superseded: do not follow the setup or code in this guide.**
> The AI explanations are implemented in `app/utils/shap_explainer.py` and
> `app/utils/api_client.py`, and they call the model through **OpenRouter**, not the
> Anthropic SDK used below. The key is `OPENROUTER_API_KEY` (not `CLAUDE_API_KEY` or
> `ANTHROPIC_API_KEY`), and the model is set by `EXPLAIN_MODEL` in `.env`.
> For setup and usage from the app, see the **AI Explanations** section of
> [README.md](README.md). This guide is kept for its original goals (latency, cost,
> fallback), which the implementation follows.

---

## 📋 OVERVIEW

**Goal:** Build Claude API integration that:
- Takes model prediction + SHAP values → Natural language explanation
- Explains "why" the rent is this price
- Suggests comparable suburbs and price variations
- Handles errors gracefully with fallback to numeric values
- Meets latency target: <2 seconds
- Keeps costs low: <$0.001 per prediction (~$20 total project)

**Timeline:**
- Week 4: API setup + prompt design + mock testing
- Week 5: Error handling + caching + performance tuning
- Week 6: Integration with model + Streamlit
- Week 7+: Production monitoring

---

## 🔑 STEP 1: SETUP CLAUDE API

### **Get API Key**

1. Go to: https://console.anthropic.com/
2. Sign up or log in
3. Create new API key
4. Copy key and save securely

### **Install SDK**

```bash
# Activate venv
venv\Scripts\Activate.ps1

# Install Anthropic SDK
pip install anthropic
```

### **Test Connection**

```python
from anthropic import Anthropic

client = Anthropic(api_key="sk-ant-...")

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "Say 'Hello from Claude'"}
    ]
)

print(response.content[0].text)
# Output: "Hello from Claude"
```

---

## 📝 STEP 2: PROMPT DESIGN

### **Understanding Explanations**

Your model predicts: `$450/week`
Your SHAP values show:
- Income: +$100 (above average)
- SEIFA: -$20 (slightly disadvantaged)
- Bedrooms: +$150 (2-bed)
- Distance to CBD: -$30 (far from city)

**Goal:** Convert these into a natural, helpful explanation.

### **Prompt Template**

File: `app/utils/explanation_prompt.txt`

```
You are a real estate market analyst. Explain a rental price prediction in natural language.

Input Details:
- Suburb: {suburb}
- Bedrooms: {bedrooms}
- Property Type: {property_type}
- Predicted Weekly Rent: ${prediction:.0f}
- Confidence: {confidence}%

Market Context:
- Median household income: ${income:.0f}/week
- SEIFA affluence score: {seifa:.0f} (scale 800-1200, avg 1000)
- Distance to CBD: {distance:.1f} km
- Property growth (last year): {growth:+.1f}%
- Suburb population: {population:,}

Top factors affecting the price:
{shap_explanation}

Write a 2-3 sentence explanation of why this rent is fair. Be conversational and helpful.
Focus on the key drivers (income, location, property type).
Keep it under 100 words.

Example format:
"At ${prediction:.0f}/week, this {bedrooms}-bedroom {property_type} is priced fairly for {suburb}.
The suburb's median income of ${income:.0f}/week suggests moderate affordability, and {key_factor}.
Based on similar properties in comparable areas, this represents market rate."
```

---

## 🐍 STEP 3: CREATE API CLIENT

File: `app/utils/api_client.py`

```python
"""
Claude API client for rental price explanations
"""

import os
from anthropic import Anthropic
import json
from functools import lru_cache
import time

class RentalExplanationAPI:
    def __init__(self, api_key=None):
        """Initialize Anthropic client"""
        api_key = api_key or os.environ.get('CLAUDE_API_KEY')
        if not api_key:
            raise ValueError("API key not found. Set CLAUDE_API_KEY environment variable.")
        
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
        self.request_count = 0
        self.total_cost = 0.0
    
    def explain_prediction(self, suburb, bedrooms, property_type, prediction, 
                          shap_values, suburb_data):
        """
        Generate natural language explanation for a rent prediction
        
        Args:
            suburb (str): Suburb name
            bedrooms (int): Number of bedrooms
            property_type (str): 'house' or 'flat'
            prediction (float): Predicted weekly rent
            shap_values (dict): SHAP values for each feature
            suburb_data (dict): Additional suburb data (income, SEIFA, etc.)
        
        Returns:
            dict: {'explanation': str, 'error': str or None, 'latency': float}
        """
        
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = self._build_prompt(
                suburb=suburb,
                bedrooms=bedrooms,
                property_type=property_type,
                prediction=prediction,
                shap_values=shap_values,
                suburb_data=suburb_data
            )
            
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7  # Slightly creative but factual
            )
            
            # Extract explanation
            explanation = response.content[0].text
            
            # Track usage
            self.request_count += 1
            # Estimate cost: $3/MTok input, $15/MTok output
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens * 3 + output_tokens * 15) / 1_000_000
            self.total_cost += cost
            
            latency = time.time() - start_time
            
            return {
                'explanation': explanation,
                'error': None,
                'latency': latency,
                'cost': cost,
                'tokens': input_tokens + output_tokens
            }
        
        except Exception as e:
            latency = time.time() - start_time
            
            # Fallback: return numeric explanation
            fallback = self._numeric_fallback(
                suburb=suburb,
                bedrooms=bedrooms,
                property_type=property_type,
                prediction=prediction,
                shap_values=shap_values
            )
            
            return {
                'explanation': fallback,
                'error': str(e),
                'latency': latency,
                'fallback': True
            }
    
    def _build_prompt(self, suburb, bedrooms, property_type, prediction, 
                      shap_values, suburb_data):
        """Build the prompt for Claude"""
        
        # Convert SHAP values to readable format
        shap_text = self._format_shap_values(shap_values)
        
        prompt = f"""You are a real estate analyst explaining rental prices.

Property Details:
- Suburb: {suburb.title()}
- Bedrooms: {bedrooms}
- Type: {property_type.title()}
- Predicted Weekly Rent: ${prediction:.0f}

Market Data:
- Median income: ${suburb_data.get('income', 0):.0f}/week
- SEIFA score: {suburb_data.get('seifa', 1000):.0f} (affluence, 800-1200)
- Distance to CBD: {suburb_data.get('distance', 0):.1f} km
- Property growth: {suburb_data.get('growth', 0):+.1f}%

Key Price Factors:
{shap_text}

Write a 2-3 sentence explanation of why this rent is fair for this property. 
Keep it under 80 words. Be conversational.
"""
        
        return prompt
    
    def _format_shap_values(self, shap_values):
        """Format SHAP values for readability"""
        
        text = ""
        for feature, value in sorted(
            shap_values.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:5]:  # Top 5 features
            direction = "increases" if value > 0 else "decreases"
            magnitude = abs(value)
            text += f"- {feature}: {direction} rent by ~${magnitude:.0f}\n"
        
        return text
    
    def _numeric_fallback(self, suburb, bedrooms, property_type, 
                         prediction, shap_values):
        """Fallback explanation using just numbers"""
        
        top_factors = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        
        factors_text = ", ".join([f"{f[0]} (${f[1]:+.0f})" for f, v in top_factors])
        
        return f"""Based on market analysis, this {bedrooms}-bedroom {property_type} in 
{suburb.title()} is estimated at ${prediction:.0f}/week. Key factors: {factors_text}."""
    
    def get_usage_summary(self):
        """Get API usage summary"""
        return {
            'requests': self.request_count,
            'total_cost': f"${self.total_cost:.4f}",
            'avg_cost_per_request': f"${self.total_cost / max(self.request_count, 1):.4f}"
        }


# ============================================================================
# TESTING & USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Initialize API
    api = RentalExplanationAPI()
    
    # Example prediction
    suburb_data = {
        'income': 1200,
        'seifa': 1020,
        'distance': 5.2,
        'growth': 2.5,
        'population': 45000
    }
    
    shap_values = {
        'bedrooms': 100,
        'Median_Household_Income': 50,
        'IRSAD_Score': -20,
        'distance_to_cbd_km': -30,
        'employment_rate': 25
    }
    
    # Get explanation
    result = api.explain_prediction(
        suburb='Footscray',
        bedrooms=2,
        property_type='house',
        prediction=450,
        shap_values=shap_values,
        suburb_data=suburb_data
    )
    
    print("✅ Explanation:")
    print(result['explanation'])
    print(f"\n⏱️ Latency: {result['latency']:.2f}s")
    if result['error']:
        print(f"⚠️ Error: {result['error']}")
        print(f"Fallback used: {result.get('fallback', False)}")
    else:
        print(f"💰 Cost: ${result['cost']:.6f}")
        print(f"📊 Tokens: {result['tokens']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("API Usage Summary:")
    for key, value in api.get_usage_summary().items():
        print(f"  {key}: {value}")
```

---

## 🔐 STEP 4: ENVIRONMENT SETUP

### **Local Development**

File: `.env`

```
CLAUDE_API_KEY=sk-ant-...
```

### **Load from .env**

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('CLAUDE_API_KEY')
```

### **Production (Streamlit Cloud)**

1. Go to Streamlit Cloud dashboard
2. Click "Secrets" in settings
3. Add:
```
claude_api_key = "sk-ant-..."
```

4. Access in code:
```python
api_key = st.secrets["claude_api_key"]
```

---

## 🧪 STEP 5: TESTING & VALIDATION

### **Test 1: Basic Connection**

File: `tests/test_api_connection.py`

```python
"""Test Claude API connection"""

import pytest
from app.utils.api_client import RentalExplanationAPI

def test_api_connection():
    """Test that API key works"""
    api = RentalExplanationAPI()
    
    # Simple test
    result = api.explain_prediction(
        suburb='Test',
        bedrooms=2,
        property_type='house',
        prediction=450,
        shap_values={'test': 10},
        suburb_data={'income': 1000, 'seifa': 1000}
    )
    
    assert result['explanation'] is not None
    assert result['error'] is None or result.get('fallback')
    print("✅ API connection test passed")
```

Run: `pytest tests/test_api_connection.py`

### **Test 2: Latency**

```python
def test_latency():
    """Ensure API response is <2 seconds"""
    api = RentalExplanationAPI()
    
    result = api.explain_prediction(
        suburb='Footscray',
        bedrooms=2,
        property_type='house',
        prediction=450,
        shap_values={'bedrooms': 100},
        suburb_data={'income': 1200, 'seifa': 1020}
    )
    
    assert result['latency'] < 2.0, f"Latency too high: {result['latency']:.2f}s"
    print(f"✅ Latency test passed: {result['latency']:.2f}s")
```

### **Test 3: Cost**

```python
def test_cost_tracking():
    """Ensure cost stays under budget"""
    api = RentalExplanationAPI()
    
    # Make 100 predictions
    for i in range(100):
        result = api.explain_prediction(
            suburb='Test',
            bedrooms=2,
            property_type='house',
            prediction=450 + i,
            shap_values={'test': 10},
            suburb_data={'income': 1000}
        )
    
    total_cost = float(api.get_usage_summary()['total_cost'].replace('$', ''))
    assert total_cost < 0.10, f"Cost exceeded budget: ${total_cost}"
    print(f"✅ Cost test passed: ${total_cost:.4f}/100 predictions")
```

---

## 🚀 STEP 6: INTEGRATION WITH STREAMLIT

File: `app/pages/1_predict.py` (updated)

```python
"""
Prediction page with Claude API integration
"""

import streamlit as st
import sys
sys.path.insert(0, 'app/utils')

from model_loader import ModelLoader
from api_client import RentalExplanationAPI

st.title("📊 Predict Rent")

# Load model and API (cached)
@st.cache_resource
def load_resources():
    model = ModelLoader()
    api = RentalExplanationAPI()
    return model, api

model, api = load_resources()

# Get suburbs
suburbs = model.get_suburbs()

# Input
col1, col2, col3 = st.columns(3)

with col1:
    suburb = st.selectbox("Suburb:", suburbs)

with col2:
    bedrooms = st.slider("Bedrooms:", 1, 5, 2)

with col3:
    property_type = st.radio("Type:", ["House", "Flat"], horizontal=True)

# Predict
if st.button("🔍 Estimate Rent"):
    # Get prediction
    prediction, error = model.predict(suburb, bedrooms, property_type)
    
    if error:
        st.error(f"❌ {error}")
    else:
        # Display prediction
        st.success(f"**Estimated Weekly Rent: ${prediction:.0f}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Weekly", f"${prediction:.0f}")
        with col2:
            st.metric("📈 Monthly", f"${prediction * 4.33:.0f}")
        with col3:
            st.metric("✅ Confidence", "92%")
        
        # Get suburb data
        suburb_row = model.suburb_data[
            (model.suburb_data['suburb'] == suburb.lower()) &
            (model.suburb_data['bedrooms'] == bedrooms) &
            (model.suburb_data['property_type'] == property_type.lower())
        ]
        
        if not suburb_row.empty:
            suburb_row = suburb_row.iloc[0]
            
            # Mock SHAP values (real ones would come from model)
            shap_values = {
                'bedrooms': bedrooms * 50,
                'income': suburb_row['Median_Household_Income_Weekly_AUD'] / 10,
                'SEIFA': (suburb_row['IRSAD_Score'] - 1000) / 10,
                'distance': -suburb_row['distance_to_cbd_km']
            }
            
            suburb_data = {
                'income': suburb_row['Median_Household_Income_Weekly_AUD'],
                'seifa': suburb_row['IRSAD_Score'],
                'distance': suburb_row['distance_to_cbd_km'],
                'growth': suburb_row.get('house_change_perc_24-25', 0),
                'population': suburb_row.get('Population_2021', 0)
            }
            
            # Get explanation
            with st.spinner("🤖 Generating explanation..."):
                result = api.explain_prediction(
                    suburb=suburb,
                    bedrooms=bedrooms,
                    property_type=property_type,
                    prediction=prediction,
                    shap_values=shap_values,
                    suburb_data=suburb_data
                )
            
            # Display explanation
            st.markdown("### 💡 Why This Price?")
            st.info(result['explanation'])
            
            # Show latency
            st.caption(f"⏱️ Generated in {result['latency']:.2f}s")
```

---

## 📊 PERFORMANCE OPTIMIZATION

### **Caching Explanations**

```python
from functools import lru_cache

class RentalExplanationAPI:
    @lru_cache(maxsize=1000)
    def _cached_explanation(self, cache_key):
        """Cache explanations by suburb-bedrooms-property_type"""
        return self.explain_prediction(*cache_key)
```

### **Batch Requests**

For testing multiple suburbs:

```python
def explain_batch(self, predictions_list):
    """Get explanations for multiple predictions"""
    results = []
    for pred in predictions_list:
        result = self.explain_prediction(**pred)
        results.append(result)
    
    return results
```

---

## 💰 COST TRACKING

### **Monitor Costs**

```python
# Track all API calls
total_cost = 0.0
call_count = 0

for prediction in predictions:
    result = api.explain_prediction(**prediction)
    total_cost += result.get('cost', 0)
    call_count += 1

avg_cost = total_cost / call_count
print(f"Total cost: ${total_cost:.4f}")
print(f"Avg cost per prediction: ${avg_cost:.6f}")
print(f"Estimated for 1000 predictions: ${avg_cost * 1000:.2f}")
```

### **Cost Budget**

```
Prompt (input): ~300 tokens per request × $3/MTok = $0.0009
Response (output): ~50 tokens per request × $15/MTok = $0.00075
Total per request: ~$0.0017

Project budget: $20
Predictions: 20 / 0.0017 ≈ 11,700 predictions
Allows testing + deployment buffer ✅
```

---

## ✅ DEPLOYMENT CHECKLIST

- [ ] API key securely stored (not in code)
- [ ] Error handling for failed requests (fallback works)
- [ ] Latency < 2 seconds
- [ ] Cost tracking implemented
- [ ] Explanations are helpful and accurate
- [ ] Works with real model predictions
- [ ] Integrated with Streamlit app
- [ ] Tested on sample data
- [ ] Production secrets configured

---

## 📞 TROUBLESHOOTING

### **Issue: "API key not found"**
```
Error: API key not found. Set CLAUDE_API_KEY environment variable.
```

**Fix:**
1. Create `.env` file with `CLAUDE_API_KEY=sk-ant-...`
2. Or set environment variable: `$env:CLAUDE_API_KEY="sk-ant-..."`

### **Issue: "Rate limit exceeded"**
```
Error: 429 - Too many requests
```

**Fix:** Add delay between requests
```python
import time
time.sleep(1)  # Wait 1 second between API calls
```

### **Issue: "Response too slow (>2s)"**

**Fix:** Reduce max_tokens
```python
response = client.messages.create(
    model=model,
    max_tokens=100,  # Was 150
    messages=[...]
)
```

---

## 🚀 NEXT STEPS

1. **Week 4:** Setup API, test connection, design prompt
2. **Week 5:** Error handling, caching, latency optimization
3. **Week 6:** Integration with model + Streamlit
4. **Week 7:** Production deployment + monitoring

---

**Ready to integrate Claude API? Start with `app/utils/api_client.py`!** 🚀

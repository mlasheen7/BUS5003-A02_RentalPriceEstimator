# BUS5003-A02_RentalPriceEstimator

## Setup Instructions

### 1. Clone Repository
\`\`\`bash
git clone https://github.com/[your-username]/rental-price-estimator.git
cd rental-price-estimator
\`\`\`

### 2. Create Virtual Environment
\`\`\`bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# OR venv\Scripts\activate  # Windows
\`\`\`

### 3. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Set Up Environment Variables
\`\`\`bash
cp .env.example .env
# Edit .env and add your actual API keys
\`\`\`

### 5. Verify Installation
\`\`\`bash
python -c "import pandas, xgboost, anthropic; print('✅ All imports successful')"
\`\`\`

### Data Access
Raw data files are in `data/raw/`. Processed data will be in `data/processed/`.
See `data/README.md` for documentation.
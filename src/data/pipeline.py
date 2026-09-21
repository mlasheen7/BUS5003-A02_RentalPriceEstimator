"""
================================================================================
DATA MERGING & FEATURE ENGINEERING PIPELINE (REVISED V2)
================================================================================
File: src/data/pipeline.py

KEY CHANGES FROM V1:
1. Splits hyphen-separated suburbs (e.g., "Albert Park-Middle Park" → 2 rows)
2. Includes property type (house/flat) as a feature
3. Treats flat and unit as same property type
4. Uses last 12 months of data (more stable than 1 quarter)
5. Loads all sheets from rental file (handles different property types/sizes)

Output structure:
  - One row per: suburb-bedroom-property_type combination
  - Example: Footscray, 2-bed, house vs Footscray, 2-bed, flat
  - Expected: ~250 suburbs × 3 bedrooms × 2 property_types = ~1500 rows

Run: python src/data/pipeline.py

Output files:
- data/processed/merged_rental_data.csv
- data/processed/X_train.pkl (with bedrooms & property_type features)
- data/processed/y_train.pkl
- data/processed/feature_names.txt
- data/processed/data_dictionary.txt
================================================================================
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from datetime import datetime, timedelta
import os

# ============================================================================
# SETUP
# ============================================================================

print("=" * 80)
print("DATA MERGING & FEATURE ENGINEERING PIPELINE (V2)")
print("WITH SUBURB SPLITTING, PROPERTY TYPES, AND 12-MONTH AGGREGATION")
print("=" * 80)

# Create output directory if doesn't exist
os.makedirs('data/processed', exist_ok=True)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[STEP 1] Loading data files from data/raw/...")

try:
    # Load the main rental file - it may have multiple sheets for different property types
    rental_file = 'data/raw/Moving_Annual_Median_Rent_by_SuburbTown_-_Main_File.csv'
    rental_main = pd.read_csv(rental_file)
    print(f"  ✓ Rental (CSV): {len(rental_main)} rows")
except FileNotFoundError:
    print("  ✗ ERROR: Moving_Annual_Median_Rent_by_SuburbTown_-_Main_File.csv not found")
    exit(1)

try:
    demographics = pd.read_excel('data/raw/OBA_Victoria_Suburb_Profiling_Clean.xlsx', sheet_name='Clean_Data')
    print(f"  ✓ Demographics (OBA): {len(demographics)} suburbs")
except FileNotFoundError:
    print("  ✗ ERROR: OBA_Victoria_Suburb_Profiling_Clean.xlsx not found")
    exit(1)

try:
    valuation = pd.read_excel('data/raw/valuation_data.xlsx', sheet_name='Sheet1')
    print(f"  ✓ Valuation data: {len(valuation)} suburbs")
except FileNotFoundError:
    print("  ✗ ERROR: valuation_data.xlsx not found")
    exit(1)

print("\n✅ All data files loaded successfully!")

# ============================================================================
# STEP 2: EXPLORE RENTAL DATA STRUCTURE
# ============================================================================

print("\n[STEP 2] Exploring rental data structure...")

print(f"\n  Rental data columns:")
print(f"    {rental_main.columns.tolist()}")

print(f"\n  Rental data info:")
print(f"    - Rows: {len(rental_main)}")
print(f"    - Suburbs (before split): {rental_main['Suburb_Town_Area'].nunique()}")

# Check for different property types
if 'Property_Type' in rental_main.columns:
    print(f"    - Property types: {rental_main['Property_Type'].unique()}")
elif 'Type' in rental_main.columns:
    print(f"    - Types: {rental_main['Type'].unique()}")
else:
    print(f"    - No 'Property_Type' or 'Type' column found")

# Check bedroom column
if 'Bedrooms' in rental_main.columns:
    print(f"    - Bedroom types: {sorted(rental_main['Bedrooms'].dropna().unique())}")

# Check time period column
if 'Period_End' in rental_main.columns:
    print(f"    - Unique periods: {rental_main['Period_End'].nunique()}")
    print(f"    - Date range: {rental_main['Period_End'].min()} to {rental_main['Period_End'].max()}")
elif 'Year' in rental_main.columns:
    print(f"    - Years available: {sorted(rental_main['Year'].unique())}")

# Show sample of suburb names (to see if they have hyphens)
print(f"\n  Sample suburbs (checking for hyphen-separated names):")
for i, suburb in enumerate(rental_main['Suburb_Town_Area'].unique()[:5]):
    print(f"    - {suburb}")

# ============================================================================
# STEP 3: SPLIT HYPHEN-SEPARATED SUBURBS
# ============================================================================

print("\n[STEP 3] Splitting hyphen-separated suburbs...")

def split_suburbs(suburb_str):
    """
    Split suburbs separated by hyphens or commas.
    Example: "Albert Park-Middle Park-West St Kilda" → ["Albert Park", "Middle Park", "West St Kilda"]
    """
    if pd.isna(suburb_str):
        return [suburb_str]
    
    # Split by hyphen and/or comma
    parts = []
    for part in str(suburb_str).split('-'):
        part = part.strip()
        if part:
            parts.append(part)
    
    return parts if parts else [suburb_str]

# Expand the dataframe to have one row per suburb
print(f"  Expanding suburbs (splitting hyphen-separated names)...")

# Create a list to store expanded rows
expanded_rows = []

for idx, row in rental_main.iterrows():
    suburbs = split_suburbs(row['Suburb_Town_Area'])
    for suburb in suburbs:
        new_row = row.copy()
        new_row['Suburb_Town_Area'] = suburb.strip()
        expanded_rows.append(new_row)

rental_expanded = pd.DataFrame(expanded_rows)

print(f"  ✓ Rows after splitting: {len(rental_expanded)}")
print(f"  ✓ Unique suburbs after split: {rental_expanded['Suburb_Town_Area'].nunique()}")
print(f"\n  Example of split suburbs:")
sample_suburbs = rental_expanded['Suburb_Town_Area'].unique()[:10]
for suburb in sample_suburbs:
    print(f"    - {suburb}")

# ============================================================================
# STEP 4: AGGREGATE LAST 12 MONTHS OF RENT DATA
# ============================================================================

print("\n[STEP 4] Aggregating rent data from last 12 months...")

# Identify the period column
period_col = None
if 'Period_End' in rental_expanded.columns:
    period_col = 'Period_End'
elif 'Date' in rental_expanded.columns:
    period_col = 'Date'
elif 'Year' in rental_expanded.columns:
    period_col = 'Year'

if period_col:
    print(f"  Using period column: {period_col}")
    
    # Convert to datetime if needed
    try:
        rental_expanded[period_col] = pd.to_datetime(rental_expanded[period_col])
    except:
        print(f"  Warning: Could not convert {period_col} to datetime, using as-is")
    
    # Get latest date
    latest_date = rental_expanded[period_col].max()
    print(f"  Latest period: {latest_date}")
    
    # Calculate 12 months back
    if pd.api.types.is_datetime64_any_dtype(rental_expanded[period_col]):
        cutoff_date = latest_date - timedelta(days=365)
        print(f"  Cutoff (12 months back): {cutoff_date}")
        
        # Filter to last 12 months
        rental_recent = rental_expanded[rental_expanded[period_col] >= cutoff_date].copy()
    else:
        # If not datetime, just use latest data
        print(f"  Period not in datetime format, using latest period data")
        rental_recent = rental_expanded[rental_expanded[period_col] == latest_date].copy()
else:
    print(f"  No period column found, using all data")
    rental_recent = rental_expanded.copy()

print(f"  ✓ Rows in last 12 months: {len(rental_recent)}")

# ============================================================================
# STEP 5: AGGREGATE RENT BY SUBURB-BEDROOM-PROPERTY_TYPE
# ============================================================================

print("\n[STEP 5] Aggregating rent by suburb, bedroom, and property type...")

# Identify property type column
property_type_col = None
if 'Property_Type' in rental_recent.columns:
    property_type_col = 'Property_Type'
elif 'Type' in rental_recent.columns:
    property_type_col = 'Type'

if property_type_col:
    print(f"  Property type column: {property_type_col}")
    print(f"  Property types in data: {rental_recent[property_type_col].unique()}")
    
    # Normalize property types (treat unit and flat as same)
    def normalize_property_type(prop_type):
        if pd.isna(prop_type):
            return 'unknown'
        prop_type = str(prop_type).lower().strip()
        if 'unit' in prop_type or 'flat' in prop_type:
            return 'flat'
        elif 'house' in prop_type:
            return 'house'
        elif 'town' in prop_type:
            return 'townhouse'
        else:
            return prop_type
    
    rental_recent[property_type_col] = rental_recent[property_type_col].apply(normalize_property_type)
    print(f"  Normalized property types: {rental_recent[property_type_col].unique()}")
    
    # Aggregate: mean rent by suburb, bedroom, property type
    rental_grouped = rental_recent.groupby(
        ['Suburb_Town_Area', 'Bedrooms', property_type_col],
        as_index=False
    ).agg({
        'Median_Weekly_Rent_AUD': 'mean',
        'Rental_Count': 'sum',
        'Region': 'first'
    })
    
    rental_grouped = rental_grouped.rename(columns={
        'Suburb_Town_Area': 'suburb',
        'Bedrooms': 'bedrooms',
        property_type_col: 'property_type',
        'Median_Weekly_Rent_AUD': 'median_weekly_rent',
        'Rental_Count': 'rental_count',
        'Region': 'region'
    })
else:
    print(f"  No property type column found, treating all as 'all'")
    
    # Aggregate without property type
    rental_grouped = rental_recent.groupby(
        ['Suburb_Town_Area', 'Bedrooms'],
        as_index=False
    ).agg({
        'Median_Weekly_Rent_AUD': 'mean',
        'Rental_Count': 'sum',
        'Region': 'first'
    })
    
    rental_grouped = rental_grouped.rename(columns={
        'Suburb_Town_Area': 'suburb',
        'Bedrooms': 'bedrooms',
        'Median_Weekly_Rent_AUD': 'median_weekly_rent',
        'Rental_Count': 'rental_count',
        'Region': 'region'
    })
    
    rental_grouped['property_type'] = 'all'

# Remove null rents
rental_grouped = rental_grouped.dropna(subset=['median_weekly_rent'])

# Clean bedrooms column (convert to numeric)
def clean_bedrooms(bed_str):
    """Extract number from bedroom string (e.g., '2 bed' -> 2)"""
    if pd.isna(bed_str):
        return np.nan
    bed_str = str(bed_str).strip().lower()
    if 'studio' in bed_str:
        return 0
    try:
        return int(''.join(filter(str.isdigit, bed_str.split()[0])))
    except:
        return np.nan

rental_grouped['bedrooms'] = rental_grouped['bedrooms'].apply(clean_bedrooms)
rental_grouped = rental_grouped.dropna(subset=['bedrooms'])
rental_grouped['bedrooms'] = rental_grouped['bedrooms'].astype(int)

print(f"\n  ✓ Rows after grouping: {len(rental_grouped)}")
print(f"  ✓ Unique suburbs: {rental_grouped['suburb'].nunique()}")
print(f"  ✓ Unique bedrooms: {sorted(rental_grouped['bedrooms'].unique())}")
print(f"  ✓ Property types: {rental_grouped['property_type'].unique()}")

# Statistics
print(f"\n  Rent statistics by property type:")
for prop_type in sorted(rental_grouped['property_type'].unique()):
    data = rental_grouped[rental_grouped['property_type'] == prop_type]
    print(f"    {prop_type}:")
    print(f"      - Records: {len(data)}")
    print(f"      - Mean rent: ${data['median_weekly_rent'].mean():.0f}±${data['median_weekly_rent'].std():.0f}/week")
    print(f"      - Rent range: ${data['median_weekly_rent'].min():.0f}–${data['median_weekly_rent'].max():.0f}/week")

print(f"\n  Rent statistics by bedroom type:")
for bed in sorted(rental_grouped['bedrooms'].unique()):
    data = rental_grouped[rental_grouped['bedrooms'] == bed]
    print(f"    {int(bed)}-bed: ${data['median_weekly_rent'].mean():.0f}±${data['median_weekly_rent'].std():.0f}/week (n={len(data)})")

# ============================================================================
# STEP 6: STANDARDIZE SUBURB NAMES
# ============================================================================

print("\n[STEP 6] Standardizing suburb names for matching...")

def standardize_name(name):
    """Standardize suburb name: lowercase, strip whitespace"""
    if pd.isna(name):
        return name
    return name.strip().lower()

rental_grouped['suburb'] = rental_grouped['suburb'].apply(standardize_name)
demographics['Suburb'] = demographics['Suburb'].apply(standardize_name)
valuation['Locality'] = valuation['Locality'].apply(standardize_name)

print(f"  ✓ Rental suburbs: {rental_grouped['suburb'].nunique()}")
print(f"  ✓ Demographics suburbs: {demographics['Suburb'].nunique()}")
print(f"  ✓ Valuation suburbs: {valuation['Locality'].nunique()}")

# Check for duplicates (by suburb-bedroom-property_type)
dup_rental = rental_grouped.duplicated(subset=['suburb', 'bedrooms', 'property_type']).sum()
print(f"  Duplicates in rental data (by suburb-bedroom-property_type): {dup_rental}")

# ============================================================================
# STEP 7: MERGE DATASETS
# ============================================================================

print("\n[STEP 7] Merging datasets on suburb (keeping bedroom and property types)...")

# Start with rental data
merged = rental_grouped.copy()
print(f"  [1] Starting with rental: {len(merged)} rows")
print(f"      ({merged['suburb'].nunique()} suburbs × {merged['bedrooms'].nunique()} bedroom types × {merged['property_type'].nunique()} property types)")

# Merge demographics (on suburb only)
print(f"  [2] Merging demographics...")

demo_cols = ['Suburb', 'Population_2021', 'Median_Household_Income_Weekly_AUD',
             'Year12_Completion_Rate_Pct', 'Unemployment_Rate_Pct',
             'Labour_Force_Participation_Rate_Pct', 'IRSAD_Score',
             'Latitude', 'Longitude']

demo_cols = [c for c in demo_cols if c in demographics.columns]

merged = merged.merge(
    demographics[demo_cols],
    left_on='suburb',
    right_on='Suburb',
    how='left'
)

merged = merged.drop('Suburb', axis=1, errors='ignore')

print(f"      After merge: {len(merged)} rows")
print(f"      With demographics: {merged['IRSAD_Score'].notna().sum()} rows")

# Merge valuation data (on suburb only)
print(f"  [3] Merging valuation data...")

val_cols = [c for c in ['Locality', 'house_change_perc_24-25', 'house_change_perc_15-25', 
                         'unit_change_perc_24-25'] if c in valuation.columns]

if len(val_cols) > 0:
    merged = merged.merge(
        valuation[val_cols],
        left_on='suburb',
        right_on='Locality',
        how='left'
    )
    merged = merged.drop('Locality', axis=1, errors='ignore')
    print(f"      After merge: {len(merged)} rows")

# ============================================================================
# STEP 8: HANDLE MISSING DATA
# ============================================================================

print("\n[STEP 8] Handling missing data...")

print(f"  [1] Rows before filtering: {len(merged)}")

# CRITICAL: Keep only rows with rent AND income
merged_clean = merged[
    (merged['median_weekly_rent'].notna()) & 
    (merged['Median_Household_Income_Weekly_AUD'].notna())
].copy()

print(f"  [2] Rows after requiring rent + income: {len(merged_clean)}")

# Fill optional columns
print(f"  [3] Filling missing values in optional columns...")

if 'IRSAD_Score' in merged_clean.columns:
    seifa_median = merged_clean['IRSAD_Score'].median()
    seifa_nulls = merged_clean['IRSAD_Score'].isna().sum()
    merged_clean['IRSAD_Score'].fillna(seifa_median, inplace=True)
    if seifa_nulls > 0:
        print(f"      IRSAD_Score: Filled {seifa_nulls} nulls with median ({seifa_median:.0f})")

for col in ['house_change_perc_24-25', 'house_change_perc_15-25', 'unit_change_perc_24-25']:
    if col in merged_clean.columns:
        nulls = merged_clean[col].isna().sum()
        merged_clean[col].fillna(0, inplace=True)

if 'Population_2021' in merged_clean.columns:
    merged_clean['Population_2021'].fillna(0, inplace=True)

print(f"  [4] Final null check: {merged_clean.isnull().sum().sum()} total nulls")

# ============================================================================
# STEP 9: FEATURE ENGINEERING
# ============================================================================

print("\n[STEP 9] Engineering features...")

# 9A: SEIFA → Affluence Bins
print(f"  [A] Creating affluence bins from SEIFA...")

if 'IRSAD_Score' in merged_clean.columns:
    try:
        merged_clean['affluence_bin'] = pd.qcut(
            merged_clean['IRSAD_Score'],
            q=5,
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
            duplicates='drop'
        )
        
        affluence_dummies = pd.get_dummies(merged_clean['affluence_bin'], prefix='affluence', drop_first=False)
        merged_clean = pd.concat([merged_clean, affluence_dummies], axis=1)
        
        print(f"      ✓ Affluence bins created")
    except Exception as e:
        print(f"      ⚠️ Warning: {e}")

# 9B: Distance to CBD
print(f"  [B] Calculating distance to CBD...")

CBD_LAT, CBD_LON = -37.8136, 144.9631

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points (km)"""
    if pd.isna(lat1) or pd.isna(lon1):
        return np.nan
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

if 'Latitude' in merged_clean.columns and 'Longitude' in merged_clean.columns:
    merged_clean['distance_to_cbd_km'] = merged_clean.apply(
        lambda row: haversine_distance(
            row['Latitude'], row['Longitude'],
            CBD_LAT, CBD_LON
        ),
        axis=1
    )
    
    valid = merged_clean['distance_to_cbd_km'].notna().sum()
    print(f"      ✓ Distance calculated for {valid} records")
    merged_clean['distance_to_cbd_km'].fillna(merged_clean['distance_to_cbd_km'].median(), inplace=True)

# 9C: Income × SEIFA Interaction
print(f"  [C] Creating interaction features...")

if 'Median_Household_Income_Weekly_AUD' in merged_clean.columns and 'IRSAD_Score' in merged_clean.columns:
    merged_clean['income_x_seifa'] = (
        merged_clean['Median_Household_Income_Weekly_AUD'] * 
        merged_clean['IRSAD_Score'] / 1000
    )
    print(f"      ✓ Income × SEIFA interaction created")

# 9D: Employment Rate
print(f"  [D] Deriving employment rate...")

if 'Labour_Force_Participation_Rate_Pct' in merged_clean.columns and 'Unemployment_Rate_Pct' in merged_clean.columns:
    merged_clean['employment_rate'] = (
        merged_clean['Labour_Force_Participation_Rate_Pct'] * 
        (1 - merged_clean['Unemployment_Rate_Pct'] / 100)
    )
    print(f"      ✓ Employment rate derived")

# 9E: Property Type Encoding
print(f"  [E] Encoding property types...")
property_type_dummies = pd.get_dummies(merged_clean['property_type'], prefix='property', drop_first=False)
merged_clean = pd.concat([merged_clean, property_type_dummies], axis=1)
print(f"      ✓ Property type one-hot encoded")
for ptype in merged_clean['property_type'].unique():
    count = len(merged_clean[merged_clean['property_type'] == ptype])
    print(f"        - {ptype}: {count} records")

# 9F: Growth Categories
print(f"  [F] Creating growth categories...")

if 'house_change_perc_24-25' in merged_clean.columns:
    merged_clean['growth_category'] = pd.cut(
        merged_clean['house_change_perc_24-25'],
        bins=[-np.inf, 0, 5, 10, np.inf],
        labels=['declining', 'slow_growth', 'moderate_growth', 'strong_growth']
    )
    
    growth_dummies = pd.get_dummies(merged_clean['growth_category'], prefix='growth', drop_first=False)
    merged_clean = pd.concat([merged_clean, growth_dummies], axis=1)
    print(f"      ✓ Growth categories created")

print(f"\n  ✅ Feature engineering complete!")

# ============================================================================
# STEP 10: PREPARE TRAINING DATA
# ============================================================================

print("\n[STEP 10] Preparing features and target for model training...")

# Include bedrooms and property_type as features!
feature_columns = [
    'bedrooms',  # Numeric: 1, 2, 3, 4, 5
    'Median_Household_Income_Weekly_AUD',
    'IRSAD_Score',
    'Labour_Force_Participation_Rate_Pct',
    'Unemployment_Rate_Pct',
    'Year12_Completion_Rate_Pct',
    'employment_rate',
    'distance_to_cbd_km',
    'income_x_seifa',
    'house_change_perc_24-25',
    'Population_2021',
    'rental_count',
]

# Add one-hot encoded features
feature_columns += [col for col in merged_clean.columns if col.startswith('affluence_')]
feature_columns += [col for col in merged_clean.columns if col.startswith('property_')]
feature_columns += [col for col in merged_clean.columns if col.startswith('growth_')]

# Filter to only features that exist
feature_columns = [col for col in feature_columns if col in merged_clean.columns]

# Create X and y
X = merged_clean[feature_columns].copy()
y = merged_clean['median_weekly_rent'].copy()

print(f"\n  Feature matrix (X):")
print(f"    - Shape: {X.shape[0]} rows × {X.shape[1]} columns")
print(f"    - Includes: bedrooms, property_type, income, SEIFA, distance, growth")

print(f"\n  Target vector (y):")
print(f"    - Shape: {y.shape[0]}")
print(f"    - Mean: ${y.mean():.2f}/week")
print(f"    - Std: ${y.std():.2f}/week")
print(f"    - Range: ${y.min():.2f}–${y.max():.2f}/week")

# Breakdown by property type
print(f"\n  Target statistics by property type:")
for ptype in sorted(merged_clean['property_type'].unique()):
    ptype_data = y[merged_clean['property_type'] == ptype]
    print(f"    {ptype}: ${ptype_data.mean():.0f}±${ptype_data.std():.0f}/week (n={len(ptype_data)})")

# Breakdown by bedroom
print(f"\n  Target statistics by bedroom:")
for bed in sorted(merged_clean['bedrooms'].unique()):
    bed_data = y[merged_clean['bedrooms'] == bed]
    print(f"    {int(bed)}-bed: ${bed_data.mean():.0f}±${bed_data.std():.0f}/week (n={len(bed_data)})")

# ============================================================================
# STEP 11: VALIDATION & OUTLIER DETECTION
# ============================================================================

print("\n[STEP 11] Validating data quality...")

# Check for outliers
q1 = y.quantile(0.25)
q3 = y.quantile(0.75)
iqr = q3 - q1
outlier_mask = (y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)
outliers = merged_clean[outlier_mask]

print(f"  Outliers detected (IQR method): {outlier_mask.sum()} records")
if len(outliers) > 0:
    print(f"  Examples:")
    for idx, row in outliers.head(5).iterrows():
        print(f"    - {row['suburb']} ({int(row['bedrooms'])}-bed {row['property_type']}): ${row['median_weekly_rent']:.0f}/week")

print(f"\n  ✅ Data quality summary:")
print(f"    - Total records (suburb-bedroom-property combos): {len(merged_clean)}")
print(f"    - Unique suburbs: {merged_clean['suburb'].nunique()}")
print(f"    - Features per record: {len(feature_columns)}")
print(f"    - Null values in X: {X.isnull().sum().sum()}")
print(f"    - Null values in y: {y.isnull().sum()}")

# ============================================================================
# STEP 12: SAVE ALL DATA
# ============================================================================

print("\n[STEP 12] Saving processed data...")

# Save full merged dataset
merged_clean.to_csv('data/processed/merged_rental_data.csv', index=False)
print(f"  ✓ Saved: data/processed/merged_rental_data.csv")
print(f"    ({len(merged_clean)} rows, {len(merged_clean.columns)} columns)")

# Save X and y
X.to_pickle('data/processed/X_train.pkl')
y.to_pickle('data/processed/y_train.pkl')
print(f"  ✓ Saved: data/processed/X_train.pkl ({X.shape[0]} samples, {X.shape[1]} features)")
print(f"  ✓ Saved: data/processed/y_train.pkl ({len(y)} targets)")

# Save feature names
with open('data/processed/feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_columns))
print(f"  ✓ Saved: data/processed/feature_names.txt")

# Save data dictionary
with open('data/processed/data_dictionary.txt', 'w') as f:
    f.write("DATA DICTIONARY\n")
    f.write("=" * 80 + "\n\n")
    f.write("STRUCTURE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total records: {len(merged_clean)}\n")
    f.write(f"Unique suburbs: {merged_clean['suburb'].nunique()}\n")
    f.write(f"Bedroom types: {sorted([int(b) for b in merged_clean['bedrooms'].unique()])}\n")
    f.write(f"Property types: {sorted(merged_clean['property_type'].unique())}\n")
    f.write(f"Data aggregation: Last 12 months of available data\n\n")
    
    f.write("FEATURES (X) - {count} total:\n".format(count=len(feature_columns)))
    f.write("-" * 80 + "\n")
    for i, col in enumerate(feature_columns, 1):
        f.write(f"  {i:2d}. {col}\n")
    
    f.write("\nTARGET (y):\n")
    f.write("-" * 80 + "\n")
    f.write("  median_weekly_rent (AUD, weekly) - What the model predicts\n")
    
    f.write("\nKEY FEATURES:\n")
    f.write("-" * 80 + "\n")
    f.write("  1. bedrooms: Numeric (0-5), from 'studio' to '5 bed'\n")
    f.write("  2. property_type: Categorical (one-hot encoded)\n")
    f.write("     - flat/unit (normalized to 'flat')\n")
    f.write("     - house\n")
    f.write("     - townhouse (if present)\n")
    f.write("  3. SEIFA: Socio-economic affluence score (800-1200)\n")
    f.write("  4. income: Median household income weekly (AUD)\n")
    f.write("  5. distance_to_cbd_km: Distance to Melbourne CBD (km)\n")
    f.write("  6. employment_rate: Derived from labour force participation\n\n")
    
    f.write("EXAMPLE PREDICTIONS:\n")
    f.write("-" * 80 + "\n")
    f.write("  Input: Footscray, 2-bed house, income $1200, SEIFA 1020\n")
    f.write("  Output: $450/week\n\n")
    f.write("  Input: Footscray, 2-bed flat, income $1200, SEIFA 1020\n")
    f.write("  Output: $400/week (different property type)\n\n")
    f.write("  Input: Toorak, 2-bed house, income $2100, SEIFA 1180\n")
    f.write("  Output: $800/week (different suburb + income)\n")

print(f"  ✓ Saved: data/processed/data_dictionary.txt")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ PIPELINE COMPLETE!")
print("=" * 80)

print(f"\n📊 SUMMARY:")
print(f"  Records (suburb-bedroom-property combos): {len(merged_clean)}")
print(f"    ({merged_clean['suburb'].nunique()} suburbs × {merged_clean['bedrooms'].nunique():.0f} bedroom types × {merged_clean['property_type'].nunique()} property types)")
print(f"  Features: {len(feature_columns)}")
print(f"  Includes: bedrooms, property_type, income, SEIFA, distance, growth")
print(f"  Data period: Last 12 months (aggregated)")
print(f"  Target: median_weekly_rent (${y.mean():.0f}±${y.std():.0f}/week)")

print(f"\n📁 OUTPUT FILES:")
print(f"  1. data/processed/merged_rental_data.csv")
print(f"     - {len(merged_clean)} records × {len(merged_clean.columns)} columns")
print(f"  2. data/processed/X_train.pkl")
print(f"     - {X.shape[0]} samples × {X.shape[1]} features")
print(f"  3. data/processed/y_train.pkl")
print(f"     - {len(y)} targets")
print(f"  4. data/processed/feature_names.txt")
print(f"  5. data/processed/data_dictionary.txt")

print(f"\n🚀 READY FOR MODEL TRAINING!")
print(f"\nThe model will learn:")
print(f"  Suburb + Bedrooms + Property_Type + Income + SEIFA + Distance → Rent")
print(f"\nExample user flow in Streamlit:")
print(f"  1. User selects: 'Footscray'")
print(f"  2. User selects: '2 bedrooms'")
print(f"  3. User selects: 'House'")
print(f"  4. Model predicts: $450/week")
print(f"\nVs. if they selected 'Flat':")
print(f"  4. Model predicts: $400/week (10% cheaper for flats)")

print("\n" + "=" * 80)
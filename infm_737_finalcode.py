import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

st.set_page_config(page_title="Water Main Repair Predictor", layout="wide")
st.title("Water Main Repair Prediction")

# Upload section
with st.expander("Upload Data", expanded=True):
    repair_file = st.file_uploader("Upload historical main breaks CSV", type="csv")
    mains_file = st.file_uploader("Upload water mains with soil data CSV", type="csv")

if not repair_file or not mains_file:
    st.warning("Please upload both files to proceed.")
    st.stop()

# Load and process data
@st.cache_data
def load_data(mains_file, repair_file):
    mains_df = pd.read_csv(mains_file)
    repairs_df = pd.read_csv(repair_file)

    mains_df.columns = mains_df.columns.str.upper()
    repairs_df.columns = repairs_df.columns.str.upper()

    # Basic cleaning
    mains_df = mains_df.dropna(subset=['ASSETTAG'])
    repairs_df = repairs_df.dropna(subset=['ASSETTAG', 'REPAIRTYPE', 'YEARREPORTED'])
    repairs_df['YEARREPORTED'] = repairs_df['YEARREPORTED'].astype(int)

    # Create AGE and SOIL_PH
    mains_df['INSTALLDATE'] = pd.to_datetime(mains_df['INSTALLDATE'], errors='coerce')
    mains_df['AGE'] = datetime.now().year - mains_df['INSTALLDATE'].dt.year

    soil_ph_map = {
        'Cf': 6.2, 'CfB': 6.0, 'CfC': 5.8,
        'Sp': 5.7, 'SpB': 5.9, 'SpC': 5.5,
        'Ud': 7.0, 'UdB': 7.1, 'UdC': 6.9
    }
    mains_df['SOIL_CODE'] = mains_df['TYPE'].astype(str).str[:3]
    mains_df['SOIL_PH'] = mains_df['SOIL_CODE'].map(soil_ph_map).fillna(6.5)

    return mains_df, repairs_df

mains_df, repairs_df = load_data(mains_file, repair_file)

# Merge and prepare training data
data = pd.merge(repairs_df, mains_df, on='ASSETTAG', how='left')
year_col = [col for col in data.columns if 'YEARREPORTED' in col][0]
data = data.sort_values(['ASSETTAG', year_col])

X_rows, y_type, y_interval = [], [], []
for asset, group in data.groupby('ASSETTAG'):
    group = group.sort_values(year_col)
    for i in range(len(group) - 1):
        current = group.iloc[i]
        next_ = group.iloc[i+1]
        X_rows.append({
            'AGE': current['AGE'] * 0.25,
            'LENGTH': current['LENGTH'] * 0.20,
            'DIAMT': current['DIAMT'] * 0.30,
            'SOIL_PH': current['SOIL_PH'] * 0.40,
            'YEARREPORTED': current[year_col],
            'MATRL': current['MATRL'], 'SLOPEDESC': current['SLOPEDESC'], 'SLOPE': current['SLOPE'],
            'OWNER': current['OWNER'], 'ROUGHSRC': current['ROUGHSRC'],
            'EXTCOATN': current['EXTCOATN'], 'JOINTYP1': current['JOINTYP1'],
            'JOINTYP2': current['JOINTYP2'], 'TYPE': current['TYPE'],
            'WARD': current['WARD'], 'SUBTYPE': current['SUBTYPE']
        })
        repair_col = [col for col in next_.index if 'REPAIRTYPE' in col][0]
        y_type.append(next_[repair_col])
        y_interval.append(next_[year_col] - current[year_col])

X_df = pd.DataFrame(X_rows)
cat_cols = ['MATRL', 'SLOPEDESC', 'SLOPE', 'OWNER', 'ROUGHSRC',
            'EXTCOATN', 'JOINTYP1', 'JOINTYP2', 'TYPE', 'WARD', 'SUBTYPE']
X_encoded = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

le_type = LabelEncoder()
y_type_encoded = le_type.fit_transform(y_type)

# Train models
clf_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
clf_model.fit(X_encoded, y_type_encoded)
reg_model = XGBRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_encoded, y_interval)

# User selection
st.header("Analyze Asset")
asset_tags = sorted(mains_df['ASSETTAG'].dropna().unique())
selected_asset = st.selectbox("Select Asset Tag", asset_tags)

if st.button("Analyze Asset"):
    asset_main = mains_df[mains_df['ASSETTAG'] == selected_asset].iloc[0]
    asset_repairs = repairs_df[repairs_df['ASSETTAG'] == selected_asset]
    last_year = asset_repairs['YEARREPORTED'].max() if not asset_repairs.empty else datetime.now().year

    input_features = pd.DataFrame([{ 
        'AGE': asset_main['AGE'] * 0.25, 'LENGTH': asset_main['LENGTH'] * 0.20,
        'DIAMT': asset_main['DIAMT'] * 0.30, 'SOIL_PH': asset_main['SOIL_PH'] * 0.40,
        'YEARREPORTED': last_year,
        'MATRL': asset_main['MATRL'], 'SLOPEDESC': asset_main.get('SLOPEDESC', ''),
        'SLOPE': asset_main.get('SLOPE', ''), 'OWNER': asset_main['OWNER'],
        'ROUGHSRC': asset_main['ROUGHSRC'], 'EXTCOATN': asset_main.get('EXTCOATN', ''),
        'JOINTYP1': asset_main.get('JOINTYP1', ''), 'JOINTYP2': asset_main.get('JOINTYP2', ''),
        'TYPE': asset_main['TYPE'], 'WARD': asset_main.get('WARD', ''), 'SUBTYPE': asset_main.get('SUBTYPE', '')
    }])

    input_encoded = pd.get_dummies(input_features, columns=cat_cols, drop_first=True)
    for col in X_encoded.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[X_encoded.columns]

    pred_type = le_type.inverse_transform(clf_model.predict(input_encoded))[0]
    pred_interval = int(round(reg_model.predict(input_encoded)[0]))
    pred_year = last_year + pred_interval

    st.success(f"Next predicted repair: **{pred_type}** in approximately **{pred_year}**")

    # Asset Summary
    st.subheader("Asset Summary")
    st.markdown(f"""
    - **Age:** {asset_main['AGE']:.1f} years  
    - **Year Reported:** {last_year}  
    - **Soil pH:** {asset_main['SOIL_PH']:.2f}  
    - **Type:** {asset_main['TYPE']}  
    - **Length:** {asset_main['LENGTH']} ft  
    - **Diameter:** {asset_main['DIAMT']} inches  
    """)

    # Risk Factors
    st.subheader("Risk Factors")
    length_pct = (mains_df['LENGTH'] < asset_main['LENGTH']).mean() * 100
    age_pct = (mains_df['AGE'] < asset_main['AGE']).mean() * 100
    st.markdown(f"""
    1. **Length ({asset_main['LENGTH']:.1f} ft)** – {length_pct:.0f}th percentile risk  
    2. **Age ({asset_main['AGE']:.1f} years)** – {age_pct:.0f}th percentile risk  
    3. **Diameter:** {asset_main['DIAMT']} inches  
    4. **Soil pH:** {asset_main['SOIL_PH']:.2f}  
    5. **Material:** {asset_main['MATRL']}  
    6. **Slope Description:** {asset_main.get('SLOPEDESC', 'N/A')}  
    7. **Slope:** {asset_main.get('SLOPE', 'N/A')}  
    8. **Owner:** {asset_main.get('OWNER', 'N/A')}  
    9. **Roughness Source:** {asset_main.get('ROUGHSRC', 'N/A')}  
    10. **Coating:** {asset_main.get('EXTCOATN', 'N/A')}  
    11. **Joint Type 1:** {asset_main.get('JOINTYP1', 'N/A')}  
    12. **Joint Type 2:** {asset_main.get('JOINTYP2', 'N/A')}  
    13. **Pipe Type:** {asset_main.get('TYPE', 'N/A')}  
    14. **Ward:** {asset_main.get('WARD', 'N/A')}  
    15. **Subtype:** {asset_main.get('SUBTYPE', 'N/A')}  
    """)

    # Download Selected
    if st.button("Download Report for Selected Asset"):
        selected_df = input_features.copy()
        selected_df["Predicted Repair Type"] = pred_type
        selected_df["Predicted Repair Year"] = pred_year
        st.download_button(
            label="Download Selected Asset Report",
            data=selected_df.to_csv(index=False),
            file_name=f"{selected_asset}_repair_prediction.csv",
            mime="text/csv"
        )

    # Download All
    if st.button("Download Report for All Assets"):
        records = []
        for asset in asset_tags:
            row = mains_df[mains_df['ASSETTAG'] == asset].iloc[0]
            asset_rep = repairs_df[repairs_df['ASSETTAG'] == asset]
            last_year = asset_rep['YEARREPORTED'].max() if not asset_rep.empty else datetime.now().year

            inp = {
                'AGE': row['AGE'] * 0.25, 'LENGTH': row['LENGTH'] * 0.20,
                'DIAMT': row['DIAMT'] * 0.30, 'SOIL_PH': row['SOIL_PH'] * 0.40,
                'YEARREPORTED': last_year,
                'MATRL': row['MATRL'], 'SLOPEDESC': row.get('SLOPEDESC', ''),
                'SLOPE': row.get('SLOPE', ''), 'OWNER': row['OWNER'], 'ROUGHSRC': row['ROUGHSRC'],
                'EXTCOATN': row.get('EXTCOATN', ''), 'JOINTYP1': row.get('JOINTYP1', ''),
                'JOINTYP2': row.get('JOINTYP2', ''), 'TYPE': row['TYPE'],
                'WARD': row.get('WARD', ''), 'SUBTYPE': row.get('SUBTYPE', '')
            }

            input_df = pd.DataFrame([inp])
            input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
            for col in X_encoded.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[X_encoded.columns]

            predicted_type = le_type.inverse_transform(clf_model.predict(input_df))[0]
            predicted_interval = int(round(reg_model.predict(input_df)[0]))
            predicted_year = last_year + predicted_interval

            records.append({
                'ASSETTAG': asset,
                'Predicted Repair Type': predicted_type,
                'Predicted Repair Year': predicted_year,
                'Age': row['AGE'], 'Length': row['LENGTH'],
                'Diameter': row['DIAMT'], 'Soil pH': row['SOIL_PH'],
                'Ward': row.get('WARD', ''), 'Subtype': row.get('SUBTYPE', '')
            })

        all_results_df = pd.DataFrame(records)
        st.download_button(
            label="Download All Asset Predictions",
            data=all_results_df.to_csv(index=False),
            file_name="all_asset_repair_predictions.csv",
            mime="text/csv"
        )

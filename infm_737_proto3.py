import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Streamlit config
st.set_page_config(page_title="XGBoost Repair Predictor", layout="wide")
st.title("Water Main Repair Predictor (XGBoost Model)")

# Upload files
with st.expander("Upload Data", expanded=True):
    repair_file = st.file_uploader("Upload historical main breaks CSV", type="csv")
    mains_file = st.file_uploader("Upload water mains with soil data CSV", type="csv")

if not repair_file or not mains_file:
    st.warning("Please upload both files to continue.")
    st.stop()

@st.cache_data
def load_and_prepare_data(mains_file, repair_file):
    mains_df = pd.read_csv(mains_file)
    repairs_df = pd.read_csv(repair_file)

    # Standardize column names
    mains_df.columns = mains_df.columns.str.upper()
    repairs_df.columns = repairs_df.columns.str.upper()

    # Clean and merge
    repairs_df = repairs_df.dropna(subset=['ASSETTAG', 'YEARREPORTED', 'REPAIRTYPE'])
    mains_df = mains_df.dropna(subset=['ASSETTAG'])

    repairs_df['YEARREPORTED'] = repairs_df['YEARREPORTED'].astype(int)
    mains_df['INSTALLDATE'] = pd.to_datetime(mains_df['INSTALLDATE'], errors='coerce')
    mains_df['AGE'] = datetime.now().year - mains_df['INSTALLDATE'].dt.year

    # Add SOIL_PH mapping based on TYPE
    def infer_soil_ph(t):
        if pd.isna(t): return 6.5
        t = str(t)
        return {
            'Cf': 6.2, 'CfB': 6.0, 'CfC': 5.8,
            'Sp': 5.7, 'SpB': 5.9, 'SpC': 5.5,
            'Ud': 7.0, 'UdB': 7.1, 'UdC': 6.9
        }.get(t[:2], 6.5)

    mains_df['SOIL_PH'] = mains_df['TYPE'].apply(infer_soil_ph)

    merged_df = pd.merge(repairs_df, mains_df[['ASSETTAG', 'LENGTH', 'AGE', 'TYPE', 'SOIL_PH']], on='ASSETTAG', how='left')
    return merged_df, mains_df, repairs_df

merged_df, mains_df, repairs_df = load_and_prepare_data(mains_file, repair_file)

@st.cache_data
def train_model(data):
    data = data.dropna(subset=['LENGTH', 'AGE', 'YEARREPORTED', 'REPAIRTYPE'])
    data = data.sort_values(by=['ASSETTAG', 'YEARREPORTED'])

    X = []
    y = []

    for asset, group in data.groupby('ASSETTAG'):
        for i in range(len(group) - 1):
            current = group.iloc[i]
            next_repair = group.iloc[i + 1]
            X.append([current['LENGTH'], current['AGE'], current['YEARREPORTED']])
            y.append(next_repair['REPAIRTYPE'])

    if not X:
        st.error("Not enough repair sequences to train model.")
        st.stop()

    X = np.array(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y_encoded)

    return model, le

model, le = train_model(merged_df)

# UI to select an asset tag
st.header("Asset Analysis")
asset_tags = sorted(mains_df['ASSETTAG'].dropna().unique())
selected_asset = st.selectbox("Select Asset Tag", asset_tags)

if st.button("Analyze Asset"):
    repairs = repairs_df[repairs_df['ASSETTAG'] == selected_asset].sort_values(by='YEARREPORTED')
    static = mains_df[mains_df['ASSETTAG'] == selected_asset].iloc[0]

    result_data = {}

    st.subheader("Prediction Output")
    if not repairs.empty:
        last_repair = repairs.iloc[-1]
        features = np.array([[static.get('LENGTH', np.nan), static.get('AGE', np.nan), last_repair['YEARREPORTED']]])

        if not np.any(np.isnan(features)):
            pred_encoded = model.predict(features)[0]
            pred_label = le.inverse_transform([pred_encoded])[0]

            if len(repairs['YEARREPORTED']) >= 2:
                intervals = np.diff(repairs['YEARREPORTED'])
                avg_interval = int(round(intervals.mean()))
                predicted_year = int(repairs['YEARREPORTED'].iloc[-1] + avg_interval)
            else:
                predicted_year = "Unknown"

            st.success(f"Next predicted repair: **{pred_label}** in approximately **{predicted_year}**")
            result_data.update({"NextRepair": pred_label, "NextYear": predicted_year})
        else:
            st.warning("Insufficient data for prediction.")
    else:
        st.warning("No repair history found for this asset.")

    st.subheader("Risk Factors")
    try:
        length = static.get('LENGTH', np.nan)
        age = static.get('AGE', np.nan)

        length_percentile = (mains_df['LENGTH'] < length).mean() * 100
        age_percentile = (mains_df['AGE'] < age).mean() * 100

        st.markdown(f"""
        1. **Length**: {length} ft – {length_percentile:.0f}th percentile risk
        2. **Age**: {age:.0f} years – {age_percentile:.0f}th percentile risk
        """)

        result_data.update({"Length": length, "LengthPercentile": length_percentile,
                            "Age": age, "AgePercentile": age_percentile})
    except:
        st.warning("Risk factor calculation failed.")

    st.subheader("Asset Snapshot (Display Only)")
    age_display = static.get('AGE', 'N/A')
    year_reported = repairs['YEARREPORTED'].iloc[-1] if not repairs.empty else 'N/A'
    soil_ph_display = static.get('SOIL_PH', 'N/A')
    type_display = static.get('TYPE', 'N/A')

    st.markdown(f"""
    - **Age**: {age_display} years  
    - **Year Reported**: {year_reported}  
    - **Soil pH**: {soil_ph_display}  
    - **Type**: {type_display}  
    """)

    result_data.update({"DisplayAge": age_display, "DisplayYear": year_reported,
                        "SoilPH": soil_ph_display, "Type": type_display})

    # Store results for download
    result_df = pd.DataFrame([result_data])
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Report",
        data=csv,
        file_name=f"asset_{selected_asset}_prediction.csv",
        mime='text/csv',
    )

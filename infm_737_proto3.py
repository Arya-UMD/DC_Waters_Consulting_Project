import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Water Main Repair Predictor", layout="centered")

st.title("Water Main Repair Predictor")

# Upload section
repair_file = st.file_uploader("Upload historical main breaks CSV", type="csv")
mains_file = st.file_uploader("Upload water mains with soil data CSV", type="csv")

if not repair_file or not mains_file:
    st.warning("Please upload both files to continue.")
    st.stop()

@st.cache_data
def load_data(mains_file, repair_file):
    mains_df = pd.read_csv(mains_file)
    repairs_df = pd.read_csv(repair_file)

    # Normalize and clean
    mains_df.columns = [col.upper() for col in mains_df.columns]
    repairs_df.columns = [col.upper() for col in repairs_df.columns]
    mains_df['INSTALLDATE'] = pd.to_datetime(mains_df['INSTALLDATE'], errors='coerce')
    mains_df['AGE'] = datetime.now().year - mains_df['INSTALLDATE'].dt.year
    mains_df['AGE'].fillna(mains_df['AGE'].median(), inplace=True)
    mains_df['LENGTH'].fillna(mains_df['LENGTH'].median(), inplace=True)
    mains_df['DIAMT'].fillna(mains_df['DIAMT'].median(), inplace=True)

    repairs_df = repairs_df.dropna(subset=['YEARREPORTED', 'REPAIRTYPE', 'ASSETTAG'])
    repairs_df['YEARREPORTED'] = repairs_df['YEARREPORTED'].astype(int)

    return mains_df, repairs_df

mains_df, repairs_df = load_data(mains_file, repair_file)

# Train model globally across all assets
@st.cache_data
def train_global_model():
    data = pd.merge(repairs_df, mains_df[['ASSETTAG', 'INSTALLDATE', 'LENGTH', 'DIAMT']],
                    on='ASSETTAG', how='left')
    data['INSTALLYEAR'] = data['INSTALLDATE'].dt.year
    data['AGEATBREAK'] = data['YEARREPORTED'] - data['INSTALLYEAR']
    data.dropna(subset=['AGEATBREAK', 'LENGTH', 'DIAMT', 'REPAIRTYPE'], inplace=True)

    # Sequence modeling
    X_rows = []
    y_labels = []
    for asset, grp in data.groupby('ASSETTAG'):
        grp = grp.sort_values('YEARREPORTED')
        for i in range(len(grp) - 1):
            curr = grp.iloc[i]
            next_rep = grp.iloc[i+1]
            X_rows.append([curr['YEARREPORTED'], curr['REPAIRTYPE'], curr['LENGTH'], curr['AGEATBREAK'], curr['DIAMT']])
            y_labels.append(next_rep['REPAIRTYPE'])

    X_df = pd.DataFrame(X_rows, columns=['YEARREPORTED', 'REPAIRTYPE', 'LENGTH', 'AGE', 'DIAMETER'])
    y_series = pd.Series(y_labels, name='NEXTREPAIRTYPE')

    X_df = pd.get_dummies(X_df, columns=['REPAIRTYPE'])
    all_types = repairs_df['REPAIRTYPE'].unique()
    for t in all_types:
        col = f'REPAIRTYPE_{t}'
        if col not in X_df.columns:
            X_df[col] = 0
    X_df = X_df.reindex(sorted(X_df.columns), axis=1)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_series)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_df, y_encoded)

    return model, le, all_types, X_df.columns

model, le, all_types, training_columns = train_global_model()

asset_tags = mains_df['ASSETTAG'].dropna().unique()
selected_asset = st.selectbox("Select Asset Tag", sorted(asset_tags))
current_year = st.number_input("Current Year", value=datetime.now().year, step=1)

if st.button("Analyze Asset"):
    asset_data = repairs_df[repairs_df['ASSETTAG'] == selected_asset]
    static_data = mains_df[mains_df['ASSETTAG'] == selected_asset].iloc[0]

    st.header("Repair Pattern Analysis")
    if asset_data.empty:
        st.info("No repair history is available for this asset.")
    else:
        asset_data = asset_data.sort_values('YEARREPORTED')
        break_years = asset_data['YEARREPORTED'].tolist()
        repair_types = asset_data['REPAIRTYPE'].dropna().tolist()
        break_count = len(break_years)

        most_common_type = pd.Series(repair_types).mode()[0] if repair_types else "Unknown"
        if break_count >= 2:
            intervals = [j - i for i, j in zip(break_years[:-1], break_years[1:])]
            avg_interval = sum(intervals) / len(intervals)
            predicted_year = int(round(break_years[-1] + avg_interval))
        else:
            avg_interval = None
            predicted_year = "N/A"

        if break_count >= 1:
            last_year = break_years[-1]
            last_type = asset_data['REPAIRTYPE'].iloc[-1]
            length_val = static_data['LENGTH']
            diam_val = static_data['DIAMT']
            install_year = static_data['INSTALLDATE'].year if pd.notna(static_data['INSTALLDATE']) else current_year
            age_at_last = last_year - install_year

            feat_dict = {
                'YEARREPORTED': last_year,
                'LENGTH': length_val,
                'AGE': age_at_last,
                'DIAMETER': diam_val,
            }
            for t in all_types:
                feat_dict[f'REPAIRTYPE_{t}'] = 0
            if f'REPAIRTYPE_{last_type}' in feat_dict:
                feat_dict[f'REPAIRTYPE_{last_type}'] = 1

            X_single = pd.DataFrame([feat_dict])
            X_single = X_single.reindex(columns=training_columns, fill_value=0)
            pred_label = model.predict(X_single)[0]
            predicted_type = le.inverse_transform([pred_label])[0]
        else:
            predicted_type = "N/A"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Most Common Repair Type", most_common_type)
        col2.metric("Predicted Next Repair Type", predicted_type)
        col3.metric("Avg Interval (years)", round(avg_interval, 1) if avg_interval else "N/A")
        col4.metric("Predicted Next Repair Year", predicted_year)

    # Export full results across all assets
if st.button("Download Full Report"):
    result_rows = []
    for asset_id in mains_df['ASSETTAG'].dropna().unique():
        static_row = mains_df[mains_df['ASSETTAG'] == asset_id].iloc[0]
        repair_subset = repairs_df[repairs_df['ASSETTAG'] == asset_id].sort_values('YEARREPORTED')
        break_years = repair_subset['YEARREPORTED'].tolist()
        repair_types = repair_subset['REPAIRTYPE'].dropna().tolist()

        most_common_type = pd.Series(repair_types).mode()[0] if repair_types else "Unknown"
        if len(break_years) >= 2:
            intervals = [j - i for i, j in zip(break_years[:-1], break_years[1:])]
            avg_interval = round(sum(intervals) / len(intervals), 1)
            next_year = int(round(break_years[-1] + avg_interval))
        else:
            avg_interval = None
            next_year = "N/A"

        if len(break_years) >= 1:
            last_year = break_years[-1]
            last_type = repair_subset['REPAIRTYPE'].iloc[-1]
            install_year = static_row['INSTALLDATE'].year if pd.notna(static_row['INSTALLDATE']) else current_year
            age_at_last = last_year - install_year

            feat = {
                'YEARREPORTED': last_year,
                'LENGTH': static_row['LENGTH'],
                'AGE': age_at_last,
                'DIAMETER': static_row['DIAMT']
            }
            for t in all_types:
                feat[f'REPAIRTYPE_{t}'] = 0
            if f'REPAIRTYPE_{last_type}' in feat:
                feat[f'REPAIRTYPE_{last_type}'] = 1

            x_input = pd.DataFrame([feat]).reindex(columns=training_columns, fill_value=0)
            pred_label = model.predict(x_input)[0]
            predicted_type = le.inverse_transform([pred_label])[0]
        else:
            predicted_type = "N/A"

        try:
            age = (datetime.now() - pd.to_datetime(static_row['INSTALLDATE'])).days / 365.25
            age_percentile = (mains_df['INSTALLDATE'].dropna().apply(lambda d: (datetime.now() - pd.to_datetime(d)).days / 365.25) < age).mean() * 100
        except:
            age_percentile = None

        try:
            length_percentile = (mains_df['LENGTH'] < static_row['LENGTH']).mean() * 100
        except:
            length_percentile = None

        result_rows.append({
            'AssetTag': asset_id,
            'MostCommonRepairType': most_common_type,
            'PredictedNextRepairType': predicted_type,
            'AvgInterval': avg_interval,
            'PredictedNextRepairYear': next_year,
            'Length': static_row['LENGTH'],
            'Diameter': static_row['DIAMT'],
            'Age': age,
            'LengthPercentile': round(length_percentile, 1) if length_percentile else "N/A",
            'AgePercentile': round(age_percentile, 1) if age_percentile else "N/A"
        })

    final_df = pd.DataFrame(result_rows)
    csv = final_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="water_main_predictions.csv", mime="text/csv")


# Asset summary
    st.header("Asset Summary")
    st.markdown(f"""
    **Asset:** {selected_asset}  
    **Location:** {static_data.get('QUAD', 'N/A')} Quadrant, Ward {static_data.get('WARD', 'N/A')}  
    **Installation:** {pd.to_datetime(static_data.get('INSTALLDATE')).year if pd.notna(static_data.get('INSTALLDATE')) else 'N/A'}  
    ({int((datetime.now() - pd.to_datetime(static_data.get('INSTALLDATE'))).days / 365.25)} years old)  
    **Material:** {static_data.get('MATRL', 'N/A')}, {static_data.get('DIAMT', 'N/A')}" diameter, {static_data.get('LENGTH', 'N/A')} ft length  
    **Soil:** {static_data.get('SOILTY', 'N/A')}, {static_data.get('SLOPEDESC', 'N/A')}  
    """)

    st.subheader("Top Risk Factors")
    try:
        length_percentile = (mains_df['LENGTH'] < static_data['LENGTH']).mean() * 100
        age = (datetime.now() - pd.to_datetime(static_data.get('INSTALLDATE'))).days / 365.25
        age_percentile = (mains_df['INSTALLDATE'].dropna().apply(lambda d: (datetime.now() - pd.to_datetime(d)).days / 365.25) < age).mean() * 100

        st.markdown(f"""
        1. **Length ({static_data['LENGTH']} ft)** – {length_percentile:.0f}th percentile exposure  
        2. **Age ({int(age)} years)** – {age_percentile:.0f}th percentile risk  
        3. **Diameter ({static_data['DIAMT']}\")** – Higher stress potential  
        """)
    except:
        st.warning("Could not calculate percentiles due to missing values.")

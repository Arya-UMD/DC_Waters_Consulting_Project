import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Initialize Streamlit
st.set_page_config(page_title="Water Main Repair Predictor", layout="wide")
st.title("Water Main Repair Predictor")

# Scientific pH mapping for urban soil types
SOIL_PH_MAP = {
    'Cf': 6.2, 'CfB': 6.0, 'CfC': 5.8,
    'Sp': 5.7, 'SpB': 5.9, 'SpC': 5.5,
    'Ud': 7.0, 'UdB': 7.1, 'UdC': 6.9,
    'OTHER': 6.5
}

# Upload section
with st.expander("Upload Data", expanded=True):
    repair_file = st.file_uploader("Upload historical main breaks CSV", type="csv",
                                   help="Must contain: ASSETTAG, YEARREPORTED, REPAIRTYPE")
    mains_file = st.file_uploader("Upload water mains with soil data CSV", type="csv",
                                  help="Must contain: ASSETTAG, INSTALLDATE, LENGTH, DIAMT, TYPE")

if not repair_file or not mains_file:
    st.warning("Please upload both files to continue.")
    st.stop()

@st.cache_data
def load_data(mains_file, repair_file):
    try:
        mains_df = pd.read_csv(mains_file)
        repairs_df = pd.read_csv(repair_file)

        mains_df.columns = [col.upper() for col in mains_df.columns]
        repairs_df.columns = [col.upper() for col in repairs_df.columns]

        required_mains = ['ASSETTAG', 'INSTALLDATE', 'LENGTH', 'DIAMT', 'TYPE', 'DEPTH']
        required_repairs = ['ASSETTAG', 'YEARREPORTED', 'REPAIRTYPE']
        
        missing_mains = [col for col in required_mains if col not in mains_df.columns]
        missing_repairs = [col for col in required_repairs if col not in repairs_df.columns]
        
        if missing_mains:
            st.error(f"Missing columns in mains data: {', '.join(missing_mains)}")
        if missing_repairs:
            st.error(f"Missing columns in repairs data: {', '.join(missing_repairs)}")
        if missing_mains or missing_repairs:
            st.stop()

        mains_df['INSTALLDATE'] = pd.to_datetime(mains_df['INSTALLDATE'], errors='coerce')
        mains_df['AGE'] = datetime.now().year - mains_df['INSTALLDATE'].dt.year
        mains_df['AGE'].fillna(mains_df['AGE'].median(), inplace=True)
        mains_df['LENGTH'].fillna(mains_df['LENGTH'].median(), inplace=True)
        mains_df['DIAMT'].fillna(mains_df['DIAMT'].median(), inplace=True)
        mains_df['DEPTH'].fillna(mains_df['DEPTH'].median(), inplace=True)

        # --- Add Soil PH mapping here ---
        mains_df['SOIL_CODE'] = mains_df['TYPE'].str[:2]
        mains_df['SOIL_PH'] = mains_df['SOIL_CODE'].map(SOIL_PH_MAP).fillna(SOIL_PH_MAP['OTHER'])

        mains_df['SOIL_ACIDITY'] = np.select(
            [
                mains_df['SOIL_PH'] < 5.5,
                mains_df['SOIL_PH'] < 6.5,
                mains_df['SOIL_PH'] < 7.5,
                mains_df['SOIL_PH'] >= 7.5
            ],
            [
                'Highly acidic',
                'Moderately acidic',
                'Neutral',
                'Alkaline'
            ],
            default='Unknown'
        )
        # --- Done Soil PH Mapping ---

        repairs_df = repairs_df.dropna(subset=['YEARREPORTED', 'REPAIRTYPE', 'ASSETTAG'])
        repairs_df['YEARREPORTED'] = repairs_df['YEARREPORTED'].astype(int)

        return mains_df, repairs_df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

mains_df, repairs_df = load_data(mains_file, repair_file)

@st.cache_data
def train_global_model(mains_df, repairs_df):
    try:
        data = pd.merge(repairs_df, mains_df[['ASSETTAG', 'INSTALLDATE', 'LENGTH', 'DIAMT', 'TYPE', 'DEPTH', 'SOIL_PH']],
                        on='ASSETTAG', how='left')
        data['INSTALLYEAR'] = data['INSTALLDATE'].dt.year
        data['AGEATBREAK'] = data['YEARREPORTED'] - data['INSTALLYEAR']
        data.dropna(subset=['AGEATBREAK', 'LENGTH', 'DIAMT', 'REPAIRTYPE', 'DEPTH', 'SOIL_PH'], inplace=True)

        X_rows = []
        y_labels = []
        for asset, grp in data.groupby('ASSETTAG'):
            grp = grp.sort_values('YEARREPORTED')
            for i in range(len(grp) - 1):
                curr = grp.iloc[i]
                next_rep = grp.iloc[i+1]
                X_rows.append([
                    curr['YEARREPORTED'],
                    curr['LENGTH'],
                    curr['AGEATBREAK'],
                    curr['DIAMT'],
                    curr['TYPE'],
                    curr['DEPTH'],
                    curr['SOIL_PH']
                ])
                y_labels.append(next_rep['REPAIRTYPE'])

        X_df = pd.DataFrame(X_rows, columns=['YEARREPORTED', 'LENGTH', 'AGE', 'DIAMETER', 'PIPE_TYPE', 'DEPTH', 'SOIL_PH'])
        
        X_encoded = pd.get_dummies(X_df, columns=['PIPE_TYPE'])
        y_series = pd.Series(y_labels, name='NEXTREPAIRTYPE')

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_series)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_encoded, y_encoded)

        return model, le, X_encoded.columns, data['REPAIRTYPE'].unique()

    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        st.stop()

model, le, training_columns, all_types = train_global_model(mains_df, repairs_df)

# UI for asset selection
st.header("Asset Analysis")
asset_tags = sorted(mains_df['ASSETTAG'].dropna().unique())
selected_asset = st.selectbox("Select Asset Tag", asset_tags)
current_year = st.number_input("Current Year", value=datetime.now().year, step=1)

if st.button("Analyze Asset"):
    asset_data = repairs_df[repairs_df['ASSETTAG'] == selected_asset]
    static_data = mains_df[mains_df['ASSETTAG'] == selected_asset].iloc[0]

    st.header("Asset Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Asset Tag:** {selected_asset}  
        **Installation Year:** {static_data.get('INSTALLDATE').year if pd.notna(static_data.get('INSTALLDATE')) else 'Unknown'}  
        **Age:** {int((datetime.now() - pd.to_datetime(static_data.get('INSTALLDATE'))).days / 365.25)} years  
        **Type:** {static_data.get('TYPE', 'N/A')}  
        **Material:** {static_data.get('MATRL', 'N/A')}  
        """)
    
    with col2:
        st.markdown(f"""
        **Diameter:** {static_data.get('DIAMT', 'N/A')} inches  
        **Length:** {static_data.get('LENGTH', 'N/A')} ft  
        **Soil pH:** {static_data.get('SOIL_PH', 'N/A')}  
        **Soil Acidity:** {static_data.get('SOIL_ACIDITY', 'N/A')}  
        **Depth:** {static_data.get('DEPTH', 'N/A')} ft  
        """)

    st.subheader("Risk Factors")
    try:
        length_percentile = (mains_df['LENGTH'] < static_data['LENGTH']).mean() * 100
        age = (datetime.now() - pd.to_datetime(static_data.get('INSTALLDATE'))).days / 365.25
        age_percentile = (mains_df['INSTALLDATE'].dropna().apply(
            lambda d: (datetime.now() - pd.to_datetime(d)).days / 365.25) < age).mean() * 100
        soilph_percentile = (mains_df['SOIL_PH'] < static_data['SOIL_PH']).mean() * 100
        depth_percentile = (mains_df['DEPTH'] < static_data['DEPTH']).mean() * 100

        st.markdown(f"""
        1. **Length ({static_data['LENGTH']} ft)** – {length_percentile:.0f}th percentile exposure  
        2. **Age ({int(age)} years)** – {age_percentile:.0f}th percentile risk  
        3. **Diameter ({static_data['DIAMT']}\")** – Higher stress potential  
        4. **Soil pH ({static_data['SOIL_PH']})** – {soilph_percentile:.0f}th percentile soil risk  
        5. **Depth ({static_data['DEPTH']} ft)** – {depth_percentile:.0f}th percentile depth vulnerability  
        """)
    except Exception as e:
        st.warning(f"Could not calculate risk factors: {str(e)}")

    st.subheader("Repair Pattern Analysis")
    if asset_data.empty:
        st.info("No repair history available for this asset.")
    else:
        asset_data = asset_data.sort_values('YEARREPORTED')
        break_years = asset_data['YEARREPORTED'].tolist()
        repair_types = asset_data['REPAIRTYPE'].dropna().tolist()
        
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.scatter(break_years, np.zeros(len(break_years)), s=100, color='red')
        
        for year, rep_type in zip(break_years, repair_types):
            ax.text(year, 0.1, rep_type, ha='center', va='bottom', rotation=45)
        
        ax.set_xlim(min(break_years)-5, current_year+5)
        ax.set_ylim(-0.5, 0.5)
        ax.axvline(current_year, color='blue', linestyle='--')
        ax.text(current_year, -0.4, 'Current', ha='center', color='blue')
        ax.set_yticks([])
        ax.set_title("Repair History Timeline")
        st.pyplot(fig)

        if len(asset_data) >= 1:
            last_repair = asset_data.iloc[-1]
            
            features = {
                'YEARREPORTED': last_repair['YEARREPORTED'],
                'LENGTH': static_data['LENGTH'],
                'AGE': (last_repair['YEARREPORTED'] - static_data['INSTALLDATE'].year),
                'DIAMETER': static_data['DIAMT'],
                'PIPE_TYPE': static_data.get('TYPE', 'Unknown'),
                'DEPTH': static_data.get('DEPTH', np.nan),
                'SOIL_PH': static_data.get('SOIL_PH', np.nan)
            }
            
            input_df = pd.DataFrame([features])
            input_encoded = pd.get_dummies(input_df, columns=['PIPE_TYPE'])
            
            for col in training_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            input_encoded = input_encoded[training_columns]
            
            try:
                pred_label = model.predict(input_encoded)[0]
                predicted_type = le.inverse_transform([pred_label])[0]
                
                if len(break_years) >= 2:
                    intervals = [j - i for i, j in zip(break_years[:-1], break_years[1:])]
                    avg_interval = np.mean(intervals)
                    predicted_year = int(round(break_years[-1] + avg_interval))
                else:
                    avg_interval = None
                    predicted_year = "N/A"
                
                st.success(f"**Prediction:** Next repair likely to be '{predicted_type}' around {predicted_year}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

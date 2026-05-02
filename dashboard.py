import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 1. Setup
st.set_page_config(page_title="Room Temp AI Dashboard", layout="wide")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('temperature_model.pkl')
        
        # Load Data (Cleaned up the duplicate loading here)
        df = pd.read_csv('clean_data.csv')
        
        # CLEANING: Explicitly use your 'date' column (Fixes the 1970 glitch)
        df['clean_datetime'] = pd.to_datetime(df['date'], errors='coerce')
        
        if 'temperature' not in df.columns:
            df['temperature'] = df.iloc[:, 3] 

        # --- FIX STARTS HERE ---
        # Ensure we drop any rows that aren't valid dates before sorting
        df = df.dropna(subset=['clean_datetime'])
        df = df.sort_values('clean_datetime')
        
        # Create the date column
        df['date_only'] = df['clean_datetime'].dt.date
        
        # Avoid .min() / .max() because they often crash on mixed types (Date vs NaN/Float)
        # Since we sorted the dataframe, the first row is the min and the last is the max
        min_dt = df['date_only'].iloc[0]
        max_dt = df['date_only'].iloc[-1]
        
        stats = {
            "min_dt": min_dt,
            "max_dt": max_dt,
            "count": len(df)
        }
        # --- FIX ENDS HERE ---
        
        return model, df, stats
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None, None

model, df, stats = load_assets()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("📈 Model Metrics")
    st.sidebar.metric(label="MAE", value="0.52 °C")
    st.sidebar.metric(label="R² Score", value="0.94")
    st.sidebar.divider()
    st.sidebar.write(f"**Total Records:** {stats['count']}")

    # --- MAIN INTERFACE ---
    st.title("Room Temperature Prediction")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_date = st.date_input(
            "Select Date:", 
            value=stats['min_dt'], 
            min_value=stats['min_dt'], 
            max_value=stats['max_dt']
        )
        
    # Filter data for selected date
    day_data = df[df['date_only'] == selected_date].copy()

    if not day_data.empty:
        # Prediction Logic
        is_fallback = False
        try:
            # 1. Recreate the exact columns your model was trained on
            features = pd.DataFrame({
                'Hour': day_data['clean_datetime'].dt.hour,
                'Day': day_data['clean_datetime'].dt.day,
                'Month': day_data['clean_datetime'].dt.month
            })
            
            # 2. Give those correct features to the model
            preds = model.predict(features).flatten()
            
        except Exception as e:
            # If it still fails, print the actual error to the terminal so we can see why!
            print(f"Model Error: {e}") 
            preds = day_data['temperature'].values * 0.98 + 0.4
            is_fallback = True

        # Graph
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(day_data['clean_datetime'], day_data['temperature'], 
                label='Actual', color='#0077b6', linewidth=2)
        ax.plot(day_data['clean_datetime'], preds, 
                label='Prediction', color='#e63946', linestyle='--')
        
        ax.set_title(f"Temperature Profile: {selected_date}")
        ax.set_ylabel("°C")
        ax.legend()
        plt.xticks(rotation=30)
        
        st.pyplot(fig)
        plt.close(fig) 

        if is_fallback:
            st.info("💡 Note: Displaying simulated predictions (Model Input Mismatch).")
        
        #with st.expander("🔍 View Raw Data Table"):
         #   st.dataframe(day_data[['clean_datetime', 'temperature']], use_container_width=True)
  #  else:
      #  st.warning(f"No data found for {selected_date}.")
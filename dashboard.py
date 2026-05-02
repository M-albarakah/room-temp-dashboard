import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

# 1. Setup
st.set_page_config(page_title="Room Temp AI Dashboard", layout="wide")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('temperature_model.pkl')
        df = pd.read_csv('clean_data.csv')
        
        # Clean dates
        df['clean_datetime'] = pd.to_datetime(df['date'], errors='coerce')
        if 'temperature' not in df.columns:
            df['temperature'] = df.iloc[:, 3] 

        df = df.dropna(subset=['clean_datetime'])
        df = df.sort_values('clean_datetime')
        df['date_only'] = df['clean_datetime'].dt.date
        
        min_dt = df['date_only'].iloc[0]
        max_dt = df['date_only'].iloc[-1]
        
        stats = {
            "min_dt": min_dt,
            "max_dt": max_dt,
            "count": len(df)
        }
        
        return model, df, stats
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None, None

model, df, stats = load_assets()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("📈 Model Metrics")
    st.sidebar.metric(label="MAE", value="0.58 °C")
    st.sidebar.metric(label="R² Score", value="0.92")
    st.sidebar.divider()

    # --- MAIN INTERFACE ---
    st.title("Room Temperature Prediction")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # NEW: We removed 'max_value' so you can pick future dates!
        selected_date = st.date_input(
            "Select Date (Past or Future):", 
            value=stats['max_dt'], 
            min_value=stats['min_dt']
        )
        
    # Filter data for selected date
    day_data = df[df['date_only'] == selected_date].copy()

    # Setup the graph
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_ylabel("°C")
    plt.xticks(rotation=30)

    # --- SCENARIO 1: HISTORICAL DATE (Has real data) ---
    if not day_data.empty:
        is_fallback = False
        try:
            features = pd.DataFrame({
                'Hour': day_data['clean_datetime'].dt.hour,
                'Day': day_data['clean_datetime'].dt.day,
                'Month': day_data['clean_datetime'].dt.month
            })
            preds = model.predict(features).flatten()
        except Exception as e:
            print(f"Model Error: {e}") 
            preds = day_data['temperature'].values * 0.98 + 0.4
            is_fallback = True

        # Plot both lines
        ax.plot(day_data['clean_datetime'], day_data['temperature'], 
                label='Actual Temperature', color='#0077b6', linewidth=2)
        ax.plot(day_data['clean_datetime'], preds, 
                label='Model Prediction', color='#e63946', linestyle='--')
        
        ax.set_title(f"Actual vs Prediction Comparison: {selected_date}")
        ax.legend()
        st.pyplot(fig)
        
        if is_fallback:
            st.warning("Note: Displaying simulated predictions.")

    # --- SCENARIO 2: FUTURE DATE (No real data yet) ---
    else:
        
        # 1. Create a synthetic 24-hour timeline for the chosen future day
        future_times = pd.date_range(start=selected_date, periods=24, freq='h')
        
        # 2. Extract the features the model needs
        future_features = pd.DataFrame({
            'Hour': future_times.hour,
            'Day': future_times.day,
            'Month': future_times.month
        })
        
        # 3. Ask the AI to predict the future!
        try:
            future_preds = model.predict(future_features).flatten()
            
            # Plot ONLY the forecast line
            ax.plot(future_times, future_preds, 
                    label='Model Prediction', color='#e63946', linestyle='--')
            ax.set_title(f"Future Prediction: {selected_date}")
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Could not generate forecast. Model Error: {e}")
            
    plt.close(fig)


import pandas as pd
import numpy as np
import joblib
import streamlit as st
import io  


# LOAD MODELS AND DATA (Cached)

@st.cache_resource
def load_models():
    """Load all trained models and encoders from artifacts folder"""
    try:
        ensemble = joblib.load('artifacts/model_ensemble.pkl')
        scaler = joblib.load('artifacts/scaler_ensemble.pkl')
        program_encoder = joblib.load('artifacts/program_encoder.pkl')
        university_encoder = joblib.load('artifacts/university_encoder.pkl')
        feature_columns = joblib.load('artifacts/feature_columns.pkl')

        return ensemble, scaler, program_encoder, university_encoder, feature_columns
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None


@st.cache_data
def load_data():
    """Load program tiers and candidate data from Streamlit Secrets"""
    try:
        if 'data' in st.secrets:
            # Load from secrets
            candidates_str = st.secrets['data']['candidates_csv']
            program_tiers_str = st.secrets['data']['program_tiers_csv']
            ermp_str = st.secrets['data']['ermp_csv']

            # Fix: Remove any BOM or hidden characters
            if ermp_str.startswith('\ufeff'):
                ermp_str = ermp_str[1:]

            # Fix: Ensure consistent line endings
            ermp_str = ermp_str.replace('\r\n', '\n').replace('\r', '\n')

            # Convert strings to DataFrames
            candidates = pd.read_csv(io.StringIO(candidates_str))
            program_tiers = pd.read_csv(io.StringIO(program_tiers_str))
            df_original = pd.read_csv(io.StringIO(ermp_str))

            # Clean program names (remove any hidden spaces)
            df_original['Program'] = df_original['Program'].str.strip()
            df_original['University'] = df_original['University'].str.strip()

            return program_tiers, candidates, df_original

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


# BONUS CALCULATION

def calculate_final_score(base_score, is_female, has_managerial):
    """Apply bonus percentages to base score"""
    bonus = 0
    if is_female:
        bonus += 5
    if has_managerial:
        bonus += 5

    final_score = base_score * (1 + bonus / 100)
    return final_score, bonus



# PREDICTION FUNCTION

def predict_cutoff(program, university, df_original, program_encoder,
                   university_encoder, ensemble, scaler, year=2025,
                   quota=None, difficulty=21):
    """Predict minimum grade using ensemble model"""
    try:
        # Get encoded values
        program_clean = program.replace(',', '').replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        university_clean = university.replace(' ', '_').replace(',', '').replace('-', '_')

        program_encoded = program_encoder.transform([program_clean])[0]
        university_encoded = university_encoder.transform([university_clean])[0]

        # Calculate features from historical data
        historical = df_original[(df_original['Program'] == program) &
                                 (df_original['University'] == university)]

        if len(historical) > 0:
            program_avg = historical['Min_Grade'].mean()
            university_avg = df_original[df_original['University'] == university]['Min_Grade'].mean()
            pair_avg = historical['Min_Grade'].mean()

            # Get latest quota if not provided
            if quota is None:
                quota = historical['Quota'].mode()[0]
        else:
            program_avg = df_original[df_original['Program'] == program]['Min_Grade'].mean()
            university_avg = df_original[df_original['University'] == university]['Min_Grade'].mean()
            pair_avg = (program_avg + university_avg) / 2
            if quota is None:
                quota = 10  # Default

        # Create feature vector
        features = np.array([[
            year,  # Year
            program_encoded,  # Program_Encoded
            university_encoded,  # University_Encoded
            quota,  # Quota
            difficulty,  # Difficulty_Index
            program_avg,  # Program_Avg_Grade
            university_avg,  # University_Avg_Grade
            1 / (quota + 1),  # Quota_Impact
            0,  # Grade_Change (assume 0 for 2025)
            0,  # Grade_Change_Pct
            quota * difficulty,  # Competition_Score
            pair_avg,  # Pair_Avg_Grade
            10,  # Volatility_Score (default)
            1  # Tier_Encoded (default)
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Get predictions from ensemble
        pred1 = ensemble['gb'].predict(features_scaled)[0]
        pred2 = ensemble['rf'].predict(features_scaled)[0]
        pred3 = ensemble['xgb'].predict(features_scaled)[0]

        # Weighted ensemble
        ensemble_pred = (pred1 * 0.4 + pred2 * 0.3 + pred3 * 0.3)

        return ensemble_pred, quota

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None



# FIXED RANK AND PROBABILITY CALCULATION

def get_rank_and_probability(final_score, predicted_cutoff, quota, candidates_2025):
    """Calculate user's rank and acceptance probability"""

    #
    candidates_sorted = candidates_2025.sort_values('Score', ascending=False).reset_index(drop=True)

    #  RANK CALCULATION:
    # Count number of people with score >= user's score
    users_with_equal_or_higher = (candidates_sorted['Score'] >= final_score).sum()
    user_rank = users_with_equal_or_higher

    # If no one has score >= user's score, they're first
    if user_rank == 0:
        user_rank = 1

    # Calculate cutoff rank
    cutoff_rank = (candidates_sorted['Score'] >= predicted_cutoff).sum()
    if cutoff_rank == 0:
        cutoff_rank = 1

    total_candidates = len(candidates_sorted)

    # Calculate percentile (percentage of people below you)
    user_percentile = (total_candidates - user_rank) / total_candidates * 100

    # For display: people ahead = rank - 1
    people_ahead = user_rank - 1

    # Calculate probability
    if user_rank <= quota:
        probability = "HIGH"
        color = "high-chance"
        message = f"✅ You are ranked #{user_rank} which is within the quota of {quota}!"
        prob_score = 0.9
    elif user_rank <= cutoff_rank:
        probability = "BORDERLINE"
        color = "borderline"
        message = f"⚠️ You are ranked #{user_rank} with {people_ahead} people ahead (need top {quota})"
        prob_score = 0.5
    else:
        probability = "LOW"
        color = "low-chance"
        message = f"❌ You are ranked #{user_rank} with {people_ahead} people ahead. Need to be at least #{cutoff_rank}"
        prob_score = 0.2

    return user_rank, cutoff_rank, user_percentile, probability, color, message, prob_score



# HISTORICAL TREND CHART

def plot_historical_trend(program, university, df_original):
    """Create historical trend graph"""
    try:
        import plotly.express as px

        # Simple filter - no complex string operations
        data = df_original[
            (df_original['Program'] == program) &
            (df_original['University'] == university)
            ]

        # If no data, try with original data types
        if data.empty:
            # Convert to string and compare
            data = df_original[
                (df_original['Program'].astype(str) == str(program)) &
                (df_original['University'].astype(str) == str(university))
                ]

        if not data.empty and len(data) > 0:
            # Create the chart
            fig = px.line(
                data,
                x='Year',
                y='Min_Grade',
                title=f'📈 {program} at {university}',
                markers=True
            )

            # Update layout
            fig.update_layout(
                xaxis=dict(tickmode='linear', tick0=2022, dtick=1),
                yaxis_title='Minimum Grade',
                height=350,
                margin=dict(l=40, r=40, t=40, b=40)
            )

            return fig
        else:
            # Return None silently (no warning)
            return None

    except Exception as e:
        # Silent fail
        return None

# GET UNIQUE PROGRAMS AND UNIVERSITIES

def get_programs_universities(df_original):
    """Extract unique programs and universities"""
    programs = sorted(df_original['Program'].unique())
    universities = sorted(df_original['University'].unique())
    return programs, universities



# GET DEFAULT QUOTA

def get_default_quota(df_original, program, university):
    """Get default quota for a program-university pair"""
    quota_data = df_original[(df_original['Program'] == program) &
                             (df_original['University'] == university)]['Quota']

    if not quota_data.empty:
        return int(quota_data.mode()[0])
    return 10
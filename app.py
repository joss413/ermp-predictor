
import streamlit as st
import pandas as pd
from utils import *


# PAGE CONFIGURATION

st.set_page_config(
    page_title="ERMP Predictor 2025",
    page_icon="👨‍⚕️👩‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-chance {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .borderline {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    .low-chance {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .bonus-badge {
        background-color: #cce5ff;
        color: #004085;
        padding: 0.2rem 0.5rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# LOAD ALL DATA (from utils)

ensemble, scaler, program_encoder, university_encoder, feature_columns = load_models()
program_tiers, candidates_2025, df_original = load_data()


# SIDEBAR

st.markdown('<h1 class="main-header">👨‍⚕️👩‍⚕️ ERMP Predictor 2025</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital.png", width=100)
    st.markdown("## ℹ️ About")
    st.markdown("""
    This predictor uses machine learning to estimate your chances of matching into Ethiopian medical residency programs.

    """)
    st.markdown("## ⚠️ Disclaimer")
    st.markdown("""
  This is a predictive tool based on historical patterns. Actual results may vary.
    """)

    st.markdown("---")
    st.markdown("### 📊 2025 Statistics")
    if candidates_2025 is not None:
        st.metric("Total Candidates", f"{len(candidates_2025):,}")
        # st.metric("Mean Score", f"{candidates_2025['Score'].mean():.1f}")
        # st.metric("Top Score", f"{candidates_2025['Score'].max():.1f}")
        # st.metric("Bottom Score", f"{candidates_2025['Score'].min():.1f}")


# MAIN CONTENT

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Your Details")

    if df_original is not None:
        programs, universities = get_programs_universities(df_original)

        # Input form
        with st.form("prediction_form"):
            base_score = st.number_input(
                "Your Base Exam Score (0-100):",
                min_value=0.0, max_value=100.0, value=70.0, step=0.5
            )

            col_bonus1, col_bonus2 = st.columns(2)
            with col_bonus1:
                is_female = st.checkbox("👩 Female candidate (+5%)")
            with col_bonus2:
                has_managerial = st.checkbox("👔 Managerial experience (+5%)")

            final_score, bonus_total = calculate_final_score(base_score, is_female, has_managerial)

            if bonus_total > 0:
                st.markdown(f"""
                <div class="bonus-badge" style="display: inline-block;">
                    ✨ +{bonus_total}% bonus applied: {base_score:.1f} → {final_score:.1f}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            col_prog, col_uni = st.columns(2)
            with col_prog:
                selected_program = st.selectbox("Select Program:", programs)
            with col_uni:
                selected_university = st.selectbox("Select University:", universities)

            default_quota = get_default_quota(df_original, selected_program, selected_university)
            st.info(f"📋 Typical quota: **{default_quota}** spots")

            use_default = st.checkbox("Use default quota", value=True)
            if use_default:
                quota_2025 = default_quota
            else:
                quota_2025 = st.number_input("Enter 2025 quota:", min_value=1, value=default_quota)

            # st.info("📊 2025 Exam Difficulty: **21% scored >70%**")

            submitted = st.form_submit_button("🔮 Predict My Chances", use_container_width=True)

with col2:
    st.markdown("### 📈 Quick Stats")
    if df_original is not None:
        total_programs = len(df_original['Program'].unique())
        total_universities = len(df_original['University'].unique())

        st.metric("Total Programs", total_programs)
        st.metric("Total Universities", total_universities)
        # st.metric("Model MAE", "2.78 points")


# RESULTS SECTION

if submitted and ensemble is not None and candidates_2025 is not None:
    with st.spinner("Analyzing your chances..."):

        final_score, bonus_total = calculate_final_score(base_score, is_female, has_managerial)

        # Get prediction
        ensemble_pred, quota_used = predict_cutoff(
            selected_program, selected_university, df_original,
            program_encoder, university_encoder, ensemble, scaler,
            quota=quota_2025, difficulty=21
        )

        # Rename for clarity
        predicted_min = ensemble_pred

        if predicted_min is not None:
            # Calculate rank and probability
            user_rank, cutoff_rank, user_percentile, probability, color_class, message, prob_score = get_rank_and_probability(
                final_score, predicted_min, quota_used, candidates_2025
            )

            st.markdown("---")
            st.markdown("## 📊 Your Prediction Results")

            # Main metrics in columns
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)

            with col_res1:
                st.markdown("### 🎯 Predicted Cutoff")
                st.markdown(f"## {predicted_min:.1f}")

            with col_res2:
                st.markdown("### 📍 Your Rank")
                st.markdown(f"## #{user_rank}")
                st.caption(f"Top {user_percentile:.1f}%")

            with col_res3:
                st.markdown("### 🎫 Quota")
                st.markdown(f"## {quota_used}")

            with col_res4:
                st.markdown("### 💯 Your Score")
                st.markdown(f"## {final_score:.1f}")
                if bonus_total > 0:
                    st.caption(f"(+{bonus_total}% bonus)")

            # Probability box
            st.markdown(f"""
            <div class="prediction-box {color_class}">
                <h3>{message}</h3>
                <div style="margin-top: 1rem;">
                    <strong>Chance of Acceptance: {probability}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Progress bar
            st.progress(prob_score)

            # Detailed metrics
            st.markdown("### 📈 Detailed Analysis")
            col_det1, col_det2, col_det3, col_det4 = st.columns(4)

            with col_det1:
                diff = final_score - predicted_min
                st.metric("vs Predicted Cutoff", f"{diff:+.1f}")

            with col_det2:
                people_ahead = user_rank - 1
                st.metric("People Ahead of You", f"{people_ahead}")

            with col_det3:
                st.metric("People at Cutoff", f"{cutoff_rank}")

            with col_det4:
                if cutoff_rank > 0:
                    competition_ratio = cutoff_rank / quota_used
                    st.metric("Competition Ratio", f"{competition_ratio:.1f}:1")

            # Historical trend chart
            st.markdown("### 📉 Historical Trend")
            fig = plot_historical_trend(selected_program, selected_university, df_original)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical data available for this combination")

            # Recommendations
            st.markdown("### 💡 Recommendations")
            if probability == "HIGH":
                st.success("""
                ✅ **You have a strong chance!**
                - Consider this as your top choice
                - Have backup options ready
                """)
            elif probability == "BORDERLINE":
                st.warning("""
                ⚠️ **You're on the borderline**
                - Definitely apply, but have alternatives
                - Consider less competitive programs as backup
                - Your rank is close to the quota
                """)
            else:
                st.error("""
                ❌ **Chances are low for this program**
                - Consider other programs where your rank is stronger
                - Look at programs with larger quotas
                """)


# FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>© 2026 ERMP (Ethiopian Residency Matching Program) Predictor | Created by Yoseph Negash</p>
    <p>Trained on 2022-2024 ERMP Data </p>
    <p style='font-size: 0.8rem;'>⚠️ Disclaimer: This is a predictive tool based on historical patterns. Actual results may vary.</p>
</div>
""", unsafe_allow_html=True)
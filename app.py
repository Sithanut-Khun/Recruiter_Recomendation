import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from typing import cast

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Recruitment Recommender",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
# Dark AI recruitment theme — deep navy with intelligent green/cyan accents
BG_PAGE    = "#07111F"   # page background — deep AI navy
BG_CARD    = "#0E1A2B"   # card surface
BG_INPUT   = "#111D31"   # input backgrounds
BORDER     = "#21324D"   # all borders
BORDER_MED = "#334867"   # stronger border

TEXT_H     = "#F8FAFC"   # headings
TEXT_BODY  = "#D7DEE8"   # body / labels
TEXT_MUTED = "#94A3B8"   # secondary / placeholder

ACCENT     = "#00833E"   # primary recruitment green
ACCENT_LT  = "rgba(0, 131, 62, 0.18)"   # soft green glow/tint

GOLD       = "#22C55E"   # rank #1 — green success
GOLD_BG    = "rgba(34, 197, 94, 0.14)"
SILVER     = "#38BDF8"   # rank #2 — AI cyan
SILVER_BG  = "rgba(56, 189, 248, 0.13)"
BRONZE     = "#A78BFA"   # rank #3 — violet
BRONZE_BG  = "rgba(167, 139, 250, 0.13)"

# Chart bar colors — 3 highlighted + rest
C_RANK1    = "#00C875"   # strong green
C_RANK2    = "#38BDF8"   # cyan
C_RANK3    = "#A78BFA"   # violet
C_REST     = "#243653"   # muted navy for others

FONT       = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
RADIUS     = "12px"
RADIUS_LG  = "18px"

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS_TEMPLATE = """
<!-- CSS styles - ignore Pylance undefined variable warnings -->
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* Base */
html, body, [class*="css"], .stApp {{
    font-family: {FONT};
    background-color: {BG_PAGE};
    color: {TEXT_BODY};
}}

.stApp {{
    background:
        radial-gradient(circle at 8% 5%, rgba(0, 131, 62, 0.25), transparent 30%),
        radial-gradient(circle at 82% 12%, rgba(56, 189, 248, 0.16), transparent 28%),
        linear-gradient(135deg, #07111F 0%, #0A1526 48%, #081B16 100%) !important;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
}}

.block-container {{
    padding-top: 1.2rem !important;
}}

/* Header */
.app-header {{
    position: relative;
    overflow: hidden;
    padding: 1.8rem 1.8rem 1.6rem 1.8rem;
    border: 1px solid {BORDER};
    border-radius: 26px;
    margin-bottom: 2rem;
    background:
        radial-gradient(circle at 8% 20%, rgba(0, 200, 117, 0.28), transparent 28%),
        radial-gradient(circle at 88% 12%, rgba(56, 189, 248, 0.20), transparent 30%),
        linear-gradient(135deg, rgba(14, 26, 43, 0.96), rgba(7, 17, 31, 0.92));
    box-shadow:
        0 24px 60px rgba(0, 0, 0, 0.34),
        inset 0 1px 0 rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
}}

.app-header::before {{
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(148, 163, 184, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148, 163, 184, 0.05) 1px, transparent 1px);
    background-size: 28px 28px;
    opacity: 0.45;
    pointer-events: none;
}}

.app-header-inner {{
    position: relative;
    z-index: 1;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1.5rem;
}}

.app-header-left {{
    flex: 1;
}}

.app-badge {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(135deg, rgba(0, 131, 62, 0.24), rgba(56, 189, 248, 0.10));
    color: #DFFFEA;
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    padding: 11px 18px;
    border-radius: 18px;
    margin-bottom: 0.75rem;
    border: 1px solid rgba(124, 255, 178, 0.28);
    box-shadow: 0 0 28px rgba(0, 200, 117, 0.22);
    line-height: 1.2;
}}

.app-badge-icon {{
    width: 34px;
    height: 34px;
    border-radius: 12px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 200, 117, 0.18);
    border: 1px solid rgba(124, 255, 178, 0.28);
    font-size: 18px;
}}

.app-subtitle {{
    font-size: 14px;
    font-weight: 400;
    color: {TEXT_MUTED};
    margin: 0;
    max-width: 650px;
    line-height: 1.6;
}}

.header-stats {{
    display: flex;
    gap: 0.65rem;
    flex-wrap: wrap;
    justify-content: flex-end;
}}

.header-chip {{
    min-width: 120px;
    padding: 0.75rem 0.9rem;
    border-radius: 16px;
    background: rgba(17, 29, 49, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}}

.header-chip-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    margin-bottom: 0.25rem;
}}

.header-chip-value {{
    font-size: 13px;
    font-weight: 700;
    color: #7CFFB2;
}}

@media (max-width: 900px) {{
    .app-header-inner {{
        flex-direction: column;
        align-items: flex-start;
    }}

    .header-stats {{
        justify-content: flex-start;
    }}

    .app-badge {{
        font-size: 1.25rem;
    }}
}}

/* Section label */
.sec-label {{
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    margin: 0 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {BORDER};
}}

/* Streamlit widget overrides */
div[data-baseweb="select"] > div {{
    background-color: {BG_INPUT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: {RADIUS} !important;
    color: {TEXT_BODY} !important;
    font-family: {FONT} !important;
    font-size: 14px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03) !important;
}}
div[data-baseweb="select"] > div:hover {{
    border-color: {ACCENT} !important;
}}
div[data-baseweb="input"] > div {{
    background-color: {BG_INPUT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: {RADIUS} !important;
    color: {TEXT_BODY} !important;
    font-family: {FONT} !important;
    font-size: 14px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03) !important;
}}
input[type="number"] {{
    font-size: 14px !important;
    color: {TEXT_BODY} !important;
    background: transparent !important;
}}
label[data-testid="stWidgetLabel"] p {{
    color: {TEXT_BODY} !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    font-family: {FONT} !important;
}}

/* Submit button */
.stFormSubmitButton > button {{
    background: linear-gradient(135deg, {ACCENT}, #00A85A) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: {RADIUS} !important;
    font-family: {FONT} !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.25rem !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
}}
.stFormSubmitButton > button {{
    box-shadow: 0 10px 28px rgba(0, 131, 62, 0.28) !important;
}}
.stFormSubmitButton > button:hover {{
    background: #00A85A !important;
    box-shadow: 0 12px 34px rgba(0, 168, 90, 0.35) !important;
}}

/* Rank cards */
.rank-card {{
    background: linear-gradient(135deg, rgba(14,26,43,0.96), rgba(17,29,49,0.88));
    border: 1px solid {BORDER};
    border-radius: {RADIUS_LG};
    padding: 1.1rem 1.2rem 1rem 1.2rem;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 14px 34px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.035);
}}
.rank-pill {{
    flex-shrink: 0;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 600;
    font-family: {FONT};
}}
.rp-1 {{ background: {GOLD_BG};   color: {GOLD};   border: 1.5px solid rgba(34,197,94,0.45); }}
.rp-2 {{ background: {SILVER_BG}; color: {SILVER}; border: 1.5px solid rgba(56,189,248,0.42); }}
.rp-3 {{ background: {BRONZE_BG}; color: {BRONZE}; border: 1.5px solid rgba(167,139,250,0.42); }}
.rank-info {{ flex: 1; min-width: 0; }}
.rank-name {{
    font-size: 14px;
    font-weight: 600;
    color: {TEXT_H};
    margin: 0 0 6px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: {FONT};
}}
.prob-track {{
    background: {BG_INPUT};
    border-radius: 4px;
    height: 5px;
    overflow: hidden;
}}
.prob-fill-1 {{ height: 100%; background: {GOLD};   border-radius: 4px; }}
.prob-fill-2 {{ height: 100%; background: {SILVER}; border-radius: 4px; }}
.prob-fill-3 {{ height: 100%; background: {BRONZE};  border-radius: 4px; }}
.rank-pct {{
    flex-shrink: 0;
    font-size: 18px;
    font-weight: 600;
    color: {TEXT_H};
    font-family: {FONT};
    letter-spacing: -0.01em;
}}

/* Empty state */
.empty-state {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem;
    text-align: center;
}}
.empty-icon {{ font-size: 2.5rem; margin-bottom: 1rem; }}
.empty-text {{
    font-size: 14px;
    color: {TEXT_MUTED};
    font-family: {FONT};
    line-height: 1.6;
}}

/* Chart wrapper */
.chart-wrap {{
    background: linear-gradient(135deg, rgba(14,26,43,0.96), rgba(17,29,49,0.86));
    border: 1px solid {BORDER};
    border-radius: {RADIUS_LG};
    padding: 1rem 0.5rem 0.5rem 0.5rem;
    margin-top: 1rem;
    box-shadow: 0 14px 34px rgba(0,0,0,0.18);
}}

/* Expander */
[data-testid="stExpander"] {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: {RADIUS_LG} !important;
    box-shadow: 0 14px 34px rgba(0,0,0,0.16) !important;
}}
[data-testid="stExpanderToggleIcon"] {{
    color: {TEXT_MUTED} !important;
}}

/* Dark dataframe/table controls */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    border-radius: 14px !important;
    overflow: hidden !important;
}}

div[role="listbox"] {{
    background-color: #0E1A2B !important;
    border: 1px solid #21324D !important;
}}

div[role="option"] {{
    color: #D7DEE8 !important;
}}

/* Subtle AI divider glow */
.sec-label {{
    border-image: linear-gradient(90deg, rgba(0,131,62,0.8), rgba(56,189,248,0.18), transparent) 1;
}}

</style>
"""
CSS_STYLES = CSS_TEMPLATE.format(
    FONT=FONT,
    BG_PAGE=BG_PAGE,
    TEXT_BODY=TEXT_BODY,
    ACCENT_LT=ACCENT_LT,
    BORDER=BORDER,
    RADIUS=RADIUS,
    RADIUS_LG=RADIUS_LG,
    BG_CARD=BG_CARD,
    TEXT_MUTED=TEXT_MUTED,
    BG_INPUT=BG_INPUT,
    TEXT_H=TEXT_H,
    ACCENT=ACCENT,
    GOLD_BG=GOLD_BG,
    GOLD=GOLD,
    SILVER_BG=SILVER_BG,
    SILVER=SILVER,
    BRONZE_BG=BRONZE_BG,
    BRONZE=BRONZE,
)
st.markdown(CSS_STYLES, unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_dir = 'models'
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder_target.pkl'))
    onehot_encoder = joblib.load(os.path.join(model_dir, 'onehot_encoder.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'robust_scaler.pkl'))
    models = {}
    model_files = ['LightGBM.pkl', 'XGBoost.pkl', 'Logistic_Regression.pkl', 'Random_Forest.pkl']
    model_names = ['LightGBM', 'XGBoost', 'Logistic Regression', 'Random Forest']
    for file, name in zip(model_files, model_names):
        path = os.path.join(model_dir, file)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return feature_names, label_encoder, onehot_encoder, scaler, models


def preprocess_input(categorical_values, numeric_values, onehot_encoder, scaler, feature_names):
    cat_df = pd.DataFrame([{
        'Business_Unit_Anon':   categorical_values['Business_Unit_Anon'],
        'JobType_Anon':         categorical_values['JobType_Anon'],
        'JobLevel_Anon':        categorical_values['JobLevel_Anon'],
        'JobPriority':          categorical_values['JobPriority'],
        'job_group':            categorical_values['job_group'],
        'ReasonToRecruit_Anon': categorical_values['ReasonToRecruit_Anon'],
    }])
    cat_encoded = onehot_encoder.transform(cat_df)
    num_df = pd.DataFrame([[
        numeric_values['NumberOfOpening'],
        numeric_values['HireDuration'],
        numeric_values['TimeToFill'],
    ]], columns=['NumberOfOpening', 'HireDuration', 'TimeToFill'])
    num_scaled = scaler.transform(num_df)
    return np.hstack([cat_encoded, num_scaled])


def get_categories_from_encoder(enc):
    return {feat: enc.categories_[i].tolist() for i, feat in enumerate(enc.feature_names_in_)}


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
<div class="app-header-inner">
<div class="app-header-left">
<div class="app-badge"><span class="app-badge-icon">🤖</span> AI Recruitment Recommender System</div>
<p class="app-subtitle">Predict the best recruiter-job fit using machine learning recommendations, confidence scoring, and recruiter matching intelligence.</p>
</div>
<div class="header-stats">
<div class="header-chip">
<div class="header-chip-label">Engine</div>
<div class="header-chip-value">ML Powered</div>
</div>
<div class="header-chip">
<div class="header-chip-label">Output</div>
<div class="header-chip-value">Top-3 Match</div>
</div>
<div class="header-chip">
<div class="header-chip-label">Metric</div>
<div class="header-chip-value">Probability</div>
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)
# ── Load models ────────────────────────────────────────────────────────────────
with st.spinner("Loading models…"):
    try:
        feature_names, label_encoder, onehot_encoder, scaler, models = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

categories_dict = get_categories_from_encoder(onehot_encoder)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_form, col_results = st.columns([1, 1.4], gap="large")

# ── Form ───────────────────────────────────────────────────────────────────────
with col_form:
    with st.form("prediction_form"):
        st.markdown('<div class="sec-label">Job Details</div>', unsafe_allow_html=True)

        jd1, jd2 = st.columns(2)
        with jd1:
            business_unit = st.selectbox("Business Unit", categories_dict['Business_Unit_Anon'])
        with jd2:
            job_type = st.selectbox("Job Type", categories_dict['JobType_Anon'])

        jd3, jd4 = st.columns(2)
        with jd3:
            job_level = st.selectbox("Job Level", categories_dict['JobLevel_Anon'])
        with jd4:
            job_priority = st.selectbox("Job Priority", categories_dict['JobPriority'])

        jd5, jd6 = st.columns(2)
        with jd5:
            job_group = st.selectbox("Job Group", categories_dict['job_group'])
        with jd6:
            reason_to_recruit = st.selectbox("Reason to Recruit", categories_dict['ReasonToRecruit_Anon'])

        st.markdown(
            '<div class="sec-label" style="margin-top:1.25rem">Quantitative Metrics</div>',
            unsafe_allow_html=True
        )

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            num_openings = st.number_input("Openings", min_value=1, value=1, step=1)
        with mc2:
            hire_duration = st.number_input("Hire Duration (d)", min_value=1, value=30, step=1)
        with mc3:
            time_to_fill = st.number_input("Time to Fill (d)", min_value=1, value=45, step=1)

        st.markdown(
            '<div class="sec-label" style="margin-top:1.25rem">Model Selection</div>',
            unsafe_allow_html=True
        )

        model_choice = st.selectbox("Prediction Model", list(models.keys()))

        submitted = st.form_submit_button(
            "Find Best Recruiters →",
            use_container_width=True
        )

# ── Results ────────────────────────────────────────────────────────────────────
with col_results:
    if not submitted:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🎯</div>
          <div class="empty-text">
            Fill in the job details on the left<br>and click <strong>Find Best Recruiters</strong> to see results.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cat_vals = {
            'Business_Unit_Anon':   business_unit,
            'JobType_Anon':         job_type,
            'JobLevel_Anon':        job_level,
            'JobPriority':          job_priority,
            'job_group':            job_group,
            'ReasonToRecruit_Anon': reason_to_recruit,
        }
        num_vals = {
            'NumberOfOpening': num_openings,
            'HireDuration':    hire_duration,
            'TimeToFill':      time_to_fill,
        }

        try:
            X_input = preprocess_input(cat_vals, num_vals, onehot_encoder, scaler, feature_names)
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

        try:
            probabilities = models[model_choice].predict_proba(X_input)[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        top3_idx        = np.argsort(probabilities)[-3:][::-1]
        top3_probs      = probabilities[top3_idx]
        top3_recruiters = label_encoder.inverse_transform(top3_idx)

        # ── Rank cards ──────────────────────────────────────────────────────
        st.markdown('<div class="sec-label">Top Recommendations</div>', unsafe_allow_html=True)

        pill_cls  = ['rp-1', 'rp-2', 'rp-3']
        fill_cls  = ['prob-fill-1', 'prob-fill-2', 'prob-fill-3']
        rank_lbl  = ['#1', '#2', '#3']

        for recruiter, prob, pc, fc, rl in zip(
            top3_recruiters, top3_probs, pill_cls, fill_cls, rank_lbl
        ):
            bar_pct = prob * 100
            st.markdown(f"""
            <div class="rank-card">
              <div class="rank-pill {pc}">{rl}</div>
              <div class="rank-info">
                <div class="rank-name">{recruiter}</div>
                <div class="prob-track">
                  <div class="{fc}" style="width:{bar_pct:.1f}%"></div>
                </div>
              </div>
              <div class="rank-pct">{prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Plotly horizontal bar chart ─────────────────────────────────────
        st.markdown(
            '<div class="sec-label" style="margin-top:1.5rem">Probability Distribution — All Recruiters</div>',
            unsafe_allow_html=True
        )

        all_recruiters = label_encoder.classes_

        # Sort all recruiters by probability
        sort_idx = np.argsort(probabilities)[::-1]
        sorted_names = all_recruiters[sort_idx]
        sorted_probs = probabilities[sort_idx]

        # Coloring: top 3 highlighted, others muted
        bar_colors = []
        for name in sorted_names:
            if name == top3_recruiters[0]:
                bar_colors.append(C_RANK1)
            elif name == top3_recruiters[1]:
                bar_colors.append(C_RANK2)
            elif name == top3_recruiters[2]:
                bar_colors.append(C_RANK3)
            else:
                bar_colors.append(C_REST)

        # Reverse so highest appears at the top
        names_plot = sorted_names[::-1]
        probs_plot = sorted_probs[::-1]
        colors_plot = bar_colors[::-1]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=names_plot,
            x=probs_plot,
            orientation='h',
            marker=dict(
                color=colors_plot,
                line=dict(width=0)
            ),
            text=[f"{p:.1%}" for p in probs_plot],
            textposition='outside',
            textfont=dict(
                color=TEXT_MUTED,
                size=12,
                family=FONT
            ),
            hovertemplate='<b>%{y}</b><br>Match score: %{x:.2%}<extra></extra>',
            cliponaxis=False,
        ))

        fig.update_layout(
            paper_bgcolor=BG_CARD,
            plot_bgcolor=BG_CARD,
            margin=dict(l=10, r=70, t=12, b=10),

            # Increase height based on number of recruiters
            height=max(420, 28 * len(sorted_names)),

            font=dict(
                family=FONT,
                size=13,
                color=TEXT_BODY
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor=BORDER,
                gridwidth=1,
                zeroline=False,
                tickformat='.0%',
                tickfont=dict(
                    color=TEXT_MUTED,
                    size=12,
                    family=FONT
                ),
                showline=False,
                range=[0, min(1.0, max(sorted_probs) * 1.4)],
            ),
            yaxis=dict(
                tickfont=dict(
                    color=TEXT_BODY,
                    size=12,
                    family=FONT
                ),
                showgrid=False,
                showline=False,
                ticklabelposition='outside',
            ),
            showlegend=False,
            bargap=0.35,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False}
        )
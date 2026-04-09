"""
AREIS — Autonomous Real Estate Interactive System
Bangalore Property Market  |  2021–2025 + Predictions
Streamlit + Plotly + Folium  |  Team Iris
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster
import streamlit.components.v1 as components
import sys
import os

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="AREIS — Autonomous Real Estate Interactive System",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import RealEstatePipeline

# ── THEME ────────────────────────────────────────────────────────
THEME = {
    "bg_dark":   "#07091A",
    "bg_card":   "rgba(14, 20, 44, 0.90)",
    "bg_glass":  "rgba(255,255,255,0.04)",
    "accent":    "#3ECFB2",
    "accent2":   "#6E9EFF",
    "gold":      "#C9A96E",
    "text_main": "#F0F4FF",
    "text_sub":  "#8A95B0",
    "border":    "rgba(62,207,178,0.18)",
}

CATEGORY_COLORS = {
    "Premium":        "#C9A96E",
    "High Growth":    "#3ECFB2",
    "High Potential": "#6E9EFF",
    "Stable":         "#A0AEC0",
    "Budget":         "#F6A35A",
}

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit, sans-serif", color="#F0F4FF", size=12),
    colorway=[THEME["accent"], THEME["accent2"], THEME["gold"], "#F6A35A", "#A0AEC0"],
    xaxis=dict(gridcolor="rgba(138,149,176,0.14)", linecolor="rgba(138,149,176,0.2)", tickfont=dict(color="#C0CADF")),
    yaxis=dict(gridcolor="rgba(138,149,176,0.14)", linecolor="rgba(138,149,176,0.2)", tickfont=dict(color="#C0CADF")),
    legend1=dict(
        bgcolor="rgba(10,15,35,0.80)",
        bordercolor="rgba(62,207,178,0.25)",
        borderwidth=1,
        font=dict(color="#F0F4FF", size=11),
    ),
)

AGENT_INFO = [
    {
        "id": "01", "name": "Data Ingestion Agent", "type": "Data Pipeline",
        "color": "#3ECFB2",
        "desc": "Loads raw property CSV data or generates synthetic Bangalore micro-market data. Validates schema integrity across 17 required columns.",
        "outputs": ["raw_df (DataFrame)", "17 validated columns", "CSV or synthetic fallback"],
    },
    {
        "id": "02", "name": "Data Cleaning Agent", "type": "Preprocessing",
        "color": "#6E9EFF",
        "desc": "Removes duplicates, coerces numeric types, fills missing values with medians, applies IQR outlier filtering on price/sqft, and validates geographic coordinates.",
        "outputs": ["clean_df", "Outlier rows removed", "Coordinate-validated records"],
    },
    {
        "id": "03", "name": "EDA Agent", "type": "Analysis",
        "color": "#C9A96E",
        "desc": "Computes market-level statistics: average prices, YoY growth rates per year pair, zone distributions, metro access coverage, and location-level summaries.",
        "outputs": ["eda_summary dict", "YoY growth 2021–2025", "Zone & property-type breakdowns"],
    },
    {
        "id": "04", "name": "Feature Engineering Agent", "type": "ML Preprocessing",
        "color": "#F6A35A",
        "desc": "Derives 12+ engineered features: price CAGR, 2-year and 4-year momentum, total value in lakhs, connectivity score (metro + IT hub + airport), and normalised variants.",
        "outputs": ["price_cagr", "connectivity_score", "value_per_bhk", "12 engineered features"],
    },
    {
        "id": "05", "name": "Prediction Agent", "type": "Machine Learning",
        "color": "#E879F9",
        "desc": "Trains three independent Gradient Boosting Regressors (200 estimators, LR=0.08) for 1yr, 3yr, and 5yr price forecasts. Reports MAPE per horizon on 20% hold-out.",
        "outputs": ["predicted_price_1yr", "predicted_price_3yr", "predicted_price_5yr", "MAPE metrics"],
    },
    {
        "id": "06", "name": "Investment Analysis Agent", "type": "Scoring",
        "color": "#34D399",
        "desc": "Computes ROI projections (1/3/5yr), a composite growth rate index, and a 0–100 investment score weighted by growth (35%), connectivity (25%), ROI (25%), value (15%). Classifies into 5 categories.",
        "outputs": ["roi_1yr/3yr/5yr", "investment_score", "category", "growth_rate_index"],
    },
    {
        "id": "07", "name": "Insight Generation Agent", "type": "NLP",
        "color": "#F59E0B",
        "desc": "Generates structured natural-language summaries for each micro-market using category-specific templates. Identifies the best overall investment city by composite score.",
        "outputs": ["location_insight (text)", "best_city", "best_city_score"],
    },
    {
        "id": "08", "name": "Visualization Prep Agent", "type": "Dashboard",
        "color": "#60A5FA",
        "desc": "Aggregates all per-property data to location-level summaries, prepares zone heatmap data, category distributions, top-20 investment ranking, and 5-year price trends.",
        "outputs": ["loc_agg", "zone_agg", "cat_dist", "top_investments", "price_trend"],
    },
]


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif !important;
        background-color: {THEME['bg_dark']};
        color: {THEME['text_main']};
    }}
    .stApp {{
        background: radial-gradient(ellipse at 20% 10%, rgba(62,207,178,0.07) 0%, transparent 50%),
                    radial-gradient(ellipse at 80% 80%, rgba(110,158,255,0.06) 0%, transparent 50%),
                    {THEME['bg_dark']};
    }}

    /* ── AREIS HEADER ── */
    .areis-header {{
        background: linear-gradient(135deg, rgba(14,20,44,0.97) 0%, rgba(7,9,26,0.98) 100%);
        border-bottom: 1px solid {THEME['border']};
        padding: 0.85rem 2rem;
        margin: -1rem -1rem 1.5rem -1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        z-index: 100;
        backdrop-filter: blur(16px);
    }}
    .areis-brand {{
        display: flex;
        align-items: baseline;
        gap: 0.75rem;
    }}
    .areis-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        background: linear-gradient(90deg, {THEME['accent']}, {THEME['accent2']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .areis-full {{
        font-family: 'Outfit', sans-serif;
        font-size: 0.78rem;
        font-weight: 500;
        color: {THEME['text_sub']};
        letter-spacing: 0.05em;
    }}
    .areis-badge {{
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {THEME['accent']};
        background: rgba(62,207,178,0.1);
        border: 1px solid rgba(62,207,178,0.3);
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
    }}

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {{
        background: rgba(7,9,26,0.98) !important;
        border-right: 1px solid {THEME['border']};
    }}
    section[data-testid="stSidebar"] * {{
        color: {THEME['text_main']} !important;
        font-family: 'Outfit', sans-serif !important;
    }}

    /* ── SIDEBAR TOGGLE BUTTON — HIGH VISIBILITY ── */
    button[data-testid="stBaseButton-headerNoPadding"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {{
        background: {THEME['accent']} !important;
        border-radius: 0 8px 8px 0 !important;
        color: #07091A !important;
        width: 28px !important;
        opacity: 1 !important;
        border: none !important;
        box-shadow: 2px 0 12px rgba(62,207,178,0.4) !important;
    }}
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg {{
        color: #07091A !important;
        fill: #07091A !important;
    }}

    /* ── SELECTBOXES / MULTISELECT ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background: rgba(14,20,44,0.9) !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 8px !important;
        color: {THEME['text_main']} !important;
        font-family: 'Outfit', sans-serif !important;
    }}
    .stSelectbox label, .stMultiSelect label, .stRadio label {{
        color: {THEME['text_main']} !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.85rem !important;
    }}

    /* ── KPI CARDS ── */
    .kpi-card {{
        background: {THEME['bg_card']};
        backdrop-filter: blur(12px);
        border: 1px solid {THEME['border']};
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 0.5rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }}
    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, {THEME['accent']}, {THEME['accent2']});
    }}
    .kpi-card:hover {{ transform: translateY(-2px); border-color: rgba(62,207,178,0.4); }}
    .kpi-label {{
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
        text-transform: uppercase; color: {THEME['text_sub']}; margin-bottom: 0.35rem;
    }}
    .kpi-value {{
        font-family: 'Space Grotesk', sans-serif; font-size: 1.75rem;
        font-weight: 700; color: {THEME['text_main']}; line-height: 1;
    }}
    .kpi-delta {{ font-size: 0.76rem; color: {THEME['accent']}; margin-top: 0.3rem; font-weight: 500; }}

    /* ── SECTION HEADERS ── */
    .section-header {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem; font-weight: 600;
        color: {THEME['text_main']}; letter-spacing: 0.02em;
        margin: 0 0 0.9rem 0; padding-bottom: 0.45rem;
        border-bottom: 1px solid {THEME['border']};
    }}
    .section-accent {{ color: {THEME['accent']}; }}

    /* ── GLASS PANELS ── */
    .glass-panel {{
        background: {THEME['bg_glass']}; backdrop-filter: blur(10px);
        border: 1px solid {THEME['border']}; border-radius: 16px;
        padding: 1.4rem; margin-bottom: 1rem;
    }}

    /* ── INSIGHT BOX ── */
    .insight-box {{
        background: linear-gradient(135deg, rgba(62,207,178,0.08) 0%, rgba(110,158,255,0.06) 100%);
        border: 1px solid rgba(62,207,178,0.25);
        border-left: 3px solid {THEME['accent']};
        border-radius: 10px; padding: 1rem 1.2rem;
        margin: 0.6rem 0; font-size: 0.87rem; line-height: 1.65;
        color: {THEME['text_main']};
    }}

    /* ── BEST CITY BADGE ── */
    .best-city-badge {{
        background: linear-gradient(135deg, {THEME['gold']}22, {THEME['gold']}11);
        border: 1px solid {THEME['gold']}55; border-radius: 12px;
        padding: 1rem 1.2rem; text-align: center; margin-bottom: 1rem;
    }}
    .badge-label {{ font-size: 0.62rem; letter-spacing: 0.15em; text-transform: uppercase; color: {THEME['gold']}; font-weight: 600; }}
    .badge-city {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 700; color: {THEME['gold']}; margin-top: 0.2rem; }}

    /* ── AGENT CARD ── */
    .agent-card {{
        background: rgba(14,20,44,0.85); border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: 0.75rem;
        transition: border-color 0.2s;
    }}
    .agent-card:hover {{ border-color: rgba(62,207,178,0.3); }}

    /* ── LOG TERMINAL ── */
    .log-terminal {{
        background: #030509; border: 1px solid rgba(62,207,178,0.2);
        border-radius: 10px; padding: 1rem 1.2rem;
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
        font-size: 0.78rem; line-height: 1.7; color: #A8FFD8;
        white-space: pre-wrap; overflow-x: auto;
        max-height: 480px; overflow-y: auto;
    }}

    /* ── DATA EXPLORER TABLE ── */
    .stDataFrame thead tr th {{
        background: rgba(62,207,178,0.12) !important;
        color: {THEME['accent']} !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
    }}
    .stDataFrame {{ border-radius: 12px; overflow: hidden; }}

    /* ── FOOTER ── */
    .team-footer {{
        border-top: 1px solid {THEME['border']};
        margin-top: 3rem; padding-top: 1.5rem;
        text-align: center;
    }}
    .team-footer-title {{
        font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase;
        color: {THEME['text_sub']}; margin-bottom: 0.6rem; font-weight: 600;
    }}
    .team-names {{
        font-family: 'Space Grotesk', sans-serif; font-size: 0.88rem;
        font-weight: 500; color: {THEME['text_main']};
    }}
    .team-tag {{
        font-size: 0.72rem; color: {THEME['accent']}; font-weight: 600;
        letter-spacing: 0.1em; margin-top: 0.3rem;
    }}

    /* ── STREAMLIT OVERRIDES ── */
    .stMetric {{ background: {THEME['bg_card']}; border: 1px solid {THEME['border']}; border-radius: 12px; padding: 0.8rem 1rem; }}
    .stMetric label {{ color: {THEME['text_sub']} !important; font-size: 0.75rem !important; font-family: 'Outfit', sans-serif !important; }}
    .stMetric [data-testid="stMetricValue"] {{ color: {THEME['text_main']} !important; font-family: 'Space Grotesk', sans-serif !important; }}
    h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif; color: {THEME['text_main']}; }}
    .stProgress > div > div > div {{ background-color: {THEME['accent']} !important; }}
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(14,20,44,0.7); border-radius: 10px; padding: 0.25rem; gap: 0.25rem; border: 1px solid {THEME['border']};
    }}
    .stTabs [data-baseweb="tab"] {{ border-radius: 8px; color: {THEME['text_sub']}; font-weight: 500; padding: 0.4rem 1rem; font-family: 'Outfit', sans-serif; }}
    .stTabs [aria-selected="true"] {{ background: rgba(62,207,178,0.15) !important; color: {THEME['accent']} !important; }}
    ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(62,207,178,0.3); border-radius: 2px; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    div[data-testid="stHorizontalBlock"] {{ gap: 0.8rem; }}
    p, div, span, li {{ color: {THEME['text_main']}; }}
    </style>
    """, unsafe_allow_html=True)


# ── DATA LOADING ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_pipeline_data(filepath: str):
    pipeline = RealEstatePipeline(filepath=filepath)
    return pipeline.run()


# ── COMPONENTS ───────────────────────────────────────────────────
def render_header(current_page: str = ""):
    st.markdown(f"""
    <div class="areis-header">
        <div class="areis-brand">
            <span class="areis-title">AREIS</span>
            <span class="areis-full">Autonomous Real Estate Interactive System</span>
        </div>
        <span class="areis-badge">Bangalore · 2021–2025</span>
    </div>
    """, unsafe_allow_html=True)


def team_footer():
    st.markdown("""
    <div class="team-footer">
        <div class="team-footer-title">Built by</div>
        <div class="team-names">
            Christo Savio George &nbsp;·&nbsp; Karthik NH &nbsp;·&nbsp; Vamshi MR &nbsp;·&nbsp; Varshith Raj B
        </div>
        <div class="team-tag">Team Iris</div>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = None, col=None):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    html = f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>"""
    if col:
        col.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def section_header(title: str, accent: str = None):
    accent_html = f' <span class="section-accent">{accent}</span>' if accent else ""
    st.markdown(f'<div class="section-header">{title}{accent_html}</div>', unsafe_allow_html=True)


# ── FOLIUM MAP ────────────────────────────────────────────────────
def build_folium_map(loc_agg: pd.DataFrame, filter_type: str = "All") -> str:
    cat_colors_hex = {
        "Premium":        "#C9A96E",
        "High Growth":    "#3ECFB2",
        "High Potential": "#6E9EFF",
        "Stable":         "#A0AEC0",
        "Budget":         "#F6A35A",
    }

    # Apply filter
    display_df = loc_agg.copy()
    if filter_type == "Top Premium":
        display_df = display_df[display_df["dominant_category"] == "Premium"].nlargest(15, "avg_investment_score")
    elif filter_type == "Top Affordable":
        display_df = display_df.nsmallest(15, "avg_price_per_sqft")
    elif filter_type == "High Growth":
        display_df = display_df[display_df["dominant_category"].isin(["High Growth", "High Potential"])].nlargest(15, "avg_growth")

    center_lat = display_df["lat"].mean() if not display_df.empty else 12.97
    center_lon = display_df["lon"].mean() if not display_df.empty else 77.59

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles=None,
        prefer_canvas=True,
    )

    # Dark tile
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="Dark",
        max_zoom=19,
    ).add_to(m)

    cluster = MarkerCluster(
        options={
            "maxClusterRadius": 50,
            "spiderfyOnMaxZoom": True,
            "showCoverageOnHover": False,
            "zoomToBoundsOnClick": True,
        }
    ).add_to(m)

    for _, row in display_df.iterrows():
        cat = row["dominant_category"]
        color = cat_colors_hex.get(cat, "#A0AEC0")
        score = row["avg_investment_score"]
        radius = 6 + (score / 100) * 8  # scale by score

        popup_html = f"""
        <div style="
            font-family:'DM Sans',sans-serif;
            background:#0d1424;
            color:#E8EDF5;
            border-radius:14px;
            padding:0;
            width:310px;
            box-shadow:0 8px 32px rgba(0,0,0,0.6);
            border:1px solid rgba(62,207,178,0.3);
            overflow:hidden;
        ">
            <div style="background:linear-gradient(135deg,rgba(62,207,178,0.15),rgba(110,158,255,0.1));
                        padding:14px 16px 10px 16px; border-bottom:1px solid rgba(62,207,178,0.2);">
                <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;
                            color:#3ECFB2;font-weight:600;margin-bottom:4px;">{row['zone']} Zone</div>
                <div style="font-size:1.25rem;font-weight:700;color:#E8EDF5;">{row['location']}</div>
                <span style="background:{color}22;color:{color};border:1px solid {color}55;
                             font-size:0.68rem;font-weight:600;padding:2px 10px;border-radius:20px;
                             letter-spacing:0.05em;">{cat}</span>
            </div>
            <div style="padding:12px 16px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">
                <div style="text-align:center;">
                    <div style="font-size:0.62rem;color:#8A95B0;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Price/sqft</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#E8EDF5;">Rs {int(row['avg_price_per_sqft']):,}</div>
                </div>
                <div style="text-align:center;border-left:1px solid rgba(255,255,255,0.08);border-right:1px solid rgba(255,255,255,0.08);">
                    <div style="font-size:0.62rem;color:#8A95B0;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Inv. Score</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#3ECFB2;">{row['avg_investment_score']:.1f}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:0.62rem;color:#8A95B0;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">ROI 5Y</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#6E9EFF;">{row['avg_roi_5yr']:.1f}%</div>
                </div>
            </div>
            <div style="padding:0 16px 12px 16px;">
                <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px 12px;">
                    <div style="font-size:0.72rem;color:#8A95B0;margin-bottom:6px;">ROI Forecast</div>
                    <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
                        <span style="color:#A0AEC0;">1 Year</span>
                        <span style="color:#3ECFB2;font-weight:600;">{row['avg_roi_1yr']:.1f}%</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-top:4px;">
                        <span style="color:#A0AEC0;">3 Years</span>
                        <span style="color:#3ECFB2;font-weight:600;">{row['avg_roi_3yr']:.1f}%</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-top:4px;">
                        <span style="color:#A0AEC0;">5 Years</span>
                        <span style="color:#6E9EFF;font-weight:600;">{row['avg_roi_5yr']:.1f}%</span>
                    </div>
                </div>
            </div>
            <div style="padding:0 16px 14px 16px;">
                <div style="font-size:0.75rem;color:#8A95B0;line-height:1.55;">{str(row.get('insight',''))[:180]}...</div>
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.82,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=330),
            tooltip=folium.Tooltip(
                f"<b style='font-family:DM Sans'>{row['location']}</b><br>"
                f"<span style='color:{color}'>{cat}</span> &bull; Score: {score:.0f}",
                sticky=False,
            ),
        ).add_to(cluster)

    # Legend (fixed top-right)
    legend_html = """
    <div style="
        position:fixed; top:14px; right:14px; z-index:9999;
        background:rgba(10,15,30,0.92);
        backdrop-filter:blur(10px);
        border:1px solid rgba(62,207,178,0.25);
        border-radius:12px;
        padding:14px 16px;
        font-family:'DM Sans',sans-serif;
        box-shadow:0 4px 24px rgba(0,0,0,0.5);
        min-width:160px;
    ">
        <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;
                    color:#3ECFB2;font-weight:700;margin-bottom:10px;">Categories</div>
    """ + "".join([
        f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;">
              <div style="width:11px;height:11px;border-radius:50%;background:{c};flex-shrink:0;"></div>
              <span style="font-size:0.78rem;color:#E8EDF5;">{cat}</span>
            </div>"""
        for cat, c in cat_colors_hex.items()
    ]) + "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))

    return m._repr_html_()

# ── CHARTS ───────────────────────────────────────────────────────
def chart_price_trends(price_trend: pd.DataFrame, selected_locations: list) -> go.Figure:
    years = ["price_2021", "price_2022", "price_2023", "price_2024", "price_2025"]
    year_labels = [2021, 2022, 2023, 2024, 2025]
    fig = go.Figure()
    filtered = price_trend[price_trend["location"].isin(selected_locations)]
    palette = [THEME["accent"], THEME["accent2"], THEME["gold"], "#F6A35A", "#A0AEC0",
               "#E879F9", "#34D399", "#F59E0B", "#60A5FA", "#FB923C"]
    for i, (_, row) in enumerate(filtered.iterrows()):
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=year_labels, y=[row[y] for y in years],
            name=row["location"], mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color, line=dict(width=1.5, color=THEME["bg_dark"])),
        ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title=dict(text="Price Trend 2021–2025 (Rs)", font=dict(size=13, color="#F0F4FF")),
        xaxis_title="Year", yaxis_title="Avg Price (Rs)",
        # Legend placed outside the chart area to prevent overlap
        legend=dict(
            bgcolor="rgba(10,15,35,0.85)", bordercolor="rgba(62,207,178,0.3)", borderwidth=1,
            font=dict(color="#F0F4FF", size=10),
            orientation="v", x=1.02, y=1, xanchor="left", yanchor="top",
        ),
        hovermode="x unified", height=360,
        margin=dict(r=160),
    )
    return fig


def chart_roi_comparison(loc_agg: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = loc_agg.nlargest(top_n, "avg_roi_5yr").sort_values("avg_roi_5yr")
    
    fig = go.Figure()
    for roi_col, label, color in [
        ("avg_roi_1yr", "1 Year", THEME["accent"]),
        ("avg_roi_3yr", "3 Years", THEME["accent2"]),
        ("avg_roi_5yr", "5 Years", THEME["gold"]),
    ]:
        fig.add_trace(go.Bar(
            name=label, y=top["location"], x=top[roi_col],
            orientation="h", marker=dict(color=color, opacity=0.85, line=dict(width=0)),
        ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        barmode="group",
        title=dict(text="ROI Forecast by Location (%)", font=dict(size=13, color="#F0F4FF")),
        xaxis_title="ROI (%)", height=410, bargap=0.2, bargroupgap=0.08,
        legend=dict(bgcolor="rgba(10,15,35,0.85)", bordercolor="rgba(62,207,178,0.25)",
                    borderwidth=1, font=dict(color="#F0F4FF", size=11),
                    orientation="h", x=0.5, y=1.15),
    )
    return fig


def chart_investment_ranking(top_investments: pd.DataFrame) -> go.Figure:
    df = top_investments.sort_values("avg_investment_score", ascending=True)
    colors = [CATEGORY_COLORS.get(c, THEME["accent"]) for c in df["dominant_category"]]
    fig = go.Figure(go.Bar(
        x=df["avg_investment_score"], y=df["location"],
        orientation="h", marker=dict(color=colors, opacity=0.88, line=dict(width=0)),
        text=df["avg_investment_score"].round(1), textposition="outside",
        textfont=dict(size=11, color="#C0CADF"),
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title=dict(text="Investment Score Ranking (Top 20)", font=dict(size=13, color="#F0F4FF")),
        xaxis_title="Investment Score", height=480, margin=dict(l=120),
    )
    return fig


def chart_category_donut(cat_dist: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=cat_dist["category"], values=cat_dist["count"], hole=0.62,
        marker=dict(
            colors=[CATEGORY_COLORS.get(c, "#A0AEC0") for c in cat_dist["category"]],
            line=dict(color=THEME["bg_dark"], width=3),
        ),
        textinfo="label+percent",
        textfont=dict(size=11, color="#F0F4FF"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title=dict(text="Property Category Distribution", font=dict(size=13, color="#F0F4FF")),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(10,15,35,0.85)", bordercolor="rgba(62,207,178,0.25)", borderwidth=1,
            font=dict(color="#F0F4FF", size=10), x=0.82, y=0.5,
        ),
        height=320,
    )
    return fig


def chart_zone_heatmap(zone_agg: pd.DataFrame) -> go.Figure:
    metrics = ["avg_price", "avg_growth", "avg_score"]
    labels = ["Avg Price/sqft", "Avg Growth %", "Inv. Score"]
    z_data = zone_agg[metrics].values.T
    z_norm = np.zeros_like(z_data, dtype=float)
    for i, row in enumerate(z_data):
        mn, mx = row.min(), row.max()
        z_norm[i] = (row - mn) / (mx - mn) if mx != mn else row * 0
    fig = go.Figure(go.Heatmap(
        z=z_norm, x=zone_agg["zone"].tolist(), y=labels,
        colorscale=[[0, THEME["bg_card"]], [0.5, THEME["accent2"]], [1, THEME["accent"]]],
        showscale=True,
        text=[[f"{zone_agg.iloc[j][metrics[i]]:.1f}" for j in range(len(zone_agg))] for i in range(len(metrics))],
        texttemplate="%{text}", textfont=dict(size=12, color="#F0F4FF"),
        hovertemplate="Zone: %{x}<br>Metric: %{y}<br>Value: %{text}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title=dict(text="Zone Performance Matrix", font=dict(size=13, color="#F0F4FF")),
        height=240, margin=dict(t=50, b=20),
    )
    return fig


def chart_future_trend(row: pd.Series, location: str) -> go.Figure:
    years_hist = [2021, 2022, 2023, 2024, 2025]
    price_hist = [row["avg_price_2021"], row["avg_price_2022"],
                  row["avg_price_2023"], row["avg_price_2024"], row["avg_price_2025"]]
    years_pred = [2026, 2028, 2030]
    price_pred = [row["pred_1yr"], row["pred_3yr"], row["pred_5yr"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years_hist, y=price_hist, name="Historical", mode="lines+markers",
        line=dict(color=THEME["accent"], width=2.5),
        marker=dict(size=8, color=THEME["accent"]),
    ))
    fig.add_trace(go.Scatter(
        x=[2025] + years_pred, y=[price_hist[-1]] + price_pred,
        name="Forecast", mode="lines+markers",
        line=dict(color=THEME["gold"], width=2, dash="dot"),
        marker=dict(size=8, color=THEME["gold"], symbol="diamond"),
    ))
    upper = [(p * 1.12) for p in price_pred]
    lower = [(p * 0.88) for p in price_pred]
    fig.add_trace(go.Scatter(
        x=[2025] + years_pred + years_pred[::-1],
        y=[price_hist[-1]] + upper + lower[::-1],
        fill="toself", fillcolor="rgba(201,169,110,0.08)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title=dict(text=f"{location} — Price Trajectory & Forecast", font=dict(size=13, color="#F0F4FF")),
        xaxis_title="Year", yaxis_title="Price (Rs)",
        height=300,
        # Legend placed above, non-overlapping
        legend=dict(
            bgcolor="rgba(10,15,35,0.85)", bordercolor="rgba(62,207,178,0.25)", borderwidth=1,
            font=dict(color="#F0F4FF", size=11),
            orientation="h", x=0, y=1.18, xanchor="left",
        ),
        margin=dict(t=60),
    )
    return fig


def chart_scatter_score_vs_roi(loc_agg: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        loc_agg, x="avg_investment_score", y="avg_roi_5yr",
        size="avg_price_per_sqft", color="dominant_category",
        color_discrete_map=CATEGORY_COLORS, hover_name="location",
        hover_data={"avg_growth": ":.1f", "avg_investment_score": ":.1f", "avg_roi_5yr": ":.1f"},
        labels={"avg_investment_score": "Investment Score", "avg_roi_5yr": "5-Year ROI (%)"},
        size_max=28,
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title=dict(text="Investment Score vs 5-Year ROI", font=dict(size=13, color="#F0F4FF")),
        height=380,
        legend=dict(
            bgcolor="rgba(10,15,35,0.85)", bordercolor="rgba(62,207,178,0.25)", borderwidth=1,
            font=dict(color="#F0F4FF", size=10),
        ),
    )
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────
def render_sidebar(loc_agg: pd.DataFrame, best_city: str, eda: dict, metrics: dict):
    with st.sidebar:
        # Compact branding (title is now in header)
        st.markdown(f"""
        <div style="padding:0.6rem 0 1rem 0;border-bottom:1px solid {THEME['border']};margin-bottom:1rem;">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:0.7rem;font-weight:600;
                        color:{THEME['accent']};letter-spacing:0.12em;text-transform:uppercase;">Navigation</div>
        </div>
        """, unsafe_allow_html=True)
        
        
        page = st.radio(
            "Navigation",
            ["Overview", "Interactive Map", "ROI Analysis", "Investment Ranking",
             "City Deep Dive", "Market Insights", "AI Agents", "Data Explorer"],
            label_visibility="collapsed",
        )

        st.markdown(f"""
        <div class="best-city-badge" style="margin-top:1.5rem;">
            <div class="badge-label">Top Investment Pick</div>
            <div class="badge-city">{best_city}</div>
        </div>
        """, unsafe_allow_html=True)

        # Pipeline Status with accuracy
        acc_1yr = round(100 - metrics.get('mape_1yr', 0), 1)
        acc_3yr = round(100 - metrics.get('mape_3yr', 0), 1)
        acc_5yr = round(100 - metrics.get('mape_5yr', 0), 1)

        st.markdown(f"""
        <div style="font-size:0.68rem;color:{THEME['text_sub']};text-transform:uppercase;
                    letter-spacing:0.1em;font-weight:600;margin-bottom:0.5rem;">Pipeline Status</div>
        <div style="background:rgba(14,20,44,0.8);border:1px solid rgba(62,207,178,0.12);
                    border-radius:10px;padding:12px 14px;font-size:0.77rem;line-height:2.1;">
            <div style="display:flex;justify-content:space-between;">
                <span style="color:{THEME['text_sub']};">Records</span>
                <span style="color:#3ECFB2;font-weight:600;">{eda.get('total_records',0):,}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:{THEME['text_sub']};">Locations</span>
                <span style="color:#3ECFB2;font-weight:600;">{eda.get('unique_locations',0)}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:{THEME['text_sub']};">Avg Growth</span>
                <span style="color:#C9A96E;font-weight:600;">{eda.get('avg_growth_rate',0)}%</span>
            </div>
            <div style="border-top:1px solid rgba(255,255,255,0.06);margin:6px 0;"></div>
            <div style="font-size:0.65rem;color:{THEME['text_sub']};text-transform:uppercase;
                        letter-spacing:0.1em;font-weight:600;margin-bottom:4px;">Model Accuracy</div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:{THEME['text_sub']};">1-Year</span>
                <span style="color:#3ECFB2;font-weight:700;">{acc_1yr}%</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:{THEME['text_sub']};">3-Year</span>
                <span style="color:#6E9EFF;font-weight:700;">{acc_3yr}%</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:{THEME['text_sub']};">5-Year</span>
                <span style="color:#C9A96E;font-weight:700;">{acc_5yr}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return page


# ── PAGE: OVERVIEW ────────────────────────────────────────────────
def page_overview(artifacts: dict, eda: dict, best_city: str):
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.9rem;font-weight:700;
                    color:#F0F4FF;line-height:1.1;">Market Overview</div>
        <div style="font-size:0.88rem;color:#8A95B0;margin-top:0.3rem;">
            Bangalore Property Intelligence &bull; 2021–2025 &bull; Live Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

    loc_agg = artifacts["loc_agg"]
    cat_dist = artifacts["cat_dist"]

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card("Total Records", f"{eda.get('total_records',0):,}", "Processed by pipeline", c1)
    kpi_card("Unique Locations", str(eda.get("unique_locations", 0)), "Bangalore micro-markets", c2)
    kpi_card("Avg Price / sqft", f"Rs {int(eda.get('avg_price_per_sqft',0)):,}", "Market midpoint 2025", c3)
    kpi_card("Avg Annual Growth", f"{eda.get('avg_growth_rate',0)}%", "CAGR 2021–2025", c4)
    kpi_card("Top City", best_city, "Highest investment score", c5)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        section_header("Price Trend", "2021–2025")
        top10_locs = loc_agg.nlargest(10, "avg_investment_score")["location"].tolist()
        fig = chart_price_trends(artifacts["price_trend"], top10_locs)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        section_header("Category", "Distribution")
        fig2 = chart_category_donut(cat_dist)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        section_header("Zone", "Performance")
        fig3 = chart_zone_heatmap(artifacts["zone_agg"])
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    section_header("Year-on-Year Growth", "Summary")
    yoy = eda.get("yoy_growth", {})
    if yoy:
        cols = st.columns(len(yoy))
        for col, (period, val) in zip(cols, yoy.items()):
            kpi_card(period, f"{val}%", "   ", col)

    team_footer()


# ── PAGE: MAP ─────────────────────────────────────────────────────
def page_map(artifacts: dict):
    st.markdown("""
    <div style="margin-bottom:1rem;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;color:#F0F4FF;">
            Geospatial Intelligence Map</div>
        <div style="font-size:0.85rem;color:#8A95B0;margin-top:0.2rem;">
            Click any marker for detailed ROI analysis and investment insights
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_filter, _ = st.columns([2, 8])
    with col_filter:
        map_filter = st.selectbox("Filter", ["All", "Top Premium", "Top Affordable", "High Growth"],
                                  label_visibility="collapsed")
    loc_agg = artifacts["loc_agg"]
    map_html = build_folium_map(loc_agg, filter_type=map_filter)
    components.html(map_html, height=620, scrolling=False)
    team_footer()


# ── PAGE: ROI ANALYSIS ────────────────────────────────────────────
def page_roi(artifacts: dict):
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;
                color:#F0F4FF;margin-bottom:0.3rem;">ROI Analysis</div>
    <div style="font-size:0.85rem;color:#8A95B0;margin-bottom:1.2rem;">
        1-year, 3-year, and 5-year projected returns across micro-markets
    </div>
    """, unsafe_allow_html=True)

    loc_agg = artifacts["loc_agg"]
    col1, col2 = st.columns([3, 2])
    with col1:
        section_header("ROI Comparison", "Top 15 Locations")
        fig = chart_roi_comparison(loc_agg)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with col2:
        section_header("Score vs ROI", "Bubble Chart")
        fig2 = chart_scatter_score_vs_roi(loc_agg)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    section_header("ROI Data", "All Locations")
    table_cols = ["location", "zone", "dominant_category", "avg_price_per_sqft",
                  "avg_roi_1yr", "avg_roi_3yr", "avg_roi_5yr", "avg_investment_score"]
    roi_table = loc_agg[table_cols].copy()
    roi_table.columns = ["Location", "Zone", "Category", "Price/sqft (Rs)",
                         "ROI 1yr (%)", "ROI 3yr (%)", "ROI 5yr (%)", "Score"]
    roi_table = roi_table.sort_values("ROI 5yr (%)", ascending=False).reset_index(drop=True)
    roi_table["Rank"] = range(1, len(roi_table) + 1)
    st.dataframe(roi_table.set_index("Rank"), use_container_width=True, height=350)
    team_footer()


# ── PAGE: INVESTMENT RANKING ──────────────────────────────────────
def page_ranking(artifacts: dict):
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;
                color:#F0F4FF;margin-bottom:0.3rem;">Investment Ranking</div>
    <div style="font-size:0.85rem;color:#8A95B0;margin-bottom:1.2rem;">
        Composite investment score ranked by growth, connectivity, ROI, and value
    </div>
    """, unsafe_allow_html=True)

    top = artifacts["top_investments"]
    col1, col2 = st.columns([3, 2])

    with col1:
        fig = chart_investment_ranking(top)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        section_header("Top 10", "Ranked Locations")
        for i, (_, r) in enumerate(top.head(10).iterrows()):
            cat = r["dominant_category"]
            color = CATEGORY_COLORS.get(cat, THEME["accent"])
            medal = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"][i]
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;background:rgba(14,20,44,0.8);
                        border:1px solid rgba(255,255,255,0.06);border-radius:10px;
                        padding:10px 14px;margin-bottom:6px;">
                <div style="font-family:'Space Grotesk',sans-serif;font-size:0.85rem;
                            font-weight:700;color:#8A95B0;width:26px;">{medal}</div>
                <div style="flex:1;">
                    <div style="font-size:0.88rem;font-weight:600;color:#F0F4FF;">{r['location']}</div>
                    <div style="font-size:0.7rem;color:#8A95B0;margin-top:1px;">{r['zone']} Zone</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:1rem;font-weight:700;color:#3ECFB2;">{r['avg_investment_score']:.1f}</div>
                    <span style="background:{color}22;color:{color};border:1px solid {color}44;
                                 font-size:0.62rem;font-weight:600;padding:1px 8px;border-radius:20px;">{cat}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    team_footer()


# ── PAGE: CITY DEEP DIVE ──────────────────────────────────────────
def page_city(artifacts: dict):
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;
                color:#F0F4FF;margin-bottom:0.3rem;">City Deep Dive</div>
    <div style="font-size:0.85rem;color:#8A95B0;margin-bottom:1.2rem;">
        Granular analysis per micro-market
    </div>
    """, unsafe_allow_html=True)

    loc_agg = artifacts["loc_agg"]
    locations = sorted(loc_agg["location"].tolist())
    selected = st.selectbox("Select Location", locations)

    row = loc_agg[loc_agg["location"] == selected].iloc[0]
    cat = row["dominant_category"]
    cat_color = CATEGORY_COLORS.get(cat, THEME["accent"])

    col1, col2, col3, col4 = st.columns(4)
    kpi_card("Price / sqft", f"Rs {int(row['avg_price_per_sqft']):,}","Average price", col1)
    kpi_card("Annual Growth", f"{row['avg_growth']:.1f}%", "CAGR 2021–2025", col2)
    kpi_card("Investment Score", f"{row['avg_investment_score']:.1f}", "Composite score", col3)
    kpi_card("5-Year ROI", f"{row['avg_roi_5yr']:.1f}%", "Projected return", col4)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    col_chart, col_insight = st.columns([3, 2])
    with col_chart:
        fig = chart_future_trend(row, selected)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_insight:
        section_header("AI Insight")
        st.markdown(f'<div class="insight-box">{row.get("insight", "No insight available.")}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-top:1rem;background:rgba(14,20,44,0.8);border:1px solid rgba(255,255,255,0.06);
                    border-radius:10px;padding:14px 16px;">
            <div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;
                        color:#8A95B0;margin-bottom:10px;font-weight:600;">Property Details</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:0.82rem;">
                <div><span style="color:#8A95B0;">Zone</span><br>
                     <span style="color:#F0F4FF;font-weight:600;">{row['zone']}</span></div>
                <div><span style="color:#8A95B0;">Category</span><br>
                     <span style="color:{cat_color};font-weight:600;">{cat}</span></div>
                <div><span style="color:#8A95B0;">ROI 1yr</span><br>
                     <span style="color:#3ECFB2;font-weight:600;">{row['avg_roi_1yr']:.1f}%</span></div>
                <div><span style="color:#8A95B0;">ROI 3yr</span><br>
                     <span style="color:#3ECFB2;font-weight:600;">{row['avg_roi_3yr']:.1f}%</span></div>
                <div><span style="color:#8A95B0;">Connectivity</span><br>
                     <span style="color:#F0F4FF;font-weight:600;">{row['avg_conn']:.0f}/75</span></div>
                <div><span style="color:#8A95B0;">Listings</span><br>
                     <span style="color:#F0F4FF;font-weight:600;">{int(row['count'])}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    section_header("Compare with Other Locations")

    all_locs = sorted(loc_agg["location"].tolist())
    options_list = [l for l in all_locs if l != selected]
    # Fix: ensure defaults never include the currently selected location
    default_locs = [l for l in all_locs if l != selected][:3]

    compare_locs = st.multiselect(
        "Add locations to compare",
        options_list,
        default=default_locs,
        max_selections=6,
    )

    # Build a safe list: filter only locations present in price_trend
    price_trend = artifacts["price_trend"]
    valid_locs = price_trend["location"].tolist()
    plot_locs = [l for l in ([selected] + compare_locs) if l in valid_locs]

    if plot_locs:
        fig2 = chart_price_trends(price_trend, plot_locs)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No valid locations to compare. Please select at least one location.")

    team_footer()


# ── PAGE: MARKET INSIGHTS ─────────────────────────────────────────
def page_insights(artifacts: dict, eda: dict, best_city: str):
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;
                color:#F0F4FF;margin-bottom:0.3rem;">Market Insights</div>
    <div style="font-size:0.85rem;color:#8A95B0;margin-bottom:1.2rem;">
        Automated intelligence summaries from the multi-agent pipeline
    </div>
    """, unsafe_allow_html=True)

    loc_agg = artifacts["loc_agg"]
    cat_dist = artifacts["cat_dist"]
    top_growth = loc_agg.nlargest(1, "avg_growth").iloc[0]
    top_score = loc_agg.nlargest(1, "avg_investment_score").iloc[0]
    top_roi = loc_agg.nlargest(1, "avg_roi_5yr").iloc[0]
    affordable = loc_agg.nsmallest(1, "avg_price_per_sqft").iloc[0]

    macro_insights = [
        f"The Bangalore property market recorded an average annual appreciation of "
        f"{eda.get('avg_growth_rate',0):.1f}% across {eda.get('unique_locations',0)} micro-markets "
        f"between 2021 and 2025, driven by sustained IT sector demand and infrastructure expansion.",
        f"{top_growth['location']} leads the growth table with {top_growth['avg_growth']:.1f}% average annual "
        f"appreciation. Proximity to emerging tech corridors and metro connectivity are primary catalysts.",
        f"{top_score['location']} achieves the highest composite investment score of "
        f"{top_score['avg_investment_score']:.1f}/100, offering a balanced combination of growth momentum and connectivity.",
        f"The 5-year ROI outlook is strongest in {top_roi['location']} at {top_roi['avg_roi_5yr']:.1f}%, "
        f"driven by infrastructure projects expected to come online before 2030.",
        f"{affordable['location']} offers the most accessible entry point at Rs {int(affordable['avg_price_per_sqft']):,}/sqft, "
        f"making it attractive for first-time investors seeking rental yield.",
        f"High Growth and High Potential properties collectively represent "
        f"{cat_dist[cat_dist['category'].isin(['High Growth','High Potential'])]['count'].sum()} listings, "
        f"underscoring a broad appreciation cycle rather than localised speculation.",
    ]

    section_header("Macro Market Analysis")
    for insight in macro_insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    section_header("Location-Level Insights")
    col_filter, _ = st.columns([2, 6])
    with col_filter:
        cat_filter = st.selectbox("Category",
            ["All", "Premium", "High Growth", "High Potential", "Stable", "Budget"],
            label_visibility="collapsed")

    filtered_locs = loc_agg.copy()
    if cat_filter != "All":
        filtered_locs = filtered_locs[filtered_locs["dominant_category"] == cat_filter]
    filtered_locs = filtered_locs.sort_values("avg_investment_score", ascending=False)

    for _, r in filtered_locs.iterrows():
        cat = r["dominant_category"]
        cat_color = CATEGORY_COLORS.get(cat, THEME["accent"])
        st.markdown(f"""
        <div style="background:rgba(14,20,44,0.8);border:1px solid rgba(255,255,255,0.06);
                    border-left:3px solid {cat_color};border-radius:10px;
                    padding:12px 16px;margin-bottom:8px;display:flex;gap:16px;align-items:flex-start;">
            <div style="min-width:130px;">
                <div style="font-size:0.88rem;font-weight:700;color:#F0F4FF;">{r['location']}</div>
                <span style="background:{cat_color}22;color:{cat_color};border:1px solid {cat_color}44;
                             font-size:0.62rem;font-weight:600;padding:1px 8px;border-radius:20px;
                             margin-top:4px;display:inline-block;">{cat}</span>
                <div style="margin-top:6px;font-size:0.75rem;color:#3ECFB2;font-weight:600;">
                    Score: {r['avg_investment_score']:.1f}
                </div>
            </div>
            <div style="font-size:0.8rem;color:#C0CADF;line-height:1.65;flex:1;">
                {r.get('insight','No insight available.')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    team_footer()


# ── PAGE: AI AGENTS ───────────────────────────────────────────────
def page_agents(execution_logs: str):
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;
                color:#F0F4FF;margin-bottom:0.3rem;">AI Agent Architecture</div>
    <div style="font-size:0.85rem;color:#8A95B0;margin-bottom:1.5rem;">
        8 specialised agents orchestrated in sequence to deliver autonomous real estate intelligence
    </div>
    """, unsafe_allow_html=True)

    # Pipeline flow diagram
    section_header("Agent Pipeline", "Flow")
    flow_cols = st.columns(8)
    for i, agent in enumerate(AGENT_INFO):
        with flow_cols[i]:
            st.markdown(f"""
            <div style="text-align:center;padding:0.4rem 0;">
                <div style="background:{agent['color']}22;border:1px solid {agent['color']}55;
                            border-radius:50%;width:40px;height:40px;
                            display:flex;align-items:center;justify-content:center;
                            margin:0 auto 0.4rem auto;
                            font-family:'Space Grotesk',sans-serif;font-size:0.85rem;
                            font-weight:700;color:{agent['color']};">{agent['id']}</div>
                <div style="font-size:0.62rem;color:#C0CADF;text-align:center;line-height:1.3;
                            font-weight:500;">{agent['name'].replace(' Agent','')}</div>
            </div>
            """, unsafe_allow_html=True)
            if i < 7:
                pass  # arrows would need HTML tricks; flow is implied by columns

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Agent detail cards — 2 per row
    section_header("Agent Details", "& Capabilities")
    for i in range(0, len(AGENT_INFO), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(AGENT_INFO):
                agent = AGENT_INFO[i + j]
                with col:
                    st.markdown(f"""
                    <div class="agent-card" style="border-left:3px solid {agent['color']};">
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.7rem;">
                            <div style="background:{agent['color']}22;border:1px solid {agent['color']}55;
                                        border-radius:8px;padding:4px 10px;
                                        font-family:'Space Grotesk',sans-serif;font-size:0.78rem;
                                        font-weight:700;color:{agent['color']};">Agent {agent['id']}</div>
                            <div>
                                <div style="font-size:0.92rem;font-weight:600;color:#F0F4FF;">{agent['name']}</div>
                                <div style="font-size:0.65rem;color:{agent['color']};font-weight:500;
                                            letter-spacing:0.08em;text-transform:uppercase;">{agent['type']}</div>
                            </div>
                        </div>
                        <div style="font-size:0.82rem;color:#C0CADF;line-height:1.65;margin-bottom:0.8rem;">
                            {agent['desc']}
                        </div>
                        <div style="font-size:0.68rem;color:#8A95B0;text-transform:uppercase;
                                    letter-spacing:0.08em;font-weight:600;margin-bottom:0.4rem;">Outputs</div>
                        <div style="display:flex;flex-wrap:wrap;gap:5px;">
                            {"".join([f'<span style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:6px;padding:2px 8px;font-size:0.7rem;color:#C0CADF;">{o}</span>' for o in agent['outputs']])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Execution logs
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    section_header("Execution Logs", "Latest Run")

    if execution_logs:
        # Colour-code log lines
        colored_lines = []
        for line in execution_logs.strip().split("\n"):
            if "✓" in line:
                colored_lines.append(f'<span style="color:#3ECFB2;">{line}</span>')
            elif "⚠" in line or "WARNING" in line.upper():
                colored_lines.append(f'<span style="color:#F6A35A;">{line}</span>')
            elif "ERROR" in line.upper():
                colored_lines.append(f'<span style="color:#FC8181;">{line}</span>')
            elif line.startswith("="):
                colored_lines.append(f'<span style="color:#6E9EFF;font-weight:600;">{line}</span>')
            elif line.startswith("[Pipeline]"):
                colored_lines.append(f'<span style="color:#C9A96E;font-weight:600;">{line}</span>')
            else:
                colored_lines.append(f'<span style="color:#A8C4FF;">{line}</span>')
        log_html = "\n".join(colored_lines)
        st.markdown(f'<div class="log-terminal">{log_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="log-terminal" style="color:#F6A35A;">
        [Pipeline] No execution logs available. Pipeline may have been loaded from cache.
        [Pipeline] Re-run without cache to see live execution output.
        </div>
        """, unsafe_allow_html=True)

    team_footer()


# ── PAGE: DATA EXPLORER ───────────────────────────────────────────
def page_data_explorer(artifacts: dict):
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;
                color:#F0F4FF;margin-bottom:0.3rem;">Data Explorer</div>
    <div style="font-size:0.85rem;color:#8A95B0;margin-bottom:1.2rem;">
        Processed output dataset — colour-coded by column group — with live filters and sorting
    </div>
    """, unsafe_allow_html=True)

    # Load processed data
    df = artifacts.get("full_df", None)
    if df is None or df.empty:
        # Try reading from disk
        csv_path = "processed_data.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            st.warning("Processed data not found. Run the pipeline first.")
            return

    # Column groups with color codes for legend
    original_cols = ["location", "zone", "property_type", "bhk", "total_sqft",
                     "price_per_sqft", "price_2021", "price_2022", "price_2023", "price_2024", "price_2025",
                     "annual_growth_rate_pct", "metro_access", "distance_to_it_hub_km", "distance_to_airport_km"]
    feature_cols = ["price_growth_21_25", "price_growth_23_25", "price_cagr",
                    "total_value_lakhs", "value_per_bhk", "sqft_per_bhk", "connectivity_score"]
    prediction_cols = ["predicted_price_1yr", "predicted_price_3yr", "predicted_price_5yr"]
    analysis_cols = ["roi_1yr", "roi_3yr", "roi_5yr", "growth_rate_index", "investment_score",
                     "category", "location_rank"]
    insight_cols = ["location_insight", "best_city_flag"]

    # Colour legend
    st.markdown("""
    <div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:1.2rem;align-items:center;">
        <div style="display:flex;align-items:center;gap:6px;">
            <div style="width:12px;height:12px;border-radius:3px;background:#3A4460;"></div>
            <span style="font-size:0.77rem;color:#C0CADF;">Original</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
            <div style="width:12px;height:12px;border-radius:3px;background:#1A3A30;"></div>
            <span style="font-size:0.77rem;color:#3ECFB2;">Engineered Features</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
            <div style="width:12px;height:12px;border-radius:3px;background:#1A2040;"></div>
            <span style="font-size:0.77rem;color:#6E9EFF;">ML Predictions</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
            <div style="width:12px;height:12px;border-radius:3px;background:#2A2010;"></div>
            <span style="font-size:0.77rem;color:#C9A96E;">Investment Analysis</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
            <div style="width:12px;height:12px;border-radius:3px;background:#1A1040;"></div>
            <span style="font-size:0.77rem;color:#E879F9;">Insights</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    section_header("Filters")
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        zone_opts = ["All"] + sorted(df["zone"].dropna().unique().tolist()) if "zone" in df.columns else ["All"]
        sel_zone = st.selectbox("Zone", zone_opts)
    with fcol2:
        cat_opts = ["All"] + sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else ["All"]
        sel_cat = st.selectbox("Category", cat_opts)
    with fcol3:
        loc_opts = ["All"] + sorted(df["location"].dropna().unique().tolist()) if "location" in df.columns else ["All"]
        sel_loc = st.selectbox("Location", loc_opts)
    with fcol4:
        col_group = st.selectbox("Column Group",
            ["All", "Original", "Engineered Features", "ML Predictions", "Investment Analysis", "Insights"])

    # Apply filters
    filtered = df.copy()
    if sel_zone != "All" and "zone" in filtered.columns:
        filtered = filtered[filtered["zone"] == sel_zone]
    if sel_cat != "All" and "category" in filtered.columns:
        filtered = filtered[filtered["category"] == sel_cat]
    if sel_loc != "All" and "location" in filtered.columns:
        filtered = filtered[filtered["location"] == sel_loc]

    # Select columns by group
    all_display_cols = [c for c in (original_cols + feature_cols + prediction_cols + analysis_cols + insight_cols) if c in filtered.columns]
    if col_group == "Original":
        display_cols = [c for c in original_cols if c in filtered.columns]
    elif col_group == "Engineered Features":
        display_cols = [c for c in (["location"] + feature_cols) if c in filtered.columns]
    elif col_group == "ML Predictions":
        display_cols = [c for c in (["location"] + prediction_cols) if c in filtered.columns]
    elif col_group == "Investment Analysis":
        display_cols = [c for c in (["location"] + analysis_cols) if c in filtered.columns]
    elif col_group == "Insights":
        display_cols = [c for c in (["location"] + insight_cols) if c in filtered.columns]
    else:
        display_cols = all_display_cols

    table_df = filtered[display_cols].copy()

    # Stats bar
    st.markdown(f"""
    <div style="display:flex;gap:20px;margin-bottom:1rem;">
        <div style="font-size:0.82rem;color:#C0CADF;">
            <span style="color:#3ECFB2;font-weight:700;">{len(table_df):,}</span> records
        </div>
        <div style="font-size:0.82rem;color:#C0CADF;">
            <span style="color:#6E9EFF;font-weight:700;">{len(display_cols)}</span> columns shown
        </div>
        <div style="font-size:0.82rem;color:#C0CADF;">
            <span style="color:#C9A96E;font-weight:700;">{len(table_df['location'].unique()) if 'location' in table_df.columns else '—'}</span> locations
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display with column config
    column_config = {}
    for col in display_cols:
        if col in prediction_cols:
            column_config[col] = st.column_config.NumberColumn(col, format="Rs %.0f")
        elif col in ["roi_1yr", "roi_3yr", "roi_5yr", "avg_growth", "annual_growth_rate_pct",
                     "price_growth_21_25", "price_growth_23_25", "price_cagr"]:
            column_config[col] = st.column_config.NumberColumn(col, format="%.2f %%")
        elif col == "investment_score":
            column_config[col] = st.column_config.ProgressColumn(
                col, min_value=0, max_value=100, format="%.1f")
        elif col == "location_insight":
            column_config[col] = st.column_config.TextColumn(col, width="large")

    st.dataframe(
        table_df.reset_index(drop=True),
        use_container_width=True,
        height=480,
        column_config=column_config,
    )

    # Summary stats for numeric cols
    numeric_display = table_df.select_dtypes(include=[np.number])
    if not numeric_display.empty:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        section_header("Summary Statistics")
        st.dataframe(numeric_display.describe().round(2), use_container_width=True, height=220)

    team_footer()


# ── MAIN ─────────────────────────────────────────────────────────
def main():
    inject_css()

    csv_candidates = [
        "BHP_3.csv",
        "/mnt/user-data/uploads/BHP_3.csv",
        os.path.join(os.path.dirname(__file__), "BHP_3.csv"),
    ]
    filepath = next((p for p in csv_candidates if os.path.exists(p)), None)

    with st.spinner("Running AREIS multi-agent pipeline..."):
        result = load_pipeline_data(filepath)

    artifacts  = result["artifacts"]
    eda        = result["eda_summary"]
    metrics    = result["model_metrics"]
    best_city  = result["best_city"]
    exec_logs  = result.get("execution_logs", "")
    loc_agg    = artifacts["loc_agg"]

    # Persistent header
    render_header()

    # Sidebar + navigation
    page = render_sidebar(loc_agg, best_city, eda, metrics)

    # Page routing
    if page == "Overview":
        page_overview(artifacts, eda, best_city)
    elif page == "Interactive Map":
        page_map(artifacts)
    elif page == "ROI Analysis":
        page_roi(artifacts)
    elif page == "Investment Ranking":
        page_ranking(artifacts)
    elif page == "City Deep Dive":
        page_city(artifacts)
    elif page == "Market Insights":
        page_insights(artifacts, eda, best_city)
    elif page == "AI Agents":
        page_agents(exec_logs)
    elif page == "Data Explorer":
        page_data_explorer(artifacts)


if __name__ == "__main__":
    main()

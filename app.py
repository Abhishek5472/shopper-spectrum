import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import difflib
import os

# Set page config FIRST
st.set_page_config(
    page_title="Shopper Spectrum - Retail Analytics Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom global CSS forcing a premium dark-analytics SaaS aesthetic
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Font & Theme */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Force Dark Theme Main Layout */
.stApp {
    background-color: #0B0F19 !important;
    color: #F8FAFC !important;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #0F1424 !important;
    border-right: 1px solid #222C4A !important;
}
section[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

/* Custom Tab/Menu Card Styling for Sidebar Radio Buttons */
div[data-testid="stRadio"] div[role="radiogroup"] {
    background-color: transparent !important;
    padding: 0 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label {
    background-color: #151B30 !important;
    border: 1px solid #222C4A !important;
    padding: 12px 16px !important;
    border-radius: 8px !important;
    margin-bottom: 10px !important;
    color: #94A3B8 !important;
    cursor: pointer !important;
    transition: all 0.25s ease-in-out !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    border-color: #00ADB5 !important;
    background-color: #1D243F !important;
    color: #00ADB5 !important;
    transform: translateX(4px);
}
/* Selected Navigation Option Gradient */
div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(135deg, #005B94 0%, #00ADB5 100%) !important;
    border-color: #00ADB5 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(0, 173, 181, 0.25) !important;
}
/* Hide standard circular radio container dot */
div[data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stRadioRoundContainer"] {
    display: none !important;
}
/* Hide widget label */
div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] {
    display: none !important;
}

/* Brand Banner Identity */
.brand-header {
    background: linear-gradient(135deg, #0F1424 0%, #151B30 100%);
    border-left: 5px solid #00ADB5;
    padding: 16px 24px;
    border-radius: 8px;
    margin-bottom: 20px;
    border-top: 1px solid #222C4A;
    border-right: 1px solid #222C4A;
    border-bottom: 1px solid #222C4A;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    animation: fadeIn 0.5s ease-in-out;
}
.brand-title {
    font-size: 24px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.brand-subtitle {
    font-size: 13px;
    color: #94A3B8;
    margin-top: 4px;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* Page Section Identity Headers */
.page-section-header {
    margin-bottom: 20px;
    border-bottom: 1px solid #222C4A;
    padding-bottom: 10px;
}
.page-title {
    color: #FFFFFF;
    font-size: 20px;
    font-weight: 700;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.page-subtitle {
    color: #00ADB5;
    font-size: 13px;
    margin: 3px 0 0 0;
    font-weight: 500;
}

/* KPI Dashboard Cards */
.kpi-card {
    background-color: #151B30;
    border-left: 4px solid #00ADB5;
    padding: 16px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    border-top: 1px solid #222C4A;
    border-right: 1px solid #222C4A;
    border-bottom: 1px solid #222C4A;
    text-align: center;
    transition: transform 0.25s ease-in-out, box-shadow 0.25s ease-in-out, border-left-color 0.25s ease-in-out;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0, 173, 181, 0.15);
    border-left-color: #005B94;
}
.kpi-title {
    color: #94A3B8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
    margin-bottom: 6px;
}
.kpi-value {
    color: #FFFFFF;
    font-size: 26px;
    font-weight: 700;
}

/* Section Header */
.section-header {
    font-size: 15px;
    font-weight: 700;
    color: #FFFFFF;
    border-bottom: 2px solid #00ADB5;
    padding-bottom: 4px;
    margin-top: 22px;
    margin-bottom: 12px;
    letter-spacing: 0.5px;
}

/* Custom Container Cards */
.custom-card {
    background-color: #151B30;
    border: 1px solid #222C4A;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 15px;
    transition: transform 0.25s ease-in-out, box-shadow 0.25s ease-in-out;
}
.custom-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.15);
}

/* Product Recommender Grid Card */
.product-card {
    background-color: #121829;
    border: 1px solid #222C4A;
    border-top: 4px solid #00ADB5;
    padding: 16px;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    min-height: 190px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.25s ease-in-out, box-shadow 0.25s ease-in-out, border-top-color 0.25s ease-in-out;
}
.product-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 173, 181, 0.15);
    border-top-color: #005B94;
}

/* Strategy & Business Insight Cards */
.insight-card {
    padding: 18px;
    border-radius: 8px;
    margin-bottom: 15px;
    border: 1px solid #222C4A;
    transition: transform 0.25s ease-in-out, box-shadow 0.25s ease-in-out;
}
.insight-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 18px rgba(0,0,0,0.2);
}

/* Gradient recommendation button styling */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #005B94 0%, #00ADB5 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 10px 24px !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 10px rgba(0, 173, 181, 0.2) !important;
    transition: all 0.25s ease-in-out !important;
    width: 100% !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-size: 13px;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(0, 173, 181, 0.45) !important;
}

/* Custom Slider Styling */
div[data-testid="stSlider"] div[data-inner-slider-tracker="true"] {
    background-color: #222C4A !important;
}
div[data-testid="stSlider"] div[data-testid="stSliderTrack"] {
    background: linear-gradient(90deg, #005B94, #00ADB5) !important;
}
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #00ADB5 !important;
    border-color: #FFFFFF !important;
}

/* Animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-5px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes loadBar {
    from { width: 0%; }
}
</style>
""", unsafe_allow_html=True)

# Helper function to load model components
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("models/scaler.pkl")
        kmeans = joblib.load("models/kmeans_model.pkl")
        product_similarity = joblib.load("models/product_similarity.pkl")
        product_names = joblib.load("models/product_names.pkl")
        metrics = joblib.load("models/dashboard_metrics.pkl")
        return scaler, kmeans, product_similarity, product_names, metrics
    except Exception as e:
        st.error(f"Error loading model files: {e}. Please ensure that the models were trained and saved successfully.")
        return None, None, None, None, None

scaler, kmeans, product_similarity, product_names, metrics = load_models()

# Render Brand Header Panel (No Welcome Screen)
st.markdown("""
<div class="brand-header">
    <div class="brand-title">
        <span>🛒</span> <span style="background: linear-gradient(90deg, #FFFFFF, #00ADB5); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Shopper Spectrum</span>
    </div>
    <div class="brand-subtitle">
        AI-Powered Customer Segmentation & Product Recommendation Platform
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar menu configuration
st.sidebar.markdown(
    "<div style='margin-bottom: 15px; border-bottom: 1px solid #222C4A; padding-bottom: 10px;'><h2 style='color:#00ADB5; margin:0; font-size:20px; font-weight:700;'>Navigation</h2>"
    "<p style='font-size:11px; color:#94A3B8; margin: 1px 0 0 0; font-weight: 500;'>Select Platform View</p></div>", 
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigate Menu",
    ["🏠 Home Dashboard", "🛒 Product Recommendation", "👥 Customer Segmentation", "📊 Business Insights"]
)

st.sidebar.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

if metrics is not None:
    # Sidebar Statistics Redesign
    st.sidebar.markdown(f"""
    <div style="background-color: #151B30; padding: 15px; border-radius: 8px; border-left: 4px solid #00ADB5; margin-bottom: 15px; border-top:1px solid #222C4A; border-right:1px solid #222C4A; border-bottom:1px solid #222C4A;">
        <div style="font-size: 10px; text-transform: uppercase; color: #94A3B8; letter-spacing: 1.2px; font-weight: 600; margin-bottom: 8px;">Platform Statistics</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
            <div style="background-color: #0B0F19; padding: 8px; border-radius: 6px; border: 1px solid #222C4A; text-align: center;">
                <div style="font-size: 8px; color: #94A3B8; text-transform: uppercase;">Revenue</div>
                <div style="font-size: 11px; font-weight: bold; color: #00ADB5; margin-top: 2px;">£{metrics['total_revenue']/1e6:.2f}M</div>
            </div>
            <div style="background-color: #0B0F19; padding: 8px; border-radius: 6px; border: 1px solid #222C4A; text-align: center;">
                <div style="font-size: 8px; color: #94A3B8; text-transform: uppercase;">Customers</div>
                <div style="font-size: 11px; font-weight: bold; color: #00ADB5; margin-top: 2px;">{metrics['total_customers']:,}</div>
            </div>
            <div style="background-color: #0B0F19; padding: 8px; border-radius: 6px; border: 1px solid #222C4A; text-align: center;">
                <div style="font-size: 8px; color: #94A3B8; text-transform: uppercase;">Invoices</div>
                <div style="font-size: 11px; font-weight: bold; color: #00ADB5; margin-top: 2px;">{metrics['total_transactions']:,}</div>
            </div>
            <div style="background-color: #0B0F19; padding: 8px; border-radius: 6px; border: 1px solid #222C4A; text-align: center;">
                <div style="font-size: 8px; color: #94A3B8; text-transform: uppercase;">Catalog</div>
                <div style="font-size: 11px; font-weight: bold; color: #00ADB5; margin-top: 2px;">{metrics['total_products']:,}</div>
            </div>
        </div>
        <hr style="margin: 10px 0; border: 0; border-top: 1px solid #222C4A;">
        <div style="font-size: 9px; color: #94A3B8; text-align: center;">Dataset: <strong>online_retail.csv</strong><br>Retained Records: <strong>72.46% (392,692 rows)</strong></div>
    </div>
    """, unsafe_allow_html=True)

    summary_df = pd.DataFrame(metrics['segment_summary'])

    # ------------------ HOME DASHBOARD ------------------
    if page == "🏠 Home Dashboard":
        st.markdown("""
        <div class="page-section-header">
            <h2 class="page-title">🏠 Home Dashboard</h2>
            <p class="page-subtitle">Retail Performance Monitoring & Customer Analytics</p>
        </div>
        """, unsafe_allow_html=True)

        # Top KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Revenue</div>
                <div class="kpi-value">£{metrics['total_revenue']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Transactions</div>
                <div class="kpi-value">{metrics['total_transactions']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Customers</div>
                <div class="kpi-value">{metrics['total_customers']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Unique Catalog Items</div>
                <div class="kpi-value">{metrics['total_products']:,}</div>
            </div>
            """, unsafe_allow_html=True)

        # REDESIGNED HOME GRID 1: Revenue Trend & Customer Base Distribution Side-by-Side
        st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
        col_mid1, col_mid2 = st.columns([3, 2])
        
        with col_mid1:
            st.markdown("<div class=\"section-header\">Monthly Sales & Transaction Volume Trends</div>", unsafe_allow_html=True)
            m_df = pd.DataFrame(metrics['monthly_data'])
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Bar(
                x=m_df['YearMonth_str'],
                y=m_df['Revenue'],
                name='Revenue (£)',
                marker_color='#005B94',
                yaxis='y1'
            ))
            fig_trend.add_trace(go.Scatter(
                x=m_df['YearMonth_str'],
                y=m_df['Transactions'],
                name='Transactions',
                line=dict(color='#00ADB5', width=3),
                yaxis='y2'
            ))
            fig_trend.update_layout(
                template='plotly_dark',
                yaxis=dict(title='Revenue (£)', side='left', title_font=dict(color='#E2E8F0'), tickfont=dict(color='#94A3B8')),
                yaxis2=dict(title='Transactions', side='right', overlaying='y', gridcolor='rgba(0,0,0,0)', title_font=dict(color='#E2E8F0'), tickfont=dict(color='#94A3B8')),
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(21,27,48,0.8)'),
                hovermode="x unified",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                height=280
            )
            st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
            
        with col_mid2:
            st.markdown("<div class=\"section-header\">Customer Base Distribution by Cohort</div>", unsafe_allow_html=True)
            fig_cust = px.pie(
                summary_df, 
                values='Customer_Count', 
                names='Segment', 
                color='Segment',
                color_discrete_map={'High Value': '#FFD700', 'Regular': '#00ADB5', 'Occasional': '#005B94', 'At Risk': '#FF4D4D'},
                hole=0.55,
                height=280
            )
            fig_cust.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=10, color='#94A3B8'))
            )
            # Center annotation label to prevent clipping and add polished donut look
            fig_cust.add_annotation(text="Customers", x=0.5, y=0.5, font_size=11, showarrow=False, font_color='#94A3B8', font_weight='bold')
            st.plotly_chart(fig_cust, use_container_width=True, config={'displayModeBar': False})

        # REDESIGNED HOME GRID 2: Revenue Contribution & Top Products Side-by-Side
        col_low1, col_low2 = st.columns([2, 3])
        
        with col_low1:
            st.markdown("<div class=\"section-header\">Revenue Contribution by Customer Cohort</div>", unsafe_allow_html=True)
            fig_rev = px.pie(
                summary_df, 
                values='Revenue_Contribution_Pct', 
                names='Segment', 
                color='Segment',
                color_discrete_map={'High Value': '#FFD700', 'Regular': '#00ADB5', 'Occasional': '#005B94', 'At Risk': '#FF4D4D'},
                hole=0.55,
                height=280
            )
            fig_rev.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=10, color='#94A3B8'))
            )
            fig_rev.add_annotation(text="Revenue", x=0.5, y=0.5, font_size=11, showarrow=False, font_color='#94A3B8', font_weight='bold')
            st.plotly_chart(fig_rev, use_container_width=True, config={'displayModeBar': False})
            
        with col_low2:
            st.markdown("<div class=\"section-header\">Top 5 Revenue Generating Products</div>", unsafe_allow_html=True)
            top_prod_df = pd.DataFrame(metrics['top_5_products'])
            
            fig_prod = px.bar(
                top_prod_df,
                x='Revenue',
                y='Description',
                orientation='h',
                labels={'Description': '', 'Revenue': 'Revenue (£)'},
                color='Revenue',
                color_continuous_scale=['#0f172a', '#005b94', '#00adb5'],
                height=280
            )
            fig_prod.update_layout(
                template='plotly_dark',
                coloraxis_showscale=False,
                yaxis={'categoryorder':'total ascending', 'tickfont': dict(size=10, color='#94A3B8')},
                xaxis={'title': 'Total Revenue (£)', 'tickfont': dict(color='#94A3B8')},
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_prod, use_container_width=True, config={'displayModeBar': False})

        # Business Insights Cards at Bottom of Home Dashboard
        st.markdown("<div class=\"section-header\">Key Cohort Intelligence Summaries</div>", unsafe_allow_html=True)
        col_insight1, col_insight2, col_insight3 = st.columns(3)
        
        with col_insight1:
            st.markdown(f"""
            <div class="insight-card" style="border-left: 4px solid #FFD700; background-color: #1C1E26; border-top: 1px solid #2B2F3A; border-right: 1px solid #2B2F3A; border-bottom: 1px solid #2B2F3A; min-height: 130px;">
                <strong style="color: #FFD700; font-size: 13.5px; display: block; margin-bottom: 5px;">👑 High Value VIP Cohort</strong>
                <p style="font-size: 12px; margin: 0; line-height: 1.45; color: #94A3B8;">
                    The <strong>High Value</strong> segment contains only <strong>{summary_df[summary_df['Segment'] == 'High Value']['Customer_Count'].values[0] / metrics['total_customers'] * 100:.1f}%</strong> of our database but drives <strong>{summary_df[summary_df['Segment'] == 'High Value']['Revenue_Contribution_Pct'].values[0]:.1f}%</strong> of total sales. Retention of this group is critical for sales.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_insight2:
            st.markdown(f"""
            <div class="insight-card" style="border-left: 4px solid #00ADB5; background-color: #121A26; border-top: 1px solid #1C2B3C; border-right: 1px solid #1C2B3C; border-bottom: 1px solid #1C2B3C; min-height: 130px;">
                <strong style="color: #00ADB5; font-size: 13.5px; display: block; margin-bottom: 5px;">⭐ Regular Growth Core</strong>
                <p style="font-size: 12px; margin: 0; line-height: 1.45; color: #94A3B8;">
                    The <strong>Regular</strong> segment represents <strong>26.6%</strong> of our customers and generates <strong>26.9%</strong> of revenue (£{summary_df[summary_df['Segment'] == 'Regular']['Revenue_Contribution_Pct'].values[0]/100*metrics['total_revenue']/1e6:.2f}M). Upselling regular clients represents a key expansion vector.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_insight3:
            st.markdown(f"""
            <div class="insight-card" style="border-left: 4px solid #FF4D4D; background-color: #21161B; border-top: 1px solid #332128; border-right: 1px solid #332128; border-bottom: 1px solid #332128; min-height: 130px;">
                <strong style="color: #FF4D4D; font-size: 13.5px; display: block; margin-bottom: 5px;">⚠️ At Risk Churn Candidates</strong>
                <p style="font-size: 12px; margin: 0; line-height: 1.45; color: #94A3B8;">
                    Over <strong>{metrics['at_risk_count']:,} customers</strong> average <strong>259 days</strong> since their last order. Winning back 10% of these users with personalized promotions represents a significant recovered revenue opportunity.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ------------------ PRODUCT RECOMMENDATION ------------------
    elif page == "🛒 Product Recommendation":
        st.markdown("""
        <div class="page-section-header">
            <h2 class="page-title">🛍️ Product Recommendation Engine</h2>
            <p class="page-subtitle">Discover similar products using collaborative filtering intelligence</p>
        </div>
        """, unsafe_allow_html=True)

        selected_product = st.selectbox(
            "Search or Select a Product Name from Active Catalog:", 
            options=product_names, 
            index=0 if "WHITE HANGING HEART T-LIGHT HOLDER" not in product_names else product_names.index("WHITE HANGING HEART T-LIGHT HOLDER")
        )

        # Selected Product Card (Premium Dark Style)
        st.markdown(f"""
        <div class="custom-card" style="border-left: 4px solid #005B94; background-color: #151B30; border-top: 1px solid #222C4A; border-right: 1px solid #222C4A; border-bottom: 1px solid #222C4A;">
            <div style="font-size: 9px; text-transform: uppercase; color: #94A3B8; letter-spacing: 1.2px; font-weight:600;">Selected Item Profile</div>
            <div style="font-size: 15px; font-weight: 700; color: #FFFFFF; margin-top: 5px;">{selected_product}</div>
            <div style="font-size: 11px; color: #00ADB5; margin-top: 3px; font-weight: 500;">Status: Verified Active • Recommendation Ready (Purchased by ≥ 5 customers)</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✨ Generate AI Recommendations", type="primary"):
            product_name_clean = selected_product.strip().upper()
            all_prods = list(product_similarity.index)
            
            if product_name_clean in product_similarity.index:
                matched_product = product_name_clean
                partial_match_msg = None
            else:
                matches = difflib.get_close_matches(product_name_clean, all_prods, n=1, cutoff=0.5)
                if matches:
                    matched_product = matches[0]
                    partial_match_msg = f"Product not found. Recommending for closest match: **{matched_product}**"
                else:
                    matched_product = None
                    partial_match_msg = None
                    st.error("Error: Product not found in catalog. Please check your spelling.")

            if matched_product:
                if partial_match_msg:
                    st.info(partial_match_msg)
                
                # Fetch similarity scores
                sim_scores = product_similarity[matched_product]
                recommendations = sim_scores.drop(labels=[matched_product]).sort_values(ascending=False).head(5)
                
                st.markdown("<div class=\"section-header\">Top 5 Recommended Cross-Sell Products</div>", unsafe_allow_html=True)
                cols = st.columns(5)
                for idx, (prod, score) in enumerate(recommendations.items()):
                    # Define Match Strength Label
                    if score >= 0.70:
                        strength_lbl = "Excellent Match"
                        lbl_color = "#FFD700"  # Gold
                    elif score >= 0.50:
                        strength_lbl = "Strong Match"
                        lbl_color = "#00ADB5"  # Cyan
                    else:
                        strength_lbl = "Good Match"
                        lbl_color = "#94A3B8"  # Slate
                        
                    with cols[idx % 5]:
                        st.markdown(f"""
                        <div class="product-card">
                            <div style="font-weight: 600; font-size: 12px; color: #FFFFFF; margin-bottom: 8px; height: 65px; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; line-height: 1.35;">
                                {prod}
                            </div>
                            <div>
                                <div style="display: flex; justify-content: space-between; font-size: 10px; color: #94A3B8; font-weight: 500;">
                                    <span>Match Score</span>
                                    <span>{score*100:.1f}%</span>
                                </div>
                                <div style="background-color: #0B0F19; border-radius: 4px; height: 6px; margin: 8px 0; overflow: hidden; width: 100%;">
                                    <div style="background: linear-gradient(90deg, #005B94, #00ADB5); height: 100%; width: {score*100}%; animation: loadBar 1s ease-out;"></div>
                                </div>
                                <div style="font-size: 10px; color: {lbl_color}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 4px; text-align: left;">
                                    ● {strength_lbl}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Similarity visualization and insights side by side
                st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
                col_vis, col_ins = st.columns([3, 2])
                
                with col_vis:
                    st.markdown("<div class=\"section-header\">Recommendation Match Strengths Comparison</div>", unsafe_allow_html=True)
                    rec_df = pd.DataFrame({'Product': recommendations.index, 'Match Score (%)': recommendations.values * 100})
                    fig_match = px.bar(
                        rec_df,
                        x='Match Score (%)',
                        y='Product',
                        orientation='h',
                        color='Match Score (%)',
                        color_continuous_scale=['#0f172a', '#005b94', '#00adb5'],
                        height=240
                    )
                    fig_match.update_layout(
                        template='plotly_dark',
                        coloraxis_showscale=False, 
                        yaxis={'categoryorder':'total ascending', 'tickfont': dict(size=9, color='#94A3B8')}, 
                        xaxis={'tickfont': dict(color='#94A3B8')},
                        plot_bgcolor='rgba(0,0,0,0)', 
                        paper_bgcolor='rgba(0,0,0,0)', 
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig_match, use_container_width=True, config={'displayModeBar': False})
                    
                with col_ins:
                    st.markdown("<div class=\"section-header\">Recommendation Insights Section</div>", unsafe_allow_html=True)
                    st.markdown("""
                    <div class="insight-card" style="border-left: 4px solid #00ADB5; background-color: #121A26; border-top: 1px solid #1C2B3C; border-right: 1px solid #1C2B3C; border-bottom: 1px solid #1C2B3C; min-height: 200px;">
                        <strong style="color: #00ADB5; font-size: 13px; display: block; margin-bottom: 8px;">💡 Cross-Selling Strategy</strong>
                        <p style="font-size: 12px; margin: 0; line-height: 1.5; color: #94A3B8;">
                            These recommendations represent items frequently purchased by customers who exhibited similar buying habits. High similarity scores (above 50%) indicate highly correlated product pairs.
                        </p>
                        <p style="font-size: 12px; margin-top: 8px; line-height: 1.5; color: #94A3B8;">
                            <strong>Checkout Triggers</strong>: Auto-suggesting these items at checkout when the selected item is in the cart has a high likelihood of increasing average basket sizes and overall order value.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

    # ------------------ CUSTOMER SEGMENTATION ------------------
    elif page == "👥 Customer Segmentation":
        st.markdown("""
        <div class="page-section-header">
            <h2 class="page-title">👥 Customer Intelligence Engine</h2>
            <p class="page-subtitle">Predict customer segments using RFM behavioral analytics</p>
        </div>
        """, unsafe_allow_html=True)

        col_in, col_res = st.columns([1, 1])
        
        with col_in:
            st.markdown("<div class=\"section-header\">Adjust Customer RFM Values</div>", unsafe_allow_html=True)
            recency_in = st.slider("Recency (Days since last purchase):", min_value=1, max_value=375, value=45, help="Smaller values indicate recent, active buyers.")
            frequency_in = st.slider("Frequency (Number of invoices):", min_value=1, max_value=210, value=3, help="Higher values indicate loyal, repeat buyers.")
            monetary_in = st.number_input("Monetary Value (Total spent £):", min_value=1.00, max_value=300000.00, value=750.00, step=50.00, help="Total monetary value spent by this customer.")

            # Dynamic prediction execution on value changes - removed the button requirement!
            # Predictions run instantly!

        # Static Segment Definitions for predict UI
        segment_details = {
            'High Value': {
                'icon': '👑',
                'bg': 'linear-gradient(135deg, #FFD700 0%, #B59410 100%)',
                'border': '#FFD700',
                'badge': 'VIP Client',
                'desc': 'VIP customers who visit frequently and spend heavily.',
                'behavior': 'Low Recency (avg ~20 days), High Frequency (avg ~16 transactions), High Monetary (avg >£9,800).',
                'action': 'Enroll in premium VIP club. Offer early access to sales, premium delivery, and personalized customer care.'
            },
            'Regular': {
                'icon': '⭐',
                'bg': 'linear-gradient(135deg, #00C6FF 0%, #0072FF 100%)',
                'border': '#00ADB5',
                'badge': 'Active Core',
                'desc': 'Active, steady buyers contributing to solid sales.',
                'behavior': 'Low/Moderate Recency (avg ~46 days), Moderate Frequency (avg ~4.2 transactions), Moderate Monetary (avg ~£1,640).',
                'action': 'Upsell with related product bundles. Cross-sell complementary inventory. Invite to satisfaction surveys.'
            },
            'Occasional': {
                'icon': '🛒',
                'bg': 'linear-gradient(135deg, #13547A 0%, #80D0C7 100%)',
                'border': '#005B94',
                'badge': 'Seasonal Buyer',
                'desc': 'Infrequent buyers, visiting occasionally or during discount sales.',
                'behavior': 'Moderate Recency (avg ~58 days), Low Frequency (avg ~1.5 transactions), Low Monetary (avg ~£380).',
                'action': 'Re-engage via discount coupon codes. Send custom product suggestions. target with seasonal sale campaigns.'
            },
            'At Risk': {
                'icon': '⚠️',
                'bg': 'linear-gradient(135deg, #F05F57 0%, #9B0000 100%)',
                'border': '#FF4D4D',
                'badge': 'Churn Candidate',
                'desc': 'Lapsed customer at high risk of permanent churn.',
                'behavior': 'High Recency (avg ~259 days), Low Frequency (avg ~1.3 transactions), Low/Moderate Monetary (avg ~£385).',
                'action': 'Deploy win-back campaigns ("We miss you! 30% off your next order"). request feedback to identify customer friction.'
            }
        }

        with col_res:
            st.markdown("<div class=\"section-header\">Real-Time Classification Results</div>", unsafe_allow_html=True)
            
            # Predict and scale instantly
            freq_log = np.log1p(frequency_in)
            mon_log = np.log1p(monetary_in)
            
            # Scale input
            scaled_arr = scaler.transform([[recency_in, freq_log, mon_log]])
            
            # Predict cluster ID
            cluster_id = kmeans.predict(scaled_arr)[0]
            
            # Reconstruct mapping dynamically from KMeans model centroids
            centroids = kmeans.cluster_centers_
            high_value_id = np.argmax(centroids[:, 2])
            rem = [i for i in range(4) if i != high_value_id]
            at_risk_id = rem[np.argmax(centroids[rem, 0])]
            rem = [i for i in rem if i != at_risk_id]
            
            if centroids[rem[0], 2] > centroids[rem[1], 2]:
                regular_id = rem[0]
                occasional_id = rem[1]
            else:
                regular_id = rem[1]
                occasional_id = rem[0]
                
            segment_mapping = {
                high_value_id: 'High Value',
                regular_id: 'Regular',
                occasional_id: 'Occasional',
                at_risk_id: 'At Risk'
            }
            
            predicted_segment = segment_mapping[cluster_id]
            details = segment_details[predicted_segment]
            
            # Retrieve dynamic statistics for predicted segment
            seg_row = summary_df[summary_df['Segment'] == predicted_segment]
            cust_count = int(seg_row['Customer_Count'].values[0])
            rev_pct = float(seg_row['Revenue_Contribution_Pct'].values[0])
            
            # Premium HTML Card styled natively to prevent markdown raw code block compilation issues
            html_card = f'<div class="predict-card" style="border: 1px solid {details["border"]}; background: #151B30; padding: 22px; border-radius: 8px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">' \
                        f'<div style="font-size: 38px; float: right; margin-top: -5px;">{details["icon"]}</div>' \
                        f'<div style="font-size: 10px; text-transform: uppercase; letter-spacing: 1.2px; color: #94A3B8; font-weight:600;">AI Cohort Classifier</div>' \
                        f'<div style="font-size: 26px; font-weight: 700; margin-bottom: 3px; color: #FFFFFF; display:inline-block;">{predicted_segment}</div>' \
                        f'<span style="background-color: {details["border"]}22; color: {details["border"]}; font-size: 10px; padding: 3px 8px; border-radius: 20px; font-weight: 700; margin-left: 10px; border: 1px solid {details["border"]}; vertical-align: middle;">{details["badge"]}</span>' \
                        f'<div style="margin-top: 15px;">' \
                        f'<strong style="font-size: 12.5px; display: block; border-bottom: 1px solid #222C4A; padding-bottom: 3px; margin-bottom: 4px; color: #00ADB5;">Customer Profile Description</strong>' \
                        f'<span style="font-size: 12px; color: #94A3B8; line-height: 1.45;">{details["desc"]}</span>' \
                        f'</div>' \
                        f'<div style="margin-top: 12px;">' \
                        f'<strong style="font-size: 12.5px; display: block; border-bottom: 1px solid #222C4A; padding-bottom: 3px; margin-bottom: 4px; color: #00ADB5;">Behavioral Cohort Mean</strong>' \
                        f'<span style="font-size: 12px; color: #94A3B8; line-height: 1.45;">{details["behavior"]}</span>' \
                        f'</div>' \
                        f'<div style="margin-top: 12px;">' \
                        f'<strong style="font-size: 12.5px; display: block; border-bottom: 1px solid #222C4A; padding-bottom: 3px; margin-bottom: 4px; color: #00ADB5;">Revenue Impact Contribution</strong>' \
                        f'<span style="font-size: 12px; color: #94A3B8; line-height: 1.45;">Generates <strong>{rev_pct:.2f}%</strong> of total business revenue, representing <strong>{cust_count:,}</strong> active customers.</span>' \
                        f'</div>' \
                        f'<div style="margin-top: 15px; border-top: 1px solid #222C4A; padding-top: 12px;">' \
                        f'<strong style="font-size: 13px; display: block; margin-bottom: 4px; font-weight: 700; color: {details["border"]};">Recommended Strategic Action</strong>' \
                        f'<span style="font-size: 12px; color: #E2E8F0; line-height: 1.45;">{details["action"]}</span>' \
                        f'</div>' \
                        f'</div>'
            
            st.markdown(html_card, unsafe_allow_html=True)

    # ------------------ BUSINESS INSIGHTS ------------------
    elif page == "📊 Business Insights":
        st.markdown("""
        <div class="page-section-header">
            <h2 class="page-title">📈 Strategic Business Intelligence</h2>
            <p class="page-subtitle">Customer insights, revenue analysis, and growth opportunities</p>
        </div>
        """, unsafe_allow_html=True)

        # Segment Summary Table
        st.markdown("<div class=\"section-header\">Customer Segment Summary Table</div>", unsafe_allow_html=True)
        styled_df = summary_df.copy()
        styled_df.columns = ['Segment', 'Customer Count', 'Revenue Contribution %', 'Average Recency (Days)', 'Average Frequency (Invoices)', 'Average Monetary Spend (£)']
        styled_df['Revenue Contribution %'] = styled_df['Revenue Contribution %'].map("{:.2f}%".format)
        styled_df['Average Recency (Days)'] = styled_df['Average Recency (Days)'].map("{:.1f}".format)
        styled_df['Average Frequency (Invoices)'] = styled_df['Average Frequency (Invoices)'].map("{:.2f}".format)
        styled_df['Average Monetary Spend (£)'] = styled_df['Average Monetary Spend (£)'].map("£{:,.2f}".format)
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        col_bi1, col_bi2 = st.columns(2)
        
        with col_bi1:
            st.markdown("<div class=\"section-header\">Key Cohort Financial Findings</div>", unsafe_allow_html=True)
            
            # Gold theme for High Value
            st.markdown(f"""
            <div class="insight-card" style="border-left: 5px solid #FFD700; background-color: #1C1E26; border-top: 1px solid #2B2F3A; border-right: 1px solid #2B2F3A; border-bottom: 1px solid #2B2F3A;">
                <h5 style="margin:0; color:#FFD700; font-size: 14px; font-weight:700;">👑 High Value VIP Cohort (Pareto Impact)</h5>
                <p style="font-size:12px; margin-top:8px; line-height:1.5; color: #94A3B8;">
                    Our **High Value** customer cohort comprises only **{summary_df[summary_df['Segment'] == 'High Value']['Customer_Count'].values[0] / metrics['total_customers'] * 100:.2f}%** of the database, but drives **{summary_df[summary_df['Segment'] == 'High Value']['Revenue_Contribution_Pct'].values[0]:.2f}%** of total sales. 
                    A 5% drop in this cohort represents an immediate cash leakage of over **£280,000**. Retention of this cohort is our highest priority.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Cyan theme for Regulars
            st.markdown("""
            <div class="insight-card" style="border-left: 5px solid #00ADB5; background-color: #121A26; border-top: 1px solid #1C2B3C; border-right: 1px solid #1C2B3C; border-bottom: 1px solid #1C2B3C;">
                <h5 style="margin:0; color:#00ADB5; font-size: 14px; font-weight:700;">⭐ Regular Core Cohort (Expansion Focus)</h5>
                <p style="font-size:12px; margin-top:8px; line-height:1.5; color: #94A3B8;">
                    The **Regular** segment represents **26.6%** of our active customer base and generates **26.9%** of revenue. 
                    This is our most expandable cohort. Shifting 10% of Regular buyers into VIP spending brackets via loyalty campaigns raises revenue by over **£150,000**.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Coral/Red theme for At Risk
            st.markdown(f"""
            <div class="insight-card" style="border-left: 5px solid #FF4D4D; background-color: #21161B; border-top: 1px solid #332128; border-right: 1px solid #332128; border-bottom: 1px solid #332128;">
                <h5 style="margin:0; color:#FF4D4D; font-size: 14px; font-weight:700;">⚠️ At Risk Cohort (Churn Exposure)</h5>
                <p style="font-size:12px; margin-top:8px; line-height:1.5; color: #94A3B8;">
                    Over **{metrics['at_risk_count']:,} customers** reside in the **At Risk** cohort, averaging **259 days** since their last purchase. 
                    Deploying automated win-back triggers is highly recommended to recover a fraction of this lapsed user group.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_bi2:
            st.markdown("<div class=\"section-header\">Suggested Actionable Campaigns</div>", unsafe_allow_html=True)
            
            # Gold theme
            st.markdown("""
            <div class="insight-card" style="border-left: 5px solid #FFD700; background-color: #1C1E26; border-top: 1px solid #2B2F3A; border-right: 1px solid #2B2F3A; border-bottom: 1px solid #2B2F3A;">
                <strong style="color: #FFD700; font-size:13px; display:block; margin-bottom: 4px;">👑 High Value Loyalty Campaigns:</strong>
                <ul style="font-size: 12px; padding-left: 18px; margin: 0; line-height: 1.45; color: #94A3B8;">
                    <li>Provide complimentary express shipping on all orders.</li>
                    <li>Invite to early access product previews and special catalogs.</li>
                    <li>Provide dedicated VIP customer support channels and account managers.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Cyan theme
            st.markdown("""
            <div class="insight-card" style="border-left: 5px solid #00ADB5; background-color: #121A26; border-top: 1px solid #1C2B3C; border-right: 1px solid #1C2B3C; border-bottom: 1px solid #1C2B3C;">
                <strong style="color: #00ADB5; font-size:13px; display:block; margin-bottom: 4px;">⭐ Regular Upselling Programs:</strong>
                <ul style="font-size: 12px; padding-left: 18px; margin: 0; line-height: 1.45; color: #94A3B8;">
                    <li>Suggest matching accessories / item bundles at checkout.</li>
                    <li>Reward basket upgrades (e.g. "Spend £100 for £15 cash back").</li>
                    <li>Trigger seasonal cross-category recommendations.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Deep Blue theme
            st.markdown("""
            <div class="insight-card" style="border-left: 5px solid #005B94; background-color: #0F1626; border-top: 1px solid #1C283B; border-right: 1px solid #1C283B; border-bottom: 1px solid #1C283B;">
                <strong style="color: #0088DB; font-size:13px; display:block; margin-bottom: 4px;">🛒 Occasional Re-Engagement:</strong>
                <ul style="font-size: 12px; padding-left: 18px; margin: 0; line-height: 1.45; color: #94A3B8;">
                    <li>Distribute holiday newsletters featuring top revenue generators.</li>
                    <li>Send flash discounts (e.g. "48-hour 15% off coupon").</li>
                    <li>A/B test pricing variations on favorite items.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # ------------------ FOOTER ------------------
    st.markdown("""
    <div style="margin-top: 40px; border-top: 1px solid #222C4A; padding-top: 20px; text-align: center; color: #94A3B8; font-size: 11px; margin-bottom: 10px;">
        <div style="font-weight: 700; color: #FFFFFF; font-size: 13px; margin-bottom: 3px;">🛒 Shopper Spectrum</div>
        <div style="margin-bottom: 8px; font-weight: 500;">AI-Powered Customer Segmentation & Product Recommendation Platform</div>
        <div style="margin-bottom: 4px;">Developed By: <strong>Abhishek Kulkarni</strong></div>
        <div style="margin-bottom: 4px; color: #64748B;">Technologies: Python • Streamlit • Scikit-Learn • KMeans • Collaborative Filtering</div>
        <div style="margin-bottom: 8px; color: #64748B;">Dataset: Online Retail</div>
        <div style="color: #475569; font-size: 10px;">© 2026 Shopper Spectrum. All rights reserved.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("Please verify that the datasets and pipeline script are executed successfully to launch the dashboard.")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Analytics Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force Light Theme styling for metric cards
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa; /* Light grey for light theme */
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Adjust text color for light theme */
    .stMetricLabel { color: #444; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Load Spark output files
    df_ml = pd.read_csv("ml_global_prediction.csv")
    df_countries = pd.read_csv("country_insight_full.csv")
    return df_ml, df_countries

try:
    df_trend, df_countries = load_data()
except FileNotFoundError:
    st.error("⚠️ CSV files not found! Please run the Spark script in Colab first and place 'ml_global_prediction.csv' and 'country_insight_full.csv' in this folder.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Control Panel")
    st.write("Filter data for Country Analysis (Module 13):")
    
    # Filter Continent
    all_continents = sorted(df_countries['continent'].unique())
    selected_continents = st.multiselect(
        "Select Continent", 
        all_continents, 
        default=all_continents
    )
    
    # Filter logic
    if selected_continents:
        df_filtered = df_countries[df_countries['continent'].isin(selected_continents)]
    else:
        df_filtered = df_countries
        
    st.divider()
    st.info("""
    **About Dashboard:**
    - **Module 11:** Global Trend Prediction using *Random Forest*.
    - **Module 13:** Interactive Visualization & Geo-Distribution.
    """)

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD LAYOUT
# -----------------------------------------------------------------------------

# --- HEADER SECTION ---
st.title("🛡️ Global COVID-19 Intelligence Dashboard")
st.markdown("Big Data analytics dashboard to monitor pandemic trends, AI-based predictions, and vaccination correlations.")

# --- DYNAMIC KPIs (Based on Filter) ---
col1, col2, col3, col4 = st.columns(4)

total_cases = df_filtered['total_cases'].sum()
total_deaths = df_filtered['total_deaths'].sum()
avg_vax_rate = df_filtered['vaccination_rate'].mean()
# Handle case if filter is empty
if not df_filtered.empty:
    top_country = df_filtered.loc[df_filtered['total_cases'].idxmax()]['location']
else:
    top_country = "-"

col1.metric("Total Cases (Selected)", f"{total_cases:,.0f}", "Cumulative")
col2.metric("Total Deaths", f"{total_deaths:,.0f}", "Cumulative")
col3.metric("Avg Vaccination Rate", f"{avg_vax_rate:.1f}%", "Fully Vaccinated Pop")
col4.metric("#1 Most Impacted", top_country, "Highest Cases")

st.divider()

# --- TABS FOR STORYTELLING ---
tab1, tab2, tab3 = st.tabs([
    "📈 Module 11: Global Prediction (AI)", 
    "🗺️ Module 13: Map & Distribution", 
    "🔬 Insight: Vax vs Deaths"
])

# === TAB 1: GLOBAL PREDICTION (Module 11) ===
with tab1:
    st.subheader("Global Trend Prediction (Random Forest Model)")
    st.write("The Machine Learning model was trained on 'World' data to predict future case waves.")
    
    # Plotly Time Series
    fig_ml = go.Figure()
    
    # Actual Data
    fig_ml.add_trace(go.Scatter(
        x=df_trend['date'], y=df_trend['new_cases_smoothed'],
        mode='lines', name='Actual (Smoothed)',
        line=dict(color='#3366CC', width=2)
    ))
    
    # Prediction Data
    fig_ml.add_trace(go.Scatter(
        x=df_trend['date'], y=df_trend['prediction'],
        mode='lines', name='AI Prediction (Random Forest)',
        line=dict(color='#DC3912', width=2, dash='dash')
    ))
    
    fig_ml.update_layout(
        template="plotly_white", # Force Light Theme
        height=450,
        xaxis_title="Date",
        yaxis_title="Daily New Cases",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_ml, use_container_width=True)
    
    # Model Evaluation Metric
    last_row = df_trend.iloc[-1]
    err = last_row['new_cases_smoothed'] - last_row['prediction']
    st.info(f"💡 **Model Status:** On the last available date, the difference between prediction and actual is **{err:,.0f}** cases.")

# === TAB 2: GEOGRAPHIC ANALYSIS (Module 13) ===
with tab2:
    st.subheader(f"COVID-19 Distribution Map ({', '.join(selected_continents) if selected_continents else 'All Continents'})")
    
    col_map_metrics = st.selectbox("Select Map Metric:", 
                                  ["total_cases", "total_deaths", "vaccination_rate"], 
                                  format_func=lambda x: x.replace("_", " ").title())
    
    # Choropleth Map
    fig_map = px.choropleth(
        df_filtered,
        locations="iso_code",
        color=col_map_metrics,
        hover_name="location",
        color_continuous_scale="Reds" if col_map_metrics != "vaccination_rate" else "Greens",
        title=f"World Map: {col_map_metrics.replace('_', ' ').title()}",
        template="plotly_white" # Force Light Theme
    )
    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Top 10 Bar Chart
    st.subheader("Top 10 Countries in Selection")
    top10 = df_filtered.sort_values(by=col_map_metrics, ascending=False).head(10)
    fig_bar = px.bar(
        top10, 
        x=col_map_metrics, 
        y="location", 
        orientation='h',
        color=col_map_metrics,
        text_auto='.2s',
        title=f"Top 10 Countries by {col_map_metrics.replace('_', ' ').title()}",
        template="plotly_white" # Force Light Theme
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

# === TAB 3: CORRELATION INSIGHTS ===
with tab3:
    st.subheader("Correlation Analysis: Does Vaccination Lower Mortality?")
    st.markdown("The Bubble Chart below shows the relationship between **Vaccination Rate (X)** and **Mortality Rate (Y)**. Bubble size represents Population.")
    
    # Bubble Chart
    fig_bubble = px.scatter(
        df_filtered,
        x="vaccination_rate",
        y="mortality_rate",
        size="population",
        color="continent",
        hover_name="location",
        log_x=False,
        size_max=60,
        title="Vaccination vs Mortality Rate",
        labels={"vaccination_rate": "Vaccination Rate (%)", "mortality_rate": "Mortality Rate (%)"},
        template="plotly_white" # Force Light Theme
    )
    
    # Reference Lines
    if not df_filtered.empty:
        fig_bubble.add_hline(y=df_filtered['mortality_rate'].mean(), line_dash="dot", annotation_text="Avg Mortality")
        fig_bubble.add_vline(x=df_filtered['vaccination_rate'].mean(), line_dash="dot", annotation_text="Avg Vax Rate")
    
    st.plotly_chart(fig_bubble, use_container_width=True)
    
    st.success("""
    **How to Read:**
    - **Bottom Right Quadrant:** (High Vax, Low Mortality) -> Ideal Condition.
    - **Top Left Quadrant:** (Low Vax, High Mortality) -> Needs Attention.
    """)
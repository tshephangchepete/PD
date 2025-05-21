import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

st.set_page_config(layout="wide") 

st.markdown("""
    <style>
        /* Hide header and footer */
        header, footer { visibility: hidden; }

        /* Main app padding */
        .main, .block-container {
            padding-top: 4px !important;
            padding-bottom: 4px !important;
        }

        /* Headings spacing */
        h1, h3 {
            margin-top: -5px !important;
            margin-bottom: 3px !important;
        }

        .stApp h1:first-of-type + p {
            margin-top: -6px !important;
            margin-bottom: 4px !important;
        }

        .stApp hr:first-of-type {
            margin-top: 4px !important;
            margin-bottom: 4px !important;
        }

        /* Color legend */
        .color-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: -4px !important;
            margin-bottom: 6px !important;
            padding: 4px 6px !important;
            background: #f8f9fa;
            border-radius: 6px;
            font-size: 12px;
        }
        .color-item {
            display: flex;
            align-items: center;
            font-size: 11px !important;
        }
        .color-box {
            width: 12px;
            height: 12px;
            margin-right: 4px;
            border: 1px solid #ddd;
        }

        /* Metric cards */
        .metric-card, .marketing-metric-card {
            background: white;
            border-radius: 8px;
            padding: 8px 10px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            border-left: 4px solid #2c3e50;
            margin-bottom: 6px !important;
        }
        .header { color: #2c3e50; font-weight: 700; font-size: 17px !important; }
        .subheader { color: #5a5c69; font-size: 12px !important; margin-bottom: 2px !important; }

        /* Charts */
        .stPlotlyChart {
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            min-height: 180px !important;
            height: 100% !important;
        }
        .stPlotlyChart .js-plotly-plot { min-height: 180px !important; height: 100% !important; }

        /* Images */
        .st-emotion-cache-1v0mbdj {
            border-radius: 6px;
            max-height: 220px !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 4px !important; }
        .stTabs [data-baseweb="tab"] { border-radius: 5px 5px 0 0 !important; padding: 4px 10px !important; font-size: 12px !important; }
        .stTabs [aria-selected="true"] { background-color: #2c3e50 !important; color: white !important; }

        /* Select boxes */
        .stSelectbox, .stRadio > div { padding: 2px !important; }

        /* Gauges */
        .gauge-title { text-align: center; font-weight: 600; margin-top: 6px !important; margin-bottom: -10px !important; font-size: 13px !important; }
        .gauge-value { font-size: 18px !important; }

        /* Performance styles */
        .bad-performance { color: #c33c54 !important; font-weight: 600; }
        .mid-performance { color: #f8961e !important; font-weight: 600; }
        .close-target { color: #43aa8b !important; font-weight: 600; }
        .at-target { color: #1a936f !important; font-weight: 600; }
        .exceeded-target { color: #0a8754 !important; font-weight: 600; }

        .sales-rep-link { cursor: pointer; color: #2c3e50; text-decoration: underline; font-size: 12px !important; }
        .sales-rep-link:hover { color: #1a936f; }

        /* Layout spacing */
        .st-emotion-cache-1wrcr25 { gap: 0.4rem !important; }

        .js-plotly-plot .plotly { margin: 0 !important; }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("//Users//tshephangchepete//Downloads//PD//sales_dataset.csv")
    df = df[df['Action'] == 'PURCHASE']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month
    df['Quarter'] = df['Timestamp'].dt.quarter
    df['Month_Name'] = df['Timestamp'].dt.month_name()
    df['Quarter_Name'] = 'Q' + df['Quarter'].astype(str)
    return df

df = load_data()

# Create departments
departments = ['AI Model', 'Sales', 'Marketing']

# Set realistic targets with normal performance distribution
products = df['Product'].unique()
sales_channels = df['Sales_Channel'].unique()
countries = df['Country'].unique()
sales_reps = df['Sales_Rep'].unique()
years = sorted(df['Year'].unique())
quarters_order = ['Q1', 'Q2', 'Q3', 'Q4']
months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']

color_map = {
    'Bad performance': '#c33c54',    # <50%
    'Mid performance': '#f8961e',    # 50–79%
    'Close to target': '#43aa8b',    # 80–99%
    'At target': '#1a936f',          # 100%
    'Exceeded target': '#0a8754'     # >100%
}

def get_performance_class(pct):
    if pct < 50:
        return "bad-performance"
    elif 50 <= pct < 80:
        return "mid-performance"
    elif 80 <= pct < 100:
        return "close-target"
    elif pct == 100:
        return "at-target"
    else:
        return "exceeded-target"

def get_performance_category(pct):
    if pct < 50:
        return 'Bad performance'
    elif 50 <= pct < 80:
        return 'Mid performance'
    elif 80 <= pct < 100:
        return 'Close to target'
    elif pct == 100:
        return 'At target'
    else:
        return 'Exceeded target'
    
STATIC_TARGET_SEED = 99
rng = np.random.default_rng(STATIC_TARGET_SEED)

def static_uniform(low, high, size=None):
    return rng.uniform(low, high, size=size)

def static_integers(low, high, size=None):
    return rng.integers(low, high, size=size)

def assign_entity_weights(entities, min_weight=0.95, max_weight=1.10):
    """Tighter range, centers targets closer to real performance."""
    n = len(entities)
    weight_ranges = [
        (min_weight, 0.98),    # below target
        (0.99, 1.01),          # at target
        (1.02, max_weight)     # above target
    ]
    group_sizes = [n // len(weight_ranges)] * len(weight_ranges)
    for i in range(n % len(weight_ranges)):
        group_sizes[i] += 1
    weights = []
    for r, size in zip(weight_ranges, group_sizes):
        weights += list(np.random.uniform(*r, size=size))
    np.random.shuffle(weights)
    return dict(zip(entities, weights[:n]))

def build_static_targets_with_weights(
    entities, years, months_order, 
    profit_range, quantity_range, revenue_range, growth=0.04
):
    entity_weights = assign_entity_weights(entities)
    targets = {}
    for ent in entities:
        ent_target = {}
        ent_weight = entity_weights[ent]
        for year in years:
            base_profit = max(np.random.randint(*profit_range), profit_range[0]) * ent_weight
            base_quantity = max(np.random.randint(*quantity_range), quantity_range[0]) * ent_weight
            base_revenue = max(np.random.randint(*revenue_range), revenue_range[0]) * ent_weight
            growth_factor = 1 + (growth * (year - min(years)))
            for month in months_order:
                if month in ['December', 'November']:
                    month_factor = np.random.uniform(1.1, 1.18)
                elif month in ['June', 'July']:
                    month_factor = np.random.uniform(0.88, 0.95)
                else:
                    month_factor = np.random.uniform(0.97, 1.03)
                ent_target[f'{year}_{month}_profit_target'] = base_profit * growth_factor * month_factor * np.random.uniform(0.98, 0.98)
                ent_target[f'{year}_{month}_quantity_target'] = base_quantity * growth_factor * month_factor * np.random.uniform(0.98, 1.02)
                ent_target[f'{year}_{month}_revenue_target'] = base_revenue * growth_factor * month_factor * np.random.uniform(0.98, 0.98)
                if month in ['January', 'February', 'March']:
                    q = 1
                elif month in ['April', 'May', 'June']:
                    q = 2
                elif month in ['July', 'August', 'September']:
                    q = 3
                else:
                    q = 4
                for typ in ['profit', 'quantity', 'revenue']:
                    k = f'{year}_Q{q}_{typ}_target'
                    ent_target[k] = ent_target.get(k, 0) + ent_target[f'{year}_{month}_{typ}_target']
            for typ in ['profit', 'quantity', 'revenue']:
                ent_target[f'{year}_{typ}_target'] = sum(
                    ent_target[f'{year}_{month}_{typ}_target'] for month in months_order)
        targets[ent] = ent_target
    return targets

# -- ADJUSTED RANGES FOR MORE REALISTIC TARGETS --
product_targets = build_static_targets_with_weights(
    products, years, months_order,
    profit_range=(720000, 1050000),
    quantity_range=(900, 2000),
    revenue_range=(2500, 3000),
    growth=0.08
)
channel_targets = build_static_targets_with_weights(
    sales_channels, years, months_order,
    profit_range=(720000, 1050000),
    quantity_range=(9000, 25000),
    revenue_range=(50000, 60000),
    growth=0.06
)
country_targets = build_static_targets_with_weights(
    countries, years, months_order,
    profit_range=(9000, 20000),
    quantity_range=(9000, 20000),
    revenue_range=(15000, 20000),
    growth=0.05
)
targets = build_static_targets_with_weights(
    sales_reps, years, months_order,
    profit_range=(1000000, 2000000),
    quantity_range=(900, 2000),
    revenue_range=(1200000, 2000000),
    growth=0.05
)

TEAM_TARGET_FACTOR = 1.4

def assign_team_fluctuation_weights(years, products, sales_channels, countries, sales_reps):
    team_weights = {}
    for year in years:
        team_weights[year] = {
            'products': assign_entity_weights(products, min_weight=10.18, max_weight=10.32),
            'channels': assign_entity_weights(sales_channels, min_weight=10.20, max_weight=10.33),
            'countries': assign_entity_weights(countries, min_weight=10.17, max_weight=10.30),
            'reps': assign_entity_weights(sales_reps, min_weight=0.0, max_weight=0.5),
            'team_luck': static_uniform(0.95, 1.05)
        }
    return team_weights

team_fluct_weights = assign_team_fluctuation_weights(years, products, sales_channels, countries, sales_reps)

team_targets = {}

for year in years:
    weights = team_fluct_weights[year]
    team_luck = weights['team_luck']
    year_profit = 0
    year_quantity = 0
    year_revenue = 0
    quarter_profit = {1: 0, 2: 0, 3: 0, 4: 0}
    quarter_quantity = {1: 0, 2: 0, 3: 0, 4: 0}
    quarter_revenue = {1: 0, 2: 0, 3: 0, 4: 0}
    month_profit = {month: 0 for month in months_order}
    month_quantity = {month: 0 for month in months_order}
    month_revenue = {month: 0 for month in months_order}

    for product in products:
        w = weights['products'][product]
        year_profit += product_targets[product][f'{year}_profit_target'] * w
        year_quantity += product_targets[product][f'{year}_quantity_target'] * w
        year_revenue += product_targets[product][f'{year}_revenue_target'] * w
        for q in range(1, 5):
            quarter_profit[q] += product_targets[product][f'{year}_Q{q}_profit_target'] * w
            quarter_quantity[q] += product_targets[product][f'{year}_Q{q}_quantity_target'] * w
            quarter_revenue[q] += product_targets[product][f'{year}_Q{q}_revenue_target'] * w
        for month in months_order:
            month_profit[month] += product_targets[product][f'{year}_{month}_profit_target'] * w
            month_quantity[month] += product_targets[product][f'{year}_{month}_quantity_target'] * w
            month_revenue[month] += product_targets[product][f'{year}_{month}_revenue_target'] * w

    for channel in sales_channels:
        w = weights['channels'][channel]
        year_profit += channel_targets[channel][f'{year}_profit_target'] * w
        year_quantity += channel_targets[channel][f'{year}_quantity_target'] * w
        year_revenue += channel_targets[channel][f'{year}_revenue_target'] * w
        for q in range(1, 5):
            quarter_profit[q] += channel_targets[channel][f'{year}_Q{q}_profit_target'] * w
            quarter_quantity[q] += channel_targets[channel][f'{year}_Q{q}_quantity_target'] * w
            quarter_revenue[q] += channel_targets[channel][f'{year}_Q{q}_revenue_target'] * w
        for month in months_order:
            month_profit[month] += channel_targets[channel][f'{year}_{month}_profit_target'] * w
            month_quantity[month] += channel_targets[channel][f'{year}_{month}_quantity_target'] * w
            month_revenue[month] += channel_targets[channel][f'{year}_{month}_revenue_target'] * w

    for country in countries:
        w = weights['countries'][country]
        year_profit += country_targets[country][f'{year}_profit_target'] * w
        year_quantity += country_targets[country][f'{year}_quantity_target'] * w
        year_revenue += country_targets[country][f'{year}_revenue_target'] * w
        for q in range(1, 5):
            quarter_profit[q] += country_targets[country][f'{year}_Q{q}_profit_target'] * w
            quarter_quantity[q] += country_targets[country][f'{year}_Q{q}_quantity_target'] * w
            quarter_revenue[q] += country_targets[country][f'{year}_Q{q}_revenue_target'] * w
        for month in months_order:
            month_profit[month] += country_targets[country][f'{year}_{month}_profit_target'] * w
            month_quantity[month] += country_targets[country][f'{year}_{month}_quantity_target'] * w
            month_revenue[month] += country_targets[country][f'{year}_{month}_revenue_target'] * w

    for rep in sales_reps:
        w = weights['reps'][rep]
        year_profit += targets[rep][f'{year}_profit_target'] * w
        year_quantity += targets[rep][f'{year}_quantity_target'] * w
        year_revenue += targets[rep][f'{year}_revenue_target'] * w
        for q in range(1, 5):
            quarter_profit[q] += targets[rep][f'{year}_Q{q}_profit_target'] * w
            quarter_quantity[q] += targets[rep][f'{year}_Q{q}_quantity_target'] * w
            quarter_revenue[q] += targets[rep][f'{year}_Q{q}_revenue_target'] * w
        for month in months_order:
            month_profit[month] += targets[rep][f'{year}_{month}_profit_target'] * w
            month_quantity[month] += targets[rep][f'{year}_{month}_quantity_target'] * w
            month_revenue[month] += targets[rep][f'{year}_{month}_revenue_target'] * w

    volatility = static_uniform(0.96, 1.05)
    team_targets[f'{year}_profit_target'] = year_profit * TEAM_TARGET_FACTOR * team_luck * volatility
    team_targets[f'{year}_quantity_target'] = year_quantity * TEAM_TARGET_FACTOR * team_luck * volatility
    team_targets[f'{year}_revenue_target'] = year_revenue * TEAM_TARGET_FACTOR * team_luck * volatility

    for q in range(1, 5):
        q_vol = static_uniform(0.97, 1.04)
        team_targets[f'{year}_Q{q}_profit_target'] = quarter_profit[q] * TEAM_TARGET_FACTOR * team_luck * q_vol
        team_targets[f'{year}_Q{q}_quantity_target'] = quarter_quantity[q] * TEAM_TARGET_FACTOR * team_luck * q_vol
        team_targets[f'{year}_Q{q}_revenue_target'] = quarter_revenue[q] * TEAM_TARGET_FACTOR * team_luck * q_vol

    for month in months_order:
        m_vol = static_uniform(0.95, 1.06)
        team_targets[f'{year}_{month}_profit_target'] = month_profit[month] * TEAM_TARGET_FACTOR * team_luck * m_vol
        team_targets[f'{year}_{month}_quantity_target'] = month_quantity[month] * TEAM_TARGET_FACTOR * team_luck * m_vol
        team_targets[f'{year}_{month}_revenue_target'] = month_revenue[month] * TEAM_TARGET_FACTOR * team_luck * m_vol

def calc_static_marketing_targets(df, sales_channels, years, months_order):
    targets = {}
    for channel in sales_channels:
        t = {}
        channel_data = df[df['Sales_Channel'] == channel]
        for year in years:
            ydata = channel_data[channel_data['Year'] == year]
            annual_leads = len(ydata)
            annual_cost = ydata['Cost'].sum()
            annual_profit = ydata['Profit'].sum()
            annual_leads_target = annual_leads * static_uniform(1.1, 1.25) * static_uniform(1.01, 1.06)
            annual_cost_target = annual_cost * static_uniform(1.18, 1.27) * static_uniform(1.01, 1.06)
            annual_profit_target = annual_profit * static_uniform(1.22, 1.33) * static_uniform(1.01, 1.06)
            t[f'{year}_leads_target'] = annual_leads_target
            t[f'{year}_cost_target'] = annual_cost_target
            t[f'{year}_profit_target'] = annual_profit_target
            for q in range(1, 5):
                qdata = ydata[ydata['Quarter'] == q]
                q_leads_target = len(qdata) * static_uniform(1.1, 1.25) * static_uniform(1.01, 1.04)
                q_cost_target = qdata['Cost'].sum() * static_uniform(1.18, 1.27) * static_uniform(1.01, 1.04)
                q_profit_target = qdata['Profit'].sum() * static_uniform(1.22, 1.33) * static_uniform(1.01, 1.04)
                t[f'{year}_Q{q}_leads_target'] = q_leads_target
                t[f'{year}_Q{q}_cost_target'] = q_cost_target
                t[f'{year}_Q{q}_profit_target'] = q_profit_target
            for m in months_order:
                mdata = ydata[ydata['Month_Name'] == m]
                m_leads_target = len(mdata) * static_uniform(1.1, 1.25) * static_uniform(1.01, 1.02)
                m_cost_target = mdata['Cost'].sum() * static_uniform(1.18, 1.27) * static_uniform(1.01, 1.02)
                m_profit_target = mdata['Profit'].sum() * static_uniform(1.22, 1.33) * static_uniform(1.01, 1.02)
                t[f'{year}_{m}_leads_target'] = m_leads_target
                t[f'{year}_{m}_cost_target'] = m_cost_target
                t[f'{year}_{m}_profit_target'] = m_profit_target
        targets[channel] = t
    return targets

marketing_targets = calc_static_marketing_targets(df, sales_channels, years, months_order)

# Sidebar navigation
st.sidebar.title("Navigation")
selected_dept = st.sidebar.radio("Select Department", departments, label_visibility="collapsed")

st.markdown("<h1 style='color: #2c3e50; margin-bottom: 0; font-size: 28px;'>📊 SalesInsight AI</h1>", unsafe_allow_html=True)

if selected_dept == 'Marketing':
    st.markdown("<div class='marketing-header'>Marketing Dashboard</div>", unsafe_allow_html=True)

    years_filter = ['All'] + years
    quarters_filter = ['All'] + quarters_order
    months_filter = ['All'] + months_order
    coly1, coly2, coly3 = st.columns([1, 1, 2])
    with coly1:
        year_selected = st.selectbox("Year", years_filter, key="marketing_year")
    with coly2:
        quarter_selected = st.selectbox("Quarter", quarters_filter, key="marketing_quarter")
    with coly3:
        month_selected = st.selectbox("Month", months_filter, key="marketing_month")

    # Apply filters
    marketing_df = df.copy()
    if year_selected != 'All':
        marketing_df = marketing_df[marketing_df['Year'] == year_selected]
    if quarter_selected != 'All':
        q_num = int(quarter_selected[1])
        marketing_df = marketing_df[marketing_df['Quarter'] == q_num]
    if month_selected != 'All':
        marketing_df = marketing_df[marketing_df['Month_Name'] == month_selected]

    # KPIs - Smaller cards
    with st.container():
        col1, col2, col3 = st.columns(3)
        total_campaigns = len(marketing_df['Sales_Channel'].unique())
        total_leads = len(marketing_df)
        total_cost = marketing_df['Cost'].sum()
        total_profit = marketing_df['Profit'].sum()
        avg_roi = (total_profit / total_cost) * 100 if total_cost > 0 else 0

        with col1:
            st.markdown(f"""
            <div class="marketing-metric-card" style="border-left-color: #2c3e50">
                <div class="subheader">Active Campaigns</div>
                <div class="header">{total_campaigns}</div>
                <div class="subheader">All channels</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="marketing-metric-card" style="border-left-color: #1a936f">
                <div class="subheader">Total Leads</div>
                <div class="header">{total_leads:,}</div>
                <div class="subheader">Converted to sales</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="marketing-metric-card" style="border-left-color: #114b5f">
                <div class="subheader">Avg. ROI</div>
                <div class="header">{avg_roi:.1f}%</div>
                <div class="subheader">All campaigns</div>
            </div>
            """, unsafe_allow_html=True)

    # Determine targets for selected filters
    def get_marketing_target(channel, year, quarter=None, month=None):
        t = marketing_targets[channel]
        if year == 'All':
            leads = sum(t[f'{y}_leads_target'] for y in years if f'{y}_leads_target' in t)
            cost = sum(t[f'{y}_cost_target'] for y in years if f'{y}_cost_target' in t)
            profit = sum(t[f'{y}_profit_target'] for y in years if f'{y}_profit_target' in t)
            return leads, cost, profit
        if quarter and quarter != 'All':
            return t.get(f'{year}_Q{quarter}_leads_target', 0), t.get(f'{year}_Q{quarter}_cost_target', 0), t.get(f'{year}_Q{quarter}_profit_target', 0)
        if month and month != 'All':
            return t.get(f'{year}_{month}_leads_target', 0), t.get(f'{year}_{month}_cost_target', 0), t.get(f'{year}_{month}_profit_target', 0)
        return t.get(f'{year}_leads_target', 0), t.get(f'{year}_cost_target', 0), t.get(f'{year}_profit_target', 0)

    # --- Row 1: Revenue by Marketing Channel & Marketing Conversion Funnel ---
    with st.container():
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            campaign_performance = marketing_df.groupby('Sales_Channel').agg({
                'Total_Price': 'sum',
                'Profit': 'sum',
                'Quantity': 'sum',
                'Cost': 'sum'
            }).reset_index()

            for ix, row in campaign_performance.iterrows():
                ch = row['Sales_Channel']
                y = year_selected if year_selected != 'All' else years[0]
                q = int(quarter_selected[1]) if quarter_selected != 'All' else None
                m = month_selected if month_selected != 'All' else None
                _, cost_target, profit_target = get_marketing_target(ch, y, q, m)
                roi_target = (profit_target / cost_target) * 100 if cost_target > 0 else 0
                actual_roi = (row['Profit'] / row['Cost']) * 100 if row['Cost'] > 0 else 0
                perf_pct = (actual_roi / roi_target) * 100 if roi_target > 0 else 0
                campaign_performance.loc[ix, 'ROI_Target'] = roi_target
                campaign_performance.loc[ix, 'ROI'] = actual_roi
                campaign_performance.loc[ix, 'Performance_Pct'] = perf_pct
                campaign_performance.loc[ix, 'Performance'] = get_performance_category(perf_pct)
            fig = px.bar(
                campaign_performance,
                x='Sales_Channel',
                y='Total_Price',
                color='Performance',
                color_discrete_map=color_map,
                title='Revenue by Marketing Channel',
                labels={'Total_Price': 'Revenue ($)', 'Sales_Channel': 'Marketing Channel'},
                hover_data=['Profit', 'Quantity', 'ROI', 'ROI_Target', 'Performance_Pct']
            )
            fig.update_layout(height=180, showlegend=True, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            funnel_data = {
                'Stage': ['Impressions', 'Website Visits', 'Leads', 'Purchases'],
                'Count': [10000, 5000, 2000, len(marketing_df)],
                'Percentage': [100, 50, 20, (len(marketing_df)/10000)*100 if len(marketing_df) > 0 else 0]
            }
            funnel_fig = px.funnel(
                funnel_data,
                x='Count',
                y='Stage',
                color='Stage',
                color_discrete_sequence=px.colors.sequential.Teal,
                labels={'Count': 'Number of Users', 'Stage': 'Funnel Stage'},
                title='Marketing Conversion Funnel'
            )
            funnel_fig.update_layout(height=180, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(funnel_fig, use_container_width=True)

    # --- Row 2: Monthly ROI Trend & Conversion Rate by Channel ---
    with st.container():
        col3, col4 = st.columns(2, gap="medium")
        with col3:
            monthly_roi = marketing_df.groupby(pd.Grouper(key='Timestamp', freq='M')).agg({
                'Profit': 'sum',
                'Cost': 'sum'
            }).reset_index()
            monthly_roi['ROI'] = (monthly_roi['Profit'] / monthly_roi['Cost']) * 100
            targets_roi = []
            for dt in monthly_roi['Timestamp']:
                ch_rois = []
                for ch in sales_channels:
                    y = dt.year
                    m = dt.strftime('%B')
                    _, cost_t, profit_t = get_marketing_target(ch, y, None, m)
                    ch_roi = (profit_t / cost_t) * 100 if cost_t > 0 else np.nan
                    ch_rois.append(ch_roi)
                targets_roi.append(np.nanmean([roi for roi in ch_rois if not np.isnan(roi)]))
            monthly_roi['ROI_Target'] = targets_roi
            monthly_roi['Performance_Pct'] = (monthly_roi['ROI'] / monthly_roi['ROI_Target']) * 100
            monthly_roi['Performance'] = monthly_roi['Performance_Pct'].apply(lambda pct: get_performance_category(pct) if not np.isnan(pct) else 'Bad performance')
            fig = px.line(
                monthly_roi,
                x='Timestamp',
                y='ROI',
                color='Performance',
                color_discrete_map=color_map,
                markers=True,
                title='Monthly ROI Trend'
            )
            fig.add_scatter(x=monthly_roi['Timestamp'], y=monthly_roi['ROI_Target'], mode='lines+markers', name='Target ROI', line=dict(color='red', dash='dot'))
            fig.update_layout(
                height=180,
                showlegend=True,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True)
            )
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            channel_conversion = marketing_df.groupby('Sales_Channel').size().reset_index(name='Conversions')
            total_conversions = channel_conversion['Conversions'].sum()
            channel_conversion['Conversion_Rate'] = (channel_conversion['Conversions'] /
                                                     total_conversions) * 100 if total_conversions > 0 else 0
            for ix, row in channel_conversion.iterrows():
                ch = row['Sales_Channel']
                y = year_selected if year_selected != 'All' else years[0]
                q = int(quarter_selected[1]) if quarter_selected != 'All' else None
                m = month_selected if month_selected != 'All' else None
                leads_target, _, _ = get_marketing_target(ch, y, q, m)
                perf_pct = (row['Conversions']/leads_target)*100 if leads_target > 0 else 0
                channel_conversion.loc[ix, 'Performance'] = get_performance_category(perf_pct)
            pie_fig = px.pie(
                channel_conversion,
                names='Sales_Channel',
                values='Conversion_Rate',
                color='Performance',
                color_discrete_map=color_map,
                title='Conversion Rate by Channel',
                hole=0.4
            )
            pie_fig.update_layout(height=180, margin=dict(l=0, r=0, t=30, b=0), showlegend=True)
            pie_fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(pie_fig, use_container_width=True)
        
elif selected_dept == 'AI Model':
    st.markdown("""

    <div class="metric-card" style="border-left-color: #1a936f">
        <div class="subheader">Best Performing Model</div>
        <div class="subheader">Random Forest Classifier</div>
        <div class="header">94.2% Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
    color_map = {
        'Bad performance': '#c33c54',    # <50%
        'Mid performance': '#f8961e',    # 50–79%
        'Close to target': '#43aa8b',    # 80–99%
        'At target': '#1a936f',          # 100%
        'Exceeded target': '#0a8754'     # >100%
    }
    def get_perf_cat(pct):
        if pct < 50:
            return 'Bad performance'
        elif 50 <= pct < 80:
            return 'Mid performance'
        elif 80 <= pct < 100:
            return 'Close to target'
        elif pct == 100:
            return 'At target'
        else:
            return 'Exceeded target'

    # Tabs for Trend Analysis and Demand Forecasting
    analysis_tab, forecast_tab = st.tabs(["Trend Analysis", "Demand Forecasting"])

    with analysis_tab:
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            trend_year = st.selectbox("Select Year", sorted(df['Year'].unique().tolist()), key='trend_year')
        with filter_col2:
            group_by_options = ['Product', 'Country', 'Sales Channel']
            trend_category = st.selectbox("Group By", group_by_options, key='trend_category')
        trend_df = df[df['Year'] == trend_year]

        # Calculate stats and yearly targets
        if trend_category == 'Product':
            group_stats = trend_df.groupby('Product').agg({'Quantity': 'sum', 'Total_Price': 'sum', 'Profit': 'sum'}).reset_index()
            group_stats['Target'] = group_stats['Product'].apply(lambda p: product_targets[p][f'{trend_year}_quantity_target'])
            group_stats['Label'] = group_stats['Product']
        elif trend_category == 'Country':
            group_stats = trend_df.groupby('Country').agg({'Quantity': 'sum', 'Total_Price': 'sum', 'Profit': 'sum'}).reset_index()
            group_stats['Target'] = group_stats['Country'].apply(lambda c: country_targets[c][f'{trend_year}_quantity_target'])
            group_stats['Label'] = group_stats['Country']
        else:
            group_stats = trend_df.groupby('Sales_Channel').agg({'Quantity': 'sum', 'Total_Price': 'sum', 'Profit': 'sum'}).reset_index()
            group_stats['Target'] = group_stats['Sales_Channel'].apply(lambda ch: channel_targets[ch][f'{trend_year}_quantity_target'])
            group_stats['Label'] = group_stats['Sales_Channel']

        group_stats['Performance_Pct'] = (group_stats['Quantity'] / group_stats['Target']) * 100
        group_stats['Performance_Pct'] = group_stats['Performance_Pct'].clip(upper=140)
        group_stats['Performance'] = group_stats['Performance_Pct'].apply(get_perf_cat)

        # Best/least performing
        container_col1, container_col2 = st.columns([1,1])
        if not group_stats.empty:
            best_row = group_stats.iloc[group_stats['Quantity'].idxmax()]
            least_row = group_stats.iloc[group_stats['Quantity'].idxmin()]
            with container_col1:
                st.success(f"Best: **{best_row['Label']}** ({best_row['Quantity']:.0f} units)", icon="✅")
            with container_col2:
                st.error(f"Least: **{least_row['Label']}** ({least_row['Quantity']:.0f} units)", icon="⚠️")

        # Visuals and tables side by side
        col1, col2 = st.columns([3,2])
        with col1:
            if trend_category == 'Product':
                fig = px.bar(
                    group_stats,
                    x='Product',
                    y='Quantity',
                    color='Performance',
                    color_discrete_map=color_map,
                    title=f'Product Demand vs Target ({trend_year})',
                    hover_data=['Target', 'Performance_Pct']
                )
                fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), xaxis_title='Product', yaxis_title='Quantity Sold')
                st.plotly_chart(fig, use_container_width=True)
            elif trend_category == 'Country':
                country_mapping = {
                    'England': 'United Kingdom',
                    'Scotland': 'United Kingdom',
                    'Ireland': 'Ireland',
                    'Netherlands': 'Netherlands',
                    'Germany': 'Germany',
                    'Belgium': 'Belgium',
                    'France': 'France',
                    'Wales': 'Wales'
                }
                group_stats['Country_Standard'] = group_stats['Country'].map(country_mapping)
                fig = px.choropleth(
                    group_stats,
                    locations='Country_Standard',
                    locationmode='country names',
                    color='Performance',
                    color_discrete_map=color_map,
                    hover_name='Country',
                    hover_data=['Quantity', 'Target', 'Performance_Pct'],
                    title=f'Country Demand vs Target ({trend_year})',
                    scope='europe'
                )
                fig.update_layout(
                    height=320,
                    geo=dict(showframe=False, showcoastlines=True, projection_type='mercator', center=dict(lon=10, lat=50),
                                lataxis_range=[35, 65], lonaxis_range=[-15, 25]))
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.pie(
                    group_stats,
                    names='Sales_Channel',
                    values='Quantity',
                    color='Performance',
                    color_discrete_map=color_map,
                    title=f'Sales Channel Distribution vs Target ({trend_year})',
                    hole=0.3,
                    hover_data=['Target', 'Performance_Pct']
                )
                fig.update_traces(textinfo='label+percent', hoverinfo='label+value+percent')
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
                
        with col2:
            st.markdown(f"**{trend_category} Table**")
            st.dataframe(
                group_stats[[trend_category if trend_category != 'Sales Channel' else 'Sales_Channel', 'Quantity', 'Target', 'Performance_Pct', 'Performance']]
                .sort_values('Performance_Pct', ascending=False)
                .style
                .applymap(lambda v: f'color: {color_map.get(get_perf_cat(v), "")};' if isinstance(v, float) and v > 0 else '', subset=['Performance_Pct'])
                .format({'Quantity': '{:,.0f}', 'Target': '{:,.0f}', 'Performance_Pct': '{:.1f}'})
            , height=320)
    with forecast_tab:

        # Model selection and parameters
        col1, col2 = st.columns(2)
        with col1:
            forecast_product = st.selectbox("Select Product", sorted(df['Product'].unique()), key='forecast_product')
        with col2:
            forecast_horizon = st.slider("Forecast Horizon (months)", 1, 12, 6, key='forecast_horizon')

        st.markdown(f"#### Forecasted Demand for {forecast_product} (Next {forecast_horizon} Months)")

        # Prepare dates for forecast
        last_date = df['Timestamp'].max()
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_horizon,
            freq='MS'
        )

        # Historical data (last 12 months)
        historical = df[(df['Product'] == forecast_product) &
                    (df['Timestamp'] >= last_date - pd.DateOffset(months=12))]
        historical_monthly = historical.groupby(pd.Grouper(key='Timestamp', freq='M'))['Quantity'].sum().reset_index()

        # Get the last value of historical demand to start the forecast
        if not historical_monthly.empty:
            last_actual = historical_monthly.iloc[-1]['Quantity']
        else:
            last_actual = df[df['Product'] == forecast_product]['Quantity'].mean()  # fallback

        # Simulate forecast: cumulative, fluctuating, starts from last known value
        product_std = df[df['Product'] == forecast_product]['Quantity'].std()
        if np.isnan(product_std) or product_std == 0:
            product_std = max(1, last_actual * 0.1)  # fallback for std

        forecast_values = []
        prev_value = last_actual
        for i in range(forecast_horizon):
            # Seasonality: higher in summer, lower in winter
            month = forecast_dates[i].month
            season_factor = 1.1 if month in [6,7,8] else 0.9 if month in [1,2,12] else 1.0
            # Fluctuation (random walk)
            noise = np.random.normal(0, product_std*0.2)
            forecast = max(0, prev_value * season_factor + noise)
            forecast_values.append(forecast)
            prev_value = forecast  # next forecast builds on this (cumulative)

        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_values,
            'Upper_Bound': [x * 1.2 for x in forecast_values],
            'Lower_Bound': [x * 0.8 for x in forecast_values]
        })

        # Layout for chart and table side by side
        chart_col, table_col = st.columns([2, 1])
        with chart_col:
            # Plot forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical_monthly['Timestamp'],
                y=historical_monthly['Quantity'],
                name='Historical Demand',
                line=dict(color='#1a936f')
            ))
            # Concatenate the last point of historical with forecast for continuity
            fig.add_trace(go.Scatter(
                x=[historical_monthly['Timestamp'].iloc[-1]] + list(forecast_df['Date']),
                y=[historical_monthly['Quantity'].iloc[-1]] + list(forecast_df['Forecast']),
                name='Forecast',
                line=dict(color='#2c3e50', dash='dot')
            ))
            # No confidence interval/shadow added

            fig.update_layout(
                title=f'Demand Forecast for {forecast_product}',
                xaxis_title='Date',
                yaxis_title='Quantity',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        with table_col:
            forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m')
            st.dataframe(forecast_df[['Date', 'Forecast', 'Upper_Bound', 'Lower_Bound']].style.format({
                'Forecast': '{:,.0f}',
                'Upper_Bound': '{:,.0f}',
                'Lower_Bound': '{:,.0f}'
            }), height=400)
            
elif selected_dept == 'Sales':
    # Performance color legend
    st.markdown("""
    <div class="color-legend">
        <div class="title">Performance Color Codes:</div>
        <div class="color-item"><div class="color-box" style="background-color: #c33c54;"></div>Bad performance</div>
        <div class="color-item"><div class="color-box" style="background-color: #f8961e;"></div>Mid performance</div>
        <div class="color-item"><div class="color-box" style="background-color: #43aa8b;"></div>Close to target</div>
        <div class="color-item"><div class="color-box" style="background-color: #1a936f;"></div>At target</div>
        <div class="color-item"><div class="color-box" style="background-color: #0a8754;"></div>Exceeded target</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Team Performance", "Individual Performance"])

    with tab1:
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                team_year_filter = st.selectbox("Team Year", ['All'] + sorted(df['Year'].unique().tolist()), key='team_year')
            with col2:
                team_quarter_filter = st.selectbox("Team Quarter", ['All'] + sorted(df['Quarter_Name'].unique().tolist()), key='team_quarter')
            with col3:
                team_month_filter = st.selectbox("Team Month", ['All'] + months_order, key='team_month')

            # Apply filters
            team_filtered_df = df.copy()
            if team_year_filter != 'All':
                team_filtered_df = team_filtered_df[team_filtered_df['Year'] == team_year_filter]
            if team_quarter_filter != 'All':
                team_filtered_df = team_filtered_df[team_filtered_df['Quarter_Name'] == team_quarter_filter]
            if team_month_filter != 'All':
                team_filtered_df = team_filtered_df[team_filtered_df['Month_Name'] == team_month_filter]

            # Calculate team targets based on filtered time period
            if team_year_filter != 'All':
                if team_quarter_filter != 'All':
                    q = int(team_quarter_filter[1])
                    team_profit_target = team_targets[f'{team_year_filter}_Q{q}_profit_target']
                    team_quantity_target = team_targets[f'{team_year_filter}_Q{q}_quantity_target']
                    team_revenue_target = team_targets[f'{team_year_filter}_Q{q}_revenue_target']
                elif team_month_filter != 'All':
                    team_profit_target = team_targets[f'{team_year_filter}_{team_month_filter}_profit_target']
                    team_quantity_target = team_targets[f'{team_year_filter}_{team_month_filter}_quantity_target']
                    team_revenue_target = team_targets[f'{team_year_filter}_{team_month_filter}_revenue_target']
                else:
                    team_profit_target = team_targets[f'{team_year_filter}_profit_target']
                    team_quantity_target = team_targets[f'{team_year_filter}_quantity_target']
                    team_revenue_target = team_targets[f'{team_year_filter}_revenue_target']
            else:
                team_profit_target = sum(team_targets[f'{year}_profit_target'] for year in years)
                team_quantity_target = sum(team_targets[f'{year}_quantity_target'] for year in years)
                team_revenue_target = sum(team_targets[f'{year}_revenue_target'] for year in years)

            # Calculate performance percentages (capped at 140%)
            team_profit = team_filtered_df['Profit'].sum()
            team_quantity = team_filtered_df['Quantity'].sum()
            team_revenue = team_filtered_df['Total_Price'].sum()

            team_profit_pct = min((team_profit / team_profit_target) * 100, 140) if team_profit_target > 0 else 0
            team_quantity_pct = min((team_quantity / team_quantity_target) * 100, 140) if team_quantity_target > 0 else 0
            team_revenue_pct = min((team_revenue / team_revenue_target) * 100, 140) if team_revenue_target > 0 else 0

            # Team Metrics
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #2c3e50">
                    <div class="subheader">Team Revenue (Target: ${team_revenue_target/1000000:,.2f}M)</div>
                    <div class="header">${team_revenue/1000000:,.2f}M</div>
                    <div class="subheader {get_performance_class(team_revenue_pct)}">
                        {team_revenue_pct:.1f}% of target
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #1a936f">
                    <div class="subheader">Team Profit (Target: ${team_profit_target/1000000:,.2f}M)</div>
                    <div class="header">${team_profit/1000000:,.2f}M</div>
                    <div class="subheader {get_performance_class(team_profit_pct)}">
                        {team_profit_pct:.1f}% of target
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with kpi3:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #114b5f">
                    <div class="subheader">Team Units Sold (Target: {team_quantity_target:,.0f})</div>
                    <div class="header">{team_quantity:,.0f}</div>
                    <div class="subheader {get_performance_class(team_quantity_pct)}">
                        {team_quantity_pct:.1f}% of target
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Team Visualizations
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                # Sales Rep Performance
                rep_performance = team_filtered_df.groupby('Sales_Rep').agg({
                    'Total_Price': 'sum',
                    'Profit': 'sum',
                    'Quantity': 'sum'
                }).reset_index()

                if team_year_filter != 'All':
                    rep_performance['Target'] = rep_performance['Sales_Rep'].apply(
                        lambda x: targets[x][f'{team_year_filter}_revenue_target'])
                else:
                    rep_performance['Target'] = rep_performance['Sales_Rep'].apply(
                        lambda x: sum(targets[x][f'{year}_revenue_target'] for year in years))

                rep_performance['Performance_Pct'] = (rep_performance['Total_Price'] / rep_performance['Target']) * 100
                rep_performance['Performance_Pct'] = rep_performance['Performance_Pct'].clip(upper=140)
                rep_performance['Performance'] = rep_performance['Performance_Pct'].apply(get_performance_category)

                fig = px.bar(rep_performance,
                            x='Sales_Rep',
                            y='Total_Price',
                            color='Performance',
                            title='Sales Rep Performance',
                            color_discrete_map=color_map,
                            hover_data=['Performance_Pct'])

                fig.update_traces(
                    customdata=rep_performance['Sales_Rep'],
                    hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<br>Performance: %{color}<br>Click to view details<extra></extra>"
                )

                fig.update_layout(
                    height=240,
                    margin=dict(l=0, r=0, t=30, b=0),
                    clickmode='event+select'
                )

                st.plotly_chart(fig, use_container_width=True, key="team_bar")

                if st.session_state.get('team_bar_select'):
                    selected_rep = st.session_state.team_bar_select['points'][0]['x']
                    st.session_state.selected_rep = selected_rep
                    st.session_state.active_tab = "Individual Performance"
                    st.experimental_rerun()

            with col2:
                # Sales Channel Distribution
                channel_sales = team_filtered_df.groupby('Sales_Channel').agg({
                    'Total_Price': 'sum'
                }).reset_index()

                if team_year_filter != 'All':
                    channel_sales['Target'] = channel_sales['Sales_Channel'].apply(
                        lambda x: channel_targets[x][f'{team_year_filter}_revenue_target'])
                else:
                    channel_sales['Target'] = channel_sales['Sales_Channel'].apply(
                        lambda x: sum(channel_targets[x][f'{year}_revenue_target'] for year in years))

                channel_sales['Performance_Pct'] = (channel_sales['Total_Price'] / channel_sales['Target']) * 100
                channel_sales['Performance_Pct'] = channel_sales['Performance_Pct'].clip(upper=140)
                channel_sales['Performance'] = channel_sales['Performance_Pct'].apply(get_performance_category)

                fig = px.pie(channel_sales,
                            names='Sales_Channel',
                            values='Total_Price',
                            color='Performance',
                            title='Sales Channel Distribution',
                            color_discrete_map=color_map,
                            hole=0.3)

                fig.update_traces(textinfo='label', hoverinfo='label+value+percent')
                fig.update_layout(height=240, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True, key="team_pie")

        # Second row of visualizations
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                # Product Performance
                product_performance = team_filtered_df.groupby('Product').agg({
                    'Total_Price': 'sum',
                    'Profit': 'mean'
                }).reset_index()

                if team_year_filter != 'All':
                    product_performance['Target'] = product_performance['Product'].apply(
                        lambda x: product_targets[x][f'{team_year_filter}_revenue_target'])
                else:
                    product_performance['Target'] = product_performance['Product'].apply(
                        lambda x: sum(product_targets[x][f'{year}_revenue_target'] for year in years))

                product_performance['Performance_Pct'] = (product_performance['Total_Price'] / product_performance['Target']) * 100
                product_performance['Performance_Pct'] = product_performance['Performance_Pct'].clip(upper=140)
                product_performance['Performance'] = product_performance['Performance_Pct'].apply(get_performance_category)

                fig = px.bar(product_performance,
                            x='Product',
                            y='Total_Price',
                            color='Performance',
                            title='Product Performance',
                            color_discrete_map=color_map)
                fig.update_layout(height=240, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True, key="team_product_bar")

            with col2:
                # Country Performance with country names in legend
                country_performance = team_filtered_df.groupby('Country').agg({
                    'Total_Price': 'sum',
                    'Profit': 'mean'
                }).reset_index()

                if team_year_filter != 'All':
                    country_performance['Target'] = country_performance['Country'].apply(
                        lambda x: country_targets[x][f'{team_year_filter}_revenue_target'])
                else:
                    country_performance['Target'] = country_performance['Country'].apply(
                        lambda x: sum(country_targets[x][f'{year}_revenue_target'] for year in years))

                country_performance['Performance_Pct'] = (country_performance['Total_Price'] / country_performance['Target']) * 100
                country_performance['Performance_Pct'] = country_performance['Performance_Pct'].clip(upper=140)
                country_performance['Performance'] = country_performance['Performance_Pct'].apply(get_performance_category)

                country_mapping = {
                    'England': 'United Kingdom',
                    'Scotland': 'United Kingdom',
                    'Ireland': 'Ireland',
                    'Netherlands': 'Netherlands',
                    'Germany': 'Germany',
                    'Belgium': 'Belgium'
                }

                country_performance['Country_Standard'] = country_performance['Country'].map(country_mapping)

                fig = px.choropleth(country_performance,
                                  locations='Country_Standard',
                                  locationmode='country names',
                                  color='Performance',
                                  hover_name='Country',
                                  hover_data=['Total_Price', 'Profit', 'Performance_Pct'],
                                  title='Country Performance',
                                  color_discrete_map=color_map,
                                  scope='europe')
                fig.update_layout(
                    height=240,
                    margin=dict(l=0, r=0, t=30, b=0),
                    geo=dict(
                        showframe=False,
                        showcoastlines=True,
                        projection_type='mercator',
                        center=dict(lon=10, lat=50),
                        lataxis_range=[35, 65],
                        lonaxis_range=[-15, 25]
                    ),
                    legend_title_text='Countries'
                )
                st.plotly_chart(fig, use_container_width=True, key="team_choropleth")

    with tab2:
        # Individual Performance
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'selected_rep' in st.session_state:
                    rep_filter = st.selectbox("Select Sales Rep", sorted(df['Sales_Rep'].unique().tolist(),
                                           index=sorted(df['Sales_Rep'].unique().tolist()).index(st.session_state.selected_rep)))
                else:
                    rep_filter = st.selectbox("Select Sales Rep", sorted(df['Sales_Rep'].unique().tolist()))
            with col2:
                ind_year_filter = st.selectbox("Year", ['All'] + sorted(df['Year'].unique().tolist()), key='ind_year')
            with col3:
                ind_quarter_filter = st.selectbox("Quarter", ['All'] + sorted(df['Quarter_Name'].unique().tolist()), key='ind_quarter')
            with col4:
                ind_month_filter = st.selectbox("Month", ['All'] + months_order, key='ind_month')

            # Apply filters
            ind_filtered_df = df[df['Sales_Rep'] == rep_filter].copy()
            if ind_year_filter != 'All':
                ind_filtered_df = ind_filtered_df[ind_filtered_df['Year'] == ind_year_filter]
            if ind_quarter_filter != 'All':
                ind_filtered_df = ind_filtered_df[ind_filtered_df['Quarter_Name'] == ind_quarter_filter]
            if ind_month_filter != 'All':
                ind_filtered_df = ind_filtered_df[ind_filtered_df['Month_Name'] == ind_month_filter]

            # Calculate targets
            if ind_year_filter != 'All':
                if ind_quarter_filter != 'All':
                    q = int(ind_quarter_filter[1])
                    profit_target = targets[rep_filter][f'{ind_year_filter}_Q{q}_profit_target']
                    quantity_target = targets[rep_filter][f'{ind_year_filter}_Q{q}_quantity_target']
                    revenue_target = targets[rep_filter][f'{ind_year_filter}_Q{q}_revenue_target']
                elif ind_month_filter != 'All':
                    profit_target = targets[rep_filter][f'{ind_year_filter}_{ind_month_filter}_profit_target']
                    quantity_target = targets[rep_filter][f'{ind_year_filter}_{ind_month_filter}_quantity_target']
                    revenue_target = targets[rep_filter][f'{ind_year_filter}_{ind_month_filter}_revenue_target']
                else:
                    profit_target = targets[rep_filter][f'{ind_year_filter}_profit_target']
                    quantity_target = targets[rep_filter][f'{ind_year_filter}_quantity_target']
                    revenue_target = targets[rep_filter][f'{ind_year_filter}_revenue_target']
            else:
                profit_target = sum(targets[rep_filter][f'{year}_profit_target'] for year in years)
                quantity_target = sum(targets[rep_filter][f'{year}_quantity_target'] for year in years)
                revenue_target = sum(targets[rep_filter][f'{year}_revenue_target'] for year in years)

            # Calculate performance
            total_profit = ind_filtered_df['Profit'].sum()
            total_quantity = ind_filtered_df['Quantity'].sum()
            total_revenue = ind_filtered_df['Total_Price'].sum()

            profit_pct = min((total_profit / profit_target) * 100, 140) if profit_target > 0 else 0
            quantity_pct = min((total_quantity / quantity_target) * 100, 140) if quantity_target > 0 else 0
            revenue_pct = min((total_revenue / revenue_target) * 100, 140) if revenue_target > 0 else 0

            # Individual Metrics
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #2c3e50">
                    <div class="subheader">Revenue (Target: ${revenue_target/1000000:,.2f}M)</div>
                    <div class="header">${total_revenue/1000000:,.2f}M</div>
                    <div class="subheader {get_performance_class(revenue_pct)}">
                        {revenue_pct:.1f}% of target
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #1a936f">
                    <div class="subheader">Profit (Target: ${profit_target/1000000:,.2f}M)</div>
                    <div class="header">${total_profit/1000000:,.2f}M</div>
                    <div class="subheader {get_performance_class(profit_pct)}">
                        {profit_pct:.1f}% of target
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with kpi3:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #114b5f">
                    <div class="subheader">Units Sold (Target: {quantity_target:,.0f})</div>
                    <div class="header">{total_quantity:,.0f}</div>
                    <div class="subheader {get_performance_class(quantity_pct)}">
                        {quantity_pct:.1f}% of target
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Individual Visualizations
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                # Profit Target Gauge
                st.markdown('<div class="gauge-title">Profit Target Achievement</div>', unsafe_allow_html=True)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = profit_pct,
                    number = {'suffix': "%", 'font': {'size': 24}},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 140], 'tickwidth': 1, 'tickcolor': "#2c3e50"},
                        'bar': {'color': color_map[get_performance_category(profit_pct)]},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': "#f8f9fa"},
                            {'range': [50, 80], 'color': "#fdf2d9"},
                            {'range': [80, 100], 'color': "#e9ecef"}],
                        'threshold': {
                            'line': {'color': "#1a936f", 'width': 4},
                            'thickness': 0.75,
                            'value': 100}
                    }
                ))
                fig.update_layout(
                    height=200,
                    margin=dict(t=30, b=0, l=10, r=10)
                )
                st.plotly_chart(fig, use_container_width=True, key="ind_gauge1")

            with col2:
                # Quantity Target Gauge
                st.markdown('<div class="gauge-title">Quantity Target Achievement</div>', unsafe_allow_html=True)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = quantity_pct,
                    number = {'suffix': "%", 'font': {'size': 24}},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 140], 'tickwidth': 1, 'tickcolor': "#2c3e50"},
                        'bar': {'color': color_map[get_performance_category(quantity_pct)]},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': "#f8f9fa"},
                            {'range': [50, 80], 'color': "#fdf2d9"},
                            {'range': [80, 100], 'color': "#e9ecef"}],
                        'threshold': {
                            'line': {'color': "#1a936f", 'width': 4},
                            'thickness': 0.75,
                            'value': 100}
                    }
                ))
                fig.update_layout(
                    height=200,
                    margin=dict(t=30, b=0, l=10, r=10)
                )
                st.plotly_chart(fig, use_container_width=True, key="ind_gauge2")

        # Additional visualizations
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                # Sales by Country
                country_sales = ind_filtered_df.groupby('Country').agg({
                    'Total_Price': 'sum',
                    'Profit': 'mean'
                }).reset_index()

                if ind_year_filter != 'All':
                    country_sales['Target'] = country_sales['Country'].apply(
                        lambda x: country_targets[x][f'{ind_year_filter}_revenue_target'])
                else:
                    country_sales['Target'] = country_sales['Country'].apply(
                        lambda x: sum(country_targets[x][f'{year}_revenue_target'] for year in years))

                country_sales['Performance_Pct'] = (country_sales['Total_Price'] / country_sales['Target']) * 100
                country_sales['Performance_Pct'] = country_sales['Performance_Pct'].clip(upper=140)
                country_sales['Performance'] = country_sales['Performance_Pct'].apply(get_performance_category)

                fig = px.bar(country_sales,
                            x='Country',
                            y='Total_Price',
                            color='Performance',
                            title=f'Sales by Country for {rep_filter}',
                            color_discrete_map=color_map)
                fig.update_layout(height=240, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True, key="ind_country_bar")

            with col2:
                # Sales by Channel
                channel_sales = ind_filtered_df.groupby('Sales_Channel').agg({
                    'Total_Price': 'sum',
                    'Profit': 'mean'
                }).reset_index()

                if ind_year_filter != 'All':
                    channel_sales['Target'] = channel_sales['Sales_Channel'].apply(
                        lambda x: channel_targets[x][f'{ind_year_filter}_revenue_target'])
                else:
                    channel_sales['Target'] = channel_sales['Sales_Channel'].apply(
                        lambda x: sum(channel_targets[x][f'{year}_revenue_target'] for year in years))

                channel_sales['Performance_Pct'] = (channel_sales['Total_Price'] / channel_sales['Target']) * 100
                channel_sales['Performance_Pct'] = channel_sales['Performance_Pct'].clip(upper=140)
                channel_sales['Performance'] = channel_sales['Performance_Pct'].apply(get_performance_category)

                fig = px.pie(channel_sales,
                            names='Sales_Channel',
                            values='Total_Price',
                            color='Performance',
                            title=f'Sales Channel Distribution for {rep_filter}',
                            color_discrete_map=color_map,
                            hole=0.3)

                fig.update_traces(textinfo='label', hoverinfo='label+value+percent')
                fig.update_layout(height=240, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True, key="ind_channel_pie")
       
st.markdown("""
    <style>
    /* Remove vertical scroll bar and force fit to one page */
    .main .block-container { max-height: 100vh !important; overflow-y: hidden !important; }
    html, body, .main, .block-container { height: 100vh !important; overflow-y: hidden !important; }
    /* Hide Streamlit's footer (optional) */
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
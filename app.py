import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Utilities
# ─────────────────────────────────────────────────────────────────────────────
START_DATE = pd.Timestamp('2025-07-01')  # simulation starts July 2025
YEARS = 5
MONTHS = YEARS * 12
SCENARIO_DIR = Path('scenarios')
SCENARIO_DIR.mkdir(exist_ok=True)

@st.cache_data
def load_scenarios():
    scenarios = {}
    for f in SCENARIO_DIR.glob('*.json'):
        with open(f, 'r') as fp:
            scenarios[f.stem] = json.load(fp)
    return scenarios

def save_scenario(name, data):
    path = SCENARIO_DIR / f"{name}.json"
    with open(path, 'w') as fp:
        json.dump(data, fp, default=str)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Scenario Management
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title('Scenario')
scenarios = load_scenarios()
choice = st.sidebar.selectbox('Load existing scenario', ['(new)'] + list(scenarios.keys()))
scenario_data = scenarios[choice] if choice != '(new)' else None
new_name = st.sidebar.text_input('Save scenario as', '')
save_btn = st.sidebar.button('Save scenario')

# Placeholder for inputs container so it exists for saving after declaration
current_inputs = {}

# ─────────────────────────────────────────────────────────────────────────────
# Main App: Inputs
# ─────────────────────────────────────────────────────────────────────────────
st.title('Runway & Financial Simulator for Business Neobank')

# 1) Base parameters
st.header('Base Parameters')
col1, col2 = st.columns(2)
with col1:
    cash_start = st.number_input('Cash in bank (USD)', value=70_000_000, step=1_000_000)
    cos_pct = st.slider('Cost of Sales (% of Revenue)', 0.0, 1.0, 0.4)
    hq_gna_start = st.number_input('HQ G&A annual (USD)', value=15_000_000, step=500_000)
    hq_share = st.slider('HQ tech bandwidth share (%)', 0.0, 1.0, 0.5)
with col2:
    tech_units_start = st.number_input('Tech units available per month', value=50, step=1)
    tech_unit_cost_start = st.number_input('Tech unit cost (USD)', value=30_000, step=1_000)
    gna_growth_share = st.slider('HQ G&A growth share of revenue growth (%)', 0.0, 1.0, 0.5)
    tech_growth_share = st.slider('Tech capacity growth share of revenue growth (%)', 0.0, 1.0, 1.0)

# 2) Base Markets
st.header('Existing Markets (Base)')
base_markets_cols = [
    'Market', 'Existing Clients', 'CAC', 'ARPU', 'NRR',
    'Churn Y1', 'Churn Post Y1',
    'New Clients Y1','New Clients Y2','New Clients Y3','New Clients Y4','New Clients Y5',
    'CSC', 'Tech Units/mo', 'Ongoing G&A annual'
]
base_markets_df = pd.DataFrame(
    [
        ['Singapore', 20000, 500, 1500, 1.0, 0.3, 0.1, 6000,6000,6000,6000,6000, 200, 4, 2_000_000],
        ['Hong-Kong', 10000, 500, 1500, 1.0, 0.3, 0.05, 1000,3000,6000,10000,15000, 200, 4, 2_000_000]
    ], columns=base_markets_cols
)
# Fixed row count for base markets
base_markets_df = st.data_editor(
    base_markets_df,
    key='base_markets',
    num_rows='fixed'
)

# 3) New Markets
st.header('New Markets')
new_markets_cols = [
    'Market', 'Start Date', 'Prep Duration (mo)',
    'Prep Tech/mo', 'Prep G&A/mo',
    'CAC', 'ARPU', 'Churn Y1', 'Churn Post Y1',
    'New Clients Y1','New Clients Y2','New Clients Y3','New Clients Y4','New Clients Y5',
    'CSC',
    'Maint Tech Y1','Maint Tech Y2','Maint Tech Y3','Maint Tech Y4','Maint Tech Y5',
    'Ongoing G&A Y1','Ongoing G&A Y2','Ongoing G&A Y3','Ongoing G&A Y4','Ongoing G&A Y5'
]
new_markets_df = pd.DataFrame(columns=new_markets_cols)
new_markets_df.loc[0] = [
    'United States', '2025-08-01', 0,
    0, 0,
    1000, 2000, 0.3, 0.1,
    1000,3000,10000,25000,50000,
    300,
    3,4,5,6,7,
    1_000_000,2_000_000,3_000_000,4_000_000,5_000_000
]
# Dynamic row count allows adding/removing markets
new_markets_df = st.data_editor(
    new_markets_df,
    key='new_markets',
    num_rows='dynamic'
)

# 4) New Products
st.header('New Products')
new_products_cols = [
    'Product', 'Start Date', 'Prep Duration (mo)',
    'Prep Tech/mo','Prep G&A/mo',
    'CAC Mult','ARPU Mult','Churn Y1 Mult','Churn Post Y1 Mult','CSC Mult',
    'Adopt Y1','Adopt Y2','Adopt Y3','Adopt Y4','Adopt Y5',
    'Maint Tech Y1','Maint Tech Y2','Maint Tech Y3','Maint Tech Y4','Maint Tech Y5',
    'Ongoing G&A Y1','Ongoing G&A Y2','Ongoing G&A Y3','Ongoing G&A Y4','Ongoing G&A Y5'
]
new_products_df = pd.DataFrame(columns=new_products_cols)
new_products_df.loc[0] = [
    'AI Accounting', '2025-07-01', 6,
    2, 20000,
    0.95,1.3,0.9,0.9,1.2,
    0.02,0.05,0.10,0.20,0.30,
    3,3,3,3,3,
    250000,500000,750000,1000000,1500000
]
# Dynamic row count allows adding/removing products
new_products_df = st.data_editor(
    new_products_df,
    key='new_products',
    num_rows='dynamic'
)

# 5) Efficiency Projects
st.header('Efficiency Projects')
eff_cols = [
    'Project','Start Date','Duration (mo)',
    'Tech Units/mo','CAC Mult','CSC Mult','Tech Unit Cost Mult'
]
eff_df = pd.DataFrame(columns=eff_cols)
eff_df = st.data_editor(
    eff_df,
    key='efficiency',
    num_rows='dynamic'
)

# Gather inputs for saving or simulation
def gather_inputs():
    return {
        'base': base_markets_df.to_dict(orient='list'),
        'new_markets': new_markets_df.to_dict(orient='list'),
        'new_products': new_products_df.to_dict(orient='list'),
        'efficiency': eff_df.to_dict(orient='list'),
        'params': {
            'cash_start': cash_start,
            'cos_pct': cos_pct,
            'hq_gna_start': hq_gna_start,
            'hq_share': hq_share,
            'gna_growth_share': gna_growth_share,
            'tech_units_start': tech_units_start,
            'tech_unit_cost_start': tech_unit_cost_start,
            'tech_growth_share': tech_growth_share
        }
    }

current_inputs = gather_inputs()
if save_btn and new_name:
    save_scenario(new_name, current_inputs)
    st.sidebar.success(f'Scenario "{new_name}" saved')

# ─────────────────────────────────────────────────────────────────────────────
# Simulation Logic
# ─────────────────────────────────────────────────────────────────────────────
def simulate(inputs):
    # [simulation logic unchanged…]
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Run & Display
# ─────────────────────────────────────────────────────────────────────────────
if st.button('Run Simulation'):
    with st.spinner('Simulating…'):
        results = simulate(current_inputs)
    st.success('Simulation complete!')

    # [display logic unchanged...]

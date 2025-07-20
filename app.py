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
# App Header & Scenario Management
# ─────────────────────────────────────────────────────────────────────────────
st.title('Runway & Financial Simulator for Business Neobank')

st.sidebar.title('Scenario')
scenarios = load_scenarios()
choice = st.sidebar.selectbox('Load scenario', ['(new)'] + list(scenarios.keys()))
scenario_data = scenarios[choice] if choice != '(new)' else None
new_name = st.sidebar.text_input('Save scenario as')
if st.sidebar.button('Save') and new_name:
    save_scenario(new_name, inputs)
    st.sidebar.success(f"Saved '{new_name}'")

# ─────────────────────────────────────────────────────────────────────────────
# How It Works Explanation
# ─────────────────────────────────────────────────────────────────────────────
with st.expander('How it works', expanded=False):
    st.markdown(
        """
This simulator lets you define:

• Base parameters (cash, cost structure, tech bandwidth, G&A growth)
• Existing markets with churn, ARPU, CAC, new-client schedules, tech & G&A costs
• New markets, new products, and efficiency projects via dynamic tables

Click **Run Simulation** to project, month-by-month over five years:

- Revenue by market
- Costs (CoS, Acquisition, Servicing) by market plus aggregated G&A & Tech
- Tech capacity vs. requirement
- Cash balance
- Key metrics (CAC, ARPU, NRR, churn)

**Original Prompt:**
```
I am the CFO of a series C business neobank. I want to build a simulator of my company's financials to understand what 
my revenue, costs and runway will look like based on different scenarios. ...
```"""
    )

# ─────────────────────────────────────────────────────────────────────────────
# Input Tables
# ─────────────────────────────────────────────────────────────────────────────
# 1) Base Parameters
st.header('Base Parameters')
col1, col2 = st.columns(2)
with col1:
    cash_start = st.number_input('Cash in bank (USD)', 70000000)
    cos_pct = st.slider('Cost of Sales (% of Revenue)', 0.0, 1.0, 0.4)
    hq_gna = st.number_input('HQ G&A annual (USD)', 15000000)
    hq_share = st.slider('HQ tech bandwidth share (%)', 0.0, 1.0, 0.5)
with col2:
    tech_units = st.number_input('Tech units/mo', 50)
    tech_cost = st.number_input('Tech unit cost (USD)', 30000)
    gna_share = st.slider('HQ G&A growth share', 0.0, 1.0, 0.5)
    tech_share = st.slider('Tech capacity growth share', 0.0, 1.0, 1.0)

# 2) Existing Markets
st.header('Existing Markets')
base_cols = ['Market','Existing Clients','CAC','ARPU','NRR',
             'Churn Y1','Churn Post Y1',
             'New Y1','New Y2','New Y3','New Y4','New Y5',
             'CSC','Tech/mo','G&A annual']
base_df = pd.DataFrame([
    ['Singapore',20000,500,1500,1.0,0.3,0.1,6000,6000,6000,6000,6000,200,4,2000000],
    ['Hong-Kong',10000,500,1500,1.0,0.3,0.05,1000,3000,6000,10000,15000,200,4,2000000]
], columns=base_cols)
base_df = st.data_editor(base_df, key='base', num_rows='fixed')

# 3) New Markets
st.header('New Markets')
new_cols = ['Market','Start Date','Prep mo','Prep Tech/mo','Prep G&A/mo',
            'CAC','ARPU','Churn Y1','Churn Post Y1',
            'New Y1','New Y2','New Y3','New Y4','New Y5',
            'CSC','Maint1','Maint2','Maint3','Maint4','Maint5',
            'G&A1','G&A2','G&A3','G&A4','G&A5']
new_df = pd.DataFrame(columns=new_cols)
new_df.loc[0] = ['United States','2025-08-01',0,0,0,1000,2000,0.3,0.1,
                 1000,3000,10000,25000,50000,300,3,4,5,6,7,1000000,2000000,3000000,4000000,5000000]
new_df = st.data_editor(new_df, key='new', num_rows='dynamic')

# 4) New Products
st.header('New Products')
prod_cols = ['Product','Start Date','Prep mo','Prep Tech/mo','Prep G&A/mo',
             'CAC Mult','ARPU Mult','Churn1 Mult','ChurnP Mult','CSC Mult',
             'Ad1','Ad2','Ad3','Ad4','Ad5','M1','M2','M3','M4','M5',
             'G&A1','G&A2','G&A3','G&A4','G&A5']
prod_df = pd.DataFrame(columns=prod_cols)
prod_df.loc[0] = ['AI Accounting','2025-07-01',6,2,20000,
                  0.95,1.3,0.9,0.9,1.2,0.02,0.05,0.1,0.2,0.3,3,3,3,3,3,
                  250000,500000,750000,1000000,1500000]
prod_df = st.data_editor(prod_df, key='prod', num_rows='dynamic')

# 5) Efficiency Projects
st.header('Efficiency Projects')
eff_cols = ['Project','Start Date','Duration','Tech/mo','CAC Mult','CSC Mult','TechCost Mult']
eff_df = pd.DataFrame(columns=eff_cols)
eff_df = st.data_editor(eff_df, key='eff', num_rows='dynamic')

# Filters
st.sidebar.title('Filters')
mkt_list = list(base_df['Market']) + list(new_df['Market'])
sel_mkt = st.sidebar.multiselect('Markets', mkt_list, default=mkt_list)
prod_list = list(prod_df['Product'])
sel_prod = st.sidebar.multiselect('Products', prod_list, default=prod_list)
eff_list = list(eff_df['Project'])
sel_eff = st.sidebar.multiselect('Efficiency', eff_list, default=eff_list)

# Gather inputs
def gather_inputs():
    return {
        'base': base_df.to_dict('list'),
        'new': new_df.to_dict('list'),
        'prod': prod_df.to_dict('list'),
        'eff': eff_df.to_dict('list'),
        'params': {
            'cash': cash_start,
            'cos': cos_pct,
            'hq_gna': hq_gna,
            'hq_share': hq_share,
            'gna_share': gna_share,
            'tech_units': tech_units,
            'tech_cost': tech_cost,
            'tech_share': tech_share
        }
    }
inputs = gather_inputs()

# Simulation logic stub
# def simulate(inputs):
#     ...
#     return rev_mkt, costs_agg, tech_df, cash_series, metrics_df, cos_df, acq_df, serv_df

# ─────────────────────────────────────────────────────────────────────────────
# Run & Display
# ─────────────────────────────────────────────────────────────────────────────
if st.button('Run Simulation'):
    rev_mkt, costs_agg, tech_df, cash_series, metrics_df, cos_df, acq_df, serv_df = simulate(inputs)

    # Revenue by market filtered
    st.subheader('Revenue by Market')
    st.line_chart(rev_mkt[sel_mkt])

    # Costs by market
    st.subheader('Costs by Market')
    total_costs = cos_df + acq_df + serv_df
    st.area_chart(total_costs[sel_mkt])

    # Aggregated costs breakdown
    st.subheader('Aggregated Costs Breakdown')
    st.area_chart(costs_agg)

    # Cash balance
    st.subheader('Cash Balance')
    st.line_chart(cash_series)

    # Tech capacity
    st.subheader('Tech Capacity')
    st.line_chart(tech_df)

    # Key metrics
    st.subheader('Key Metrics')
    st.line_chart(metrics_df)

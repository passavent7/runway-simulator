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

# "How it works" button to show usage and original prompt
if st.button('How it works'):
    st.markdown('## How it works:')
    st.markdown('''
This simulator lets you define:

- **Base parameters** (cash, cost structure, tech bandwidth, G&A growth)
- **Existing markets**, with churn, ARPU, CAC, new-client schedules, tech & G&A costs
- **New markets**, **new products**, and **efficiency projects** via dynamic tables

Then hit **Run Simulation** to project, month-by-month over 5 years:

- Revenue (by market & product)
- Costs (CoS, G&A, Acquisition, Servicing, Tech)
- Tech capacity vs requirement
- Cash balance
- Key metrics (CAC, ARPU, NRR, churn, CSC)

**Original specification used to generate this app:**
```
I am the CFO of a series C business neobank. I want to build a simulator of my company's financials to understand what my revenue, costs and runway will look like based on different scenarios. 
Simulator to be hosted on a webpage with streamlit, coded in Python.
 
We are contemplating a few projects to grow faster. Typically 3 kinds of projects:
- Launching our core product (a "Business Account" for SMEs) in new markets. To be able to serve more SMEs.
- Launching new Products (in all markets where we operate). Always focused on SMEs -> Goal is to lower our CAC, grow ARPU and increase NRR across markets where we'll operate.
- Efficiency increase projects - to lower our CAC and CSC (Customer Servicing Costs). 

Each of these 3 types of projects has different types of parameters.
For new markets: 
- Market name
- Project start date
- Time needed to prepare the launch (in months)
- Tech Bandwidth units to prepare launch (per month)
- G&A Cost for launch preparation (per month)
- CAC ($)
- ARPU ($)
- Churn Year 1
- Churn post Year 1 
- New clients in Year 1, 2, 3, 4, 5 (linear progression each month):
- CSC ($ per client per year)
- Tech Bandwidth units for maintenance (per month) - Y1, Y2, Y3, Y4, Y5
- Ongoing G&A cost ($ per year) - Y1, Y2, Y3, Y4, Y5

For new Products:
- Product name
- Project start date
- Time needed to prepare the launch (in months)
- Tech Bandwidth units during launch preparation (per month)
- G&A cost for launch preparation (per month)
- CAC multiplier (as new products are expected to reduce our overall CAC - applies to all clients)
- ARPU multiplier (as new products are expected to increase our overall ARPU - applies only to clients using the product)
- Churn Year 1 multiplier (as new products are expected to reduce it - applies only to clients using the product)
- Churn post Year 1 multiplier (as new products are expected to reduce it - applies only to clients using the product)
- CSC multiplier (as new products are expected to increase our overall CSC - applies only to clients using the product)
- Adoption Y1, Y2, Y3, Y4, Y5 (percentage of existing client base, spread evenly across markets, linear progression each month)
- Tech Bandwidth units for maintenance (units per month) - Y1, Y2, Y3, Y4, Y5
- Ongoing G&A cost ($ per year) - Y1, Y2, Y3, Y4, Y5

For new Efficiency Increase Projects:
- Project name:
- Project start date:
- Duration (in months):
- Tech Bandwidth units required per month:
- CAC multiplier (as Efficiency increase projects reduce our overall CAC)
- CSC multiplier (as Efficiency projects reduce overall CSC; applies to our entire client base)
- Tech bandwidth unit multiplier (as Efficiency projects reduce the cost per Tech Bandwidth unit)

In the simulator, for each of the 3 types of project, I want to be able to input a list of projects in a table. Table columns should be the required parameters for those projects. Choose self explanatory names for columns of the table - but add explainers on each, to help clear any possible doubt.
After the simulation is run, it should be possible to edit the parameters of any project easily, and re-run the simulation.

I the output of the simulator, I want to see, on different graphs, the month on month evolution of the following for next 5 years:
- Revenue
- Costs 
- Total Tech bandwidth units available
- Total Tech bandwidth units required
- Cash in the bank 
- All key metrics: CAC, ARPU, NRR, CSC, Churn year 1, Churn post year 1 (especially as the launch of new products and efficiency projects impact them)

For Revenue and Cost, have views showing contribution of each Market and Product.
For Costs, have a view showing breakdown by type of cost (CoS, G&A, Tech, Acquisition, Servicing)
For all graphs, show aggregate over all markets - but the interface should allow to select only some markets. For costs breakdown, allocate 'Headquarters' costs to a standalone, non Revenue generating 'Headquarters' market 

Base data for simulations:

- Cash in the bank: $70 million
- Cost of Sales (we assume it is always the same for simplicity): 40% of Revenue
- Headquarter G&A costs: $15 million per year. 
- Headquarter G&A cost growth rate (as a percentage of Revenue growth rate, to calculate for each month): 50%
- Available Tech Bandwidth units (per month): 50 
- Tech bandwidth unit cost: $30,000  
- Tech bandwidth required by Headquarter (bandwidth required for Product infra maintenance and enhancements; expressed as a percentage of total available Tech Bandwidth units, to calculate for each month): 50% 
- Total available Tech bandwidth units growth rate (expressed as a percentage of total Revenue's growth rate, to calculate for each month):  100%

Key numbers in existing Markets: 
Singapore:
- Existing clients: 20000
- CAC: $500
- ARPU: $1500
- NRR: 100%
- Clients churn Year 1: 30% 
- Clients churn Post Year 1: 10%
- New clients in Year 1, 2, 3, 4, 5: 6000, 6000, 6000, 6000, 6000
- CSC: $200
- Tech Bandwidth units required per month: 4
- Ongoing G&A cost: $2M per year, every year

Hong-Kong:
- Existing clients: 1000
- CAC: $500
- ARPU: $1500
- Clients churn Year 1: 30% 
- Clients churn Post Year 1: 5%
- New clients in Year 1, 2, 3, 4, 5: 1000, 3000, 6000, 10000, 15000
- CSC: $200
- Tech Bandwidth units required per month: 4
- Ongoing G&A cost: $2M per year, every year

All those base numbers should also be editable in the simulator, via tables.

Also Prefill the new markets table with this market: 
- Market name: United States
- Project start date: August 2025
- Time needed to prepare the launch (in months): 0
- Tech Bandwidth units to prepare launch (per month): 0
- G&A Cost for launch preparation (per month): 0
- CAC ($): 1000
- ARPU ($): 2000
- Clients churn Year 1: 30% 
- Clients churn Post Year 1: 10%
- New clients in Year 1, 2, 3, 4, 5: 1000, 3000, 10000, 25000, 50000
- CSC ($ per client per year): 300
- Tech Bandwidth units for maintenance (per month) - Y1, Y2, Y3, Y4, Y5: 3, 4, 5, 6, 7
- Ongoing G&A cost ($ per year) - Y1, Y2, Y3, Y4, Y5: 1000000, 2000000, 3000000, 4000000, 5000000

Also prefill the new Products table with this Product: 
- Product name: AI Accounting
- Project start date: July 2025
- Time needed to prepare the launch (in months): 6
- Tech Bandwidth units during launch preparation (per month): 2
- G&A cost for launch preparation (per month): $20000
- CAC multiplier: 0.95
- ARPU multiplier: 1.3
- Churn Year 1 multiplier: 0.9
- Churn post Year 1 multiplier: 0.9
- CSC multiplier: 1.2
- Adoption Y1, Y2, Y3, Y4, Y5 (percentage of existing client base, spread evenly across markets, linear progression each month): 2, 5, 10, 20, 30
- Tech Bandwidth units for maintenance Y1, Y2, Y3, Y4, Y5: 3, 3, 3, 3, 3
- Ongoing G&A cost ($ per year) - Y1, Y2, Y3, Y4, Y5: 250000, 500000, 750000, 1000000, 1500000

For now no need to think about exports or imports in JSON/.csv, we do everything online
Start simulation from July 2025.
Calculate churn monthly from yearly numbers given in input.
Allow to save all scenarios / each time we run a simulation
Assume no other inflows/outflows aside from operating P&L and capex from projects.
Allow to go over Tech budget capacity, but pls indicate clearly at the top of the dashboard if we're above capacity

Do let me know first if anything is not clear - important to clarify everything before building!```''')

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
            'hq_gna

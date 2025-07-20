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
st.title('Runway & Financial Simulator')
scenarios = load_scenarios()
choice = st.sidebar.selectbox('Load scenario', ['(new)'] + list(scenarios.keys()))
scenario_data = scenarios[choice] if choice != '(new)' else None
new_name = st.sidebar.text_input('Save scenario as')
if st.sidebar.button('Save') and new_name:
    save_scenario(new_name, inputs)
    st.sidebar.success(f"Saved '{new_name}'")

# ─────────────────────────────────────────────────────────────────────────────
# How It Works
# ─────────────────────────────────────────────────────────────────────────────
with st.expander('How it works'):
    st.markdown(
        '''
This simulator was generated from the following specification:

I am the CFO of a series C business neobank. I want to build a simulator of my company's financials to understand what my revenue, costs and runway will look like based on different scenarios. Simulator to be hosted on a webpage with streamlit, coded in Python.

We are contemplating a few projects to grow faster: new markets, new products, efficiency projects.

Base data:
- Cash: $70M; CoS=40%; HQ G&A $15M/yr; G&A growth 50% of revenue growth; Tech units=50/mo; Tech cost=30k; Tech growth 100% of revenue growth; HQ tech share 50%.

Existing markets (editable):
Singapore & Hong‑Kong with clients, CAC, ARPU, churn, new clients Y1–Y5, CSC, tech/mo, G&A.

New Markets: Market, start, prep time, prep tech/G&A, CAC/ARPU/churn, new clients Y1–Y5, CSC, maint tech Y1–Y5, G&A Y1–Y5.

New Products: Product, start, prep time, prep tech/G&A, CAC/ARPU/churn/CSC multipliers, adoption Y1–Y5, maint tech Y1–Y5, G&A Y1–Y5.

Efficiency: Project, start, duration, tech/mo, CAC/CSC/tech‑cost multipliers.

Click Run Simulation to project monthly over 5 years:
- Revenue by market & filters
- Costs by market & filters
- Aggregated cost breakdown
- Cash balance
- Tech capacity
- Key metrics
'''    )

# ─────────────────────────────────────────────────────────────────────────────
# Input Tables
# ─────────────────────────────────────────────────────────────────────────────
# Base Parameters
st.header('Base Parameters')
col1, col2 = st.columns(2)
with col1:
    cash_start = st.number_input('Cash in bank (USD)', 70000000)
    cos_pct = st.slider('CoS % of Revenue', 0.0, 1.0, 0.4)
    hq_gna = st.number_input('HQ G&A annual (USD)', 15000000)
    hq_share = st.slider('HQ tech share', 0.0, 1.0, 0.5)
with col2:
    tech_units = st.number_input('Tech units/mo', 50)
    tech_cost = st.number_input('Tech unit cost (USD)', 30000)
    gna_share = st.slider('G&A growth share', 0.0, 1.0, 0.5)
    tech_share = st.slider('Tech growth share', 0.0, 1.0, 1.0)

# Existing Markets
st.header('Existing Markets')
base_cols = ['Market','Existing Clients','CAC','ARPU','NRR','Churn Y1','Churn Post','New1','New2','New3','New4','New5','CSC','Tech/mo','G&A/yr']
base_df = pd.DataFrame([
    ['Singapore',20000,500,1500,1.0,0.3,0.1,6000,6000,6000,6000,6000,200,4,2000000],
    ['Hong-Kong',10000,500,1500,1.0,0.3,0.05,1000,3000,6000,10000,15000,200,4,2000000]
], columns=base_cols)
base_df = st.data_editor(base_df, key='base', num_rows='fixed')

# New Markets & Projects
st.header('New Markets')
new_cols = ['Market','Start','Prep mo','Prep Tech/mo','Prep G&A/mo','CAC','ARPU','Churn Y1','Churn Post','New1','New2','New3','New4','New5','CSC','M1','M2','M3','M4','M5','G1','G2','G3','G4','G5']
def table_editor(key, cols, init=None):
    df = pd.DataFrame(columns=cols)
    if init: df.loc[0]=init
    return st.data_editor(df, key=key, num_rows='dynamic')
new_df = table_editor('new', new_cols, ['United States','2025-08-01',0,0,0,1000,2000,0.3,0.1,1000,3000,10000,25000,50000,300,3,4,5,6,7,1000000,2000000,3000000,4000000,5000000])

st.header('New Products')
prod_cols = ['Product','Start','Prep mo','Prep Tech/mo','Prep G&A/mo','CAC Mult','ARPU Mult','Churn1 Mult','ChurnP Mult','CSC Mult','Ad1','Ad2','Ad3','Ad4','Ad5','M1','M2','M3','M4','M5','G1','G2','G3','G4','G5']
prod_df = table_editor('prod', prod_cols, ['AI Accounting','2025-07-01',6,2,20000,0.95,1.3,0.9,0.9,1.2,0.02,0.05,0.1,0.2,0.3,3,3,3,3,3,250000,500000,750000,1000000,1500000])

st.header('Efficiency Projects')
eff_cols = ['Project','Start','Duration','Tech/mo','CAC Mult','CSC Mult','TechCost Mult']
eff_df = table_editor('eff', eff_cols)

# Sidebar Filters
st.sidebar.header('Filters')
markets = list(base_df['Market']) + list(new_df['Market'])
sel_mkt = st.sidebar.multiselect('Markets', markets, default=markets)
products = list(prod_df['Product'])
sel_prod = st.sidebar.multiselect('Products', products, default=products)
eff_list = list(eff_df['Project'])
sel_eff = st.sidebar.multiselect('Efficiency', eff_list, default=eff_list)

# Gather all inputs
def gather():
    return {'base':base_df.to_dict('list'),'new':new_df.to_dict('list'),'prod':prod_df.to_dict('list'),'eff':eff_df.to_dict('list'),'params':{'cash':cash_start,'cos':cos_pct,'hq_gna':hq_gna,'hq_share':hq_share,'gna_share':gna_share,'tech_units':tech_units,'tech_cost':tech_cost,'tech_share':tech_share}}
inputs = gather()

# Simulation function
def simulate(inp):
    base = pd.DataFrame(inp['base'])
    newm = pd.DataFrame(inp['new'])
    prod = pd.DataFrame(inp['prod'])
    effp = pd.DataFrame(inp['eff'])
    p = inp['params']
    dates = pd.date_range(START_DATE, periods=MONTHS, freq='MS')
    n, m0, m1 = MONTHS, len(base), len(newm)
    markets = list(base['Market']) + list(newm['Market'])
    # Monthly churn
    churn_y1 = np.concatenate([base['Churn Y1'].astype(float), newm['Churn Y1'].astype(float)])
    churn_post = np.concatenate([base['Churn Post'].astype(float), newm['Churn Post'].astype(float)])
    cy1 = 1 - (1 - churn_y1)**(1/12)
    cp  = 1 - (1 - churn_post)**(1/12)
    # Initialize cohorts
    cohorts = np.zeros((n, m0+m1, n))
    for i in range(m0): cohorts[0, i, 12] = base.at[i, 'Existing Clients']
    # New clients schedule
    newc = np.zeros((n, m0+m1))
    for i in range(m0):
        ys = base.loc[i, ['New1','New2','New3','New4','New5']].astype(float).values
        for yi in range(5): newc[yi*12:(yi+1)*12, i] = ys[yi]/12
    for i in range(m1):
        r = newm.loc[i]
        idx = (pd.to_datetime(r['Start']).year-START_DATE.year)*12 + pd.to_datetime(r['Start']).month-1
        prep = int(r['Prep mo']); ys = r[['New1','New2','New3','New4','New5']].astype(float).values
        for yi in range(5):
            s = idx+prep+yi*12; e = min(s+12, n)
            if s< n: newc[s:e, m0+i] = ys[yi]/12
    # Simulate cohorts
    for t in range(1, n):
        for i in range(m0+m1):
            prev = cohorts[t-1, i]; curr = np.zeros(n); curr[0] = newc[t, i]
            for age in range(1, n): rate = cy1[i] if age<12 else cp[i]; curr[age] = prev[age-1]*(1-rate)
            cohorts[t, i] = curr
    active = cohorts.sum(axis=2)
    # Product adoption
    adopt = np.zeros((n, len(prod)))
    for j in range(len(prod)):
        r = prod.loc[j]
        idx = (pd.to_datetime(r['Start']).year-START_DATE.year)*12 + pd.to_datetime(r['Start']).month-1
        prep = int(r['Prep mo']); ys = r[['Ad1','Ad2','Ad3','Ad4','Ad5']].astype(float).values
        arr = np.zeros(n)
        for yi in range(5):
            s = idx+prep+yi*12; e = min(s+12, n)
            if s<n: arr[s:e] = ys[yi]/12
        adopt[:,j] = np.cumsum(arr)
    # Efficiency multipliers
    eff_act = np.zeros((n,len(effp)))
    for j in range(len(effp)):
        r = effp.loc[j]; idx = (pd.to_datetime(r['Start']).year-START_DATE.year)*12+pd.to_datetime(r['Start']).month-1
        dur = int(r['Duration']); eff_act[idx:idx+dur, j] = 1
    cac_eff, csc_eff, tc_eff = np.ones(n), np.ones(n), np.ones(n)
    for j in range(len(effp)):
        mrow = effp.loc[j, ['CAC Mult','CSC Mult','TechCost Mult']].astype(float).values
        mask = eff_act[:, j]==1
        cac_eff[mask]*=mrow[0]; csc_eff[mask]*=mrow[1]; tc_eff[mask]*=mrow[2]
    # Prepare containers
    idxs = pd.DatetimeIndex(dates)
    rev_mkt = pd.DataFrame(0, index=idxs, columns=markets)
    cos_df  = pd.DataFrame(0, index=idxs, columns=markets)
    acq_df  = pd.DataFrame(0, index=idxs, columns=markets)
    serv_df = pd.DataFrame(0, index=idxs, columns=markets)
    tech_av, tech_req = np.zeros(n), np.zeros(n)
    gna_ser, cash_ser = np.zeros(n), np.zeros(n)
    # Initialize
    gna_m = p['hq_gna']/12; cash_ser[0]=p['cash']; tech_av[0]=p['tech_units']
    # Monthly loop
    for t in range(n):
        arpu = np.concatenate([base['ARPU'], newm['ARPU']]).astype(float)
        cac  = np.concatenate([base['CAC'], newm['CAC']]).astype(float)
        csc  = np.concatenate([base['CSC'], newm['CSC']]).astype(float)
        for j in range(len(prod)):
            frac = adopt[t,j]; r=prod.loc[j]
            arpu += frac*(r['ARPU Mult']-1)*arpu
            cac  += frac*(r['CAC Mult'] -1)*cac
            csc  += frac*(r['CSC Mult'] -1)*csc
        cac *= cac_eff[t]; csc *= csc_eff[t]
        act = active[t]; nc = newc[t]
        rev_mkt.iloc[t] = act*arpu/12
        cos_df.iloc[t]  = rev_mkt.iloc[t]*p['cos']
        acq_df.iloc[t]  = nc*cac
        serv_df.iloc[t] = act*csc/12
        gna_ser[t]      = gna_m
        if t>0:
            gr = (rev_mkt.iloc[t].sum()-rev_mkt.iloc[t-1].sum())/max(rev_mkt.iloc[t-1].sum(),1)
            gna_m *= (1+p['gna_share']*gr)
            tech_av[t] = tech_av[t-1]*(1+p['tech_share']*gr)
        tech_req[t] = tech_av[t]*p['hq_share']
        total_rev = rev_mkt.iloc[t].sum()
        total_cost= cos_df.iloc[t].sum()+acq_df.iloc[t].sum()+serv_df.iloc[t].sum()+gna_ser[t]+tech_req[t]*p['tech_cost']*tc_eff[t]
        cash_ser[t]= (cash_ser[t-1]+total_rev-total_cost) if t>0 else cash_ser[0]
    # Final aggregates
    costs_agg = pd.DataFrame({'CoS':cos_df.sum(axis=1),'Acquisition':acq_df.sum(axis=1),'Servicing':serv_df.sum(axis=1),'G&A':gna_ser,'Tech':tech_req*p['tech_cost']*tc_eff}, index=idxs)
    tech_df   = pd.DataFrame({'available':tech_av,'required':tech_req}, index=idxs)
    newc_sum = newc.sum(axis=1)
    denom    = np.where(newc_sum==0,1,newc_sum)
    metrics_df= pd.DataFrame({'CAC':acq_df.sum(axis=1)/denom,'ARPU':rev_mkt.sum(axis=1)/active.sum(axis=1)*12,'Churn Y1':(cy1*active).sum(axis=1)/active.sum(axis=1),'Churn Post':(cp*active).sum(axis=1)/active.sum(axis=1)}, index=idxs)
    return rev_mkt, costs_agg, tech_df, cash_ser, metrics_df, cos_df, acq_df, serv_df

# Run Simulation & Display
if st.button('Run Simulation'):
    rev_mkt, costs_agg, tech_df, cash_ser, metrics_df, cos_df, acq_df, serv_df = simulate(inputs)

    # Revenue by market
    st.subheader('Revenue by Market')
    st.line_chart(rev_mkt[sel_mkt])

    # Costs by Market (stacked area)
    st.subheader('Costs by Market')
    # Prepare a long-form DataFrame for Altair
    cost_mkt_df = (cos_df + acq_df + serv_df)[sel_mkt].reset_index().melt(
        id_vars='index', var_name='Market', value_name='Cost'
    )
    import altair as alt
    cost_chart = alt.Chart(cost_mkt_df).mark_area().encode(
        x=alt.X('index:T', title='Date'),
        y=alt.Y('Cost:Q', stack='zero', title='Cost (USD)'),
        color=alt.Color('Market:N', title='Market')
    ).properties(width='container', height=300)
    st.altair_chart(cost_chart, use_container_width=True)

    # Aggregated Costs Breakdown (stacked area)
    st.subheader('Aggregated Costs Breakdown')
    agg_df = costs_agg.reset_index().melt(id_vars='index', var_name='Cost Type', value_name='Amount')
    agg_chart = alt.Chart(agg_df).mark_area().encode(
        x=alt.X('index:T', title='Date'),
        y=alt.Y('Amount:Q', stack='zero', title='Amount (USD)'),
        color=alt.Color('Cost Type:N', title='Cost Type')
    ).properties(width='container', height=300)
    st.altair_chart(agg_chart, use_container_width=True)

    # Cash Balance
    st.subheader('Cash Balance')
    st.line_chart(cash_ser)

    # Tech Capacity
    st.subheader('Tech Capacity')
    st.line_chart(tech_df)

    # Key Metrics
    st.subheader('Key Metrics')
    st.line_chart(metrics_df)

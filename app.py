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

# ─────────────────────────────────────────────────────────────────────────────
# App Title & How-It-Works
# ─────────────────────────────────────────────────────────────────────────────
st.title('Runway & Financial Simulator for Business Neobank')
if st.button('How it works'):
    st.markdown('## How it works')
    st.markdown(
        '''
Define base parameters, existing markets, new markets, new products, and efficiency projects in the tables. Click **Run Simulation** to project monthly over five years:

- Revenue by market
- Costs (CoS, Acquisition, Servicing) by market + aggregated G&A & Tech
- Tech capacity vs requirement
- Cash balance
- Key metrics (CAC, ARPU, NRR, churn)
'''    )

# ─────────────────────────────────────────────────────────────────────────────
# Input Tables
# ─────────────────────────────────────────────────────────────────────────────
# 1) Base parameters
st.header('Base Parameters')
col1, col2 = st.columns(2)
with col1:
    cash_start = st.number_input('Cash in bank (USD)', 70000000)
    cos_pct = st.slider('Cost of Sales (% of Revenue)', 0.0, 1.0, 0.4)
    hq_gna = st.number_input('HQ G&A annual (USD)', 15000000)
    hq_share = st.slider('HQ tech bandwidth share (%)', 0.0, 1.0, 0.5)
with col2:
    tech_units = st.number_input('Tech units available per month', 50)
    tech_cost = st.number_input('Tech unit cost (USD)', 30000)
    gna_share = st.slider('HQ G&A growth share', 0.0, 1.0, 0.5)
    tech_share = st.slider('Tech capacity growth share', 0.0, 1.0, 1.0)

# 2) Existing markets
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

# 3) New markets
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

# 4) New products
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

# 5) Efficiency projects
st.header('Efficiency Projects')
eff_cols = ['Project','Start Date','Duration','Tech/mo','CAC Mult','CSC Mult','TechCost Mult']
eff_df = pd.DataFrame(columns=eff_cols)
eff_df = st.data_editor(eff_df, key='eff', num_rows='dynamic')

# Gather all inputs
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
if save_btn and new_name:
    save_scenario(new_name, inputs)
    st.sidebar.success(f"Saved '{new_name}'")

# ─────────────────────────────────────────────────────────────────────────────
# Simulation Logic (corrected)
# ─────────────────────────────────────────────────────────────────────────────
def simulate(inputs):
    base = pd.DataFrame(inputs['base'])
    newm = pd.DataFrame(inputs['new'])
    prod = pd.DataFrame(inputs['prod'])
    effp = pd.DataFrame(inputs['eff'])
    p = inputs['params']
    # timeline
    dates = pd.date_range(START_DATE, periods=MONTHS, freq='MS')
    n = MONTHS
    # markets
    m0 = len(base)
    m1 = len(newm)
    markets = list(base['Market']) + list(newm['Market'])
    # churn monthly
    churn_y1 = np.concatenate([base['Churn Y1'], newm['Churn Y1']]).astype(float)
    churn_post = np.concatenate([base['Churn Post Y1'], newm['Churn Post Y1']]).astype(float)
    churn_y1_m = 1 - (1 - churn_y1)**(1/12)
    churn_post_m = 1 - (1 - churn_post)**(1/12)
    # build cohorts
    cohorts = np.zeros((n, m0+m1, n))
    for i in range(m0): cohorts[0,i,12] = base.at[i,'Existing Clients']
    # schedule new clients
    newc = np.zeros((n,m0+m1))
    for i in range(m0):
        ys = base.loc[i,['New Y1','New Y2','New Y3','New Y4','New Y5']].astype(float).values
        for yi in range(5): newc[yi*12:(yi+1)*12,i] = ys[yi]/12
    for i in range(m1):
        r=newm.loc[i]
        idx=(pd.to_datetime(r['Start Date']).year-START_DATE.year)*12 + pd.to_datetime(r['Start Date']).month-1
        prep=int(r['Prep mo'])
        ys=r[['New Y1','New Y2','New Y3','New Y4','New Y5']].astype(float).values
        for yi in range(5):
            s=idx+prep+yi*12; e=min(s+12,n)
            if s<n: newc[s:e,m0+i]=ys[yi]/12
    # simulate cohorts
    for t in range(1,n):
        for i in range(m0+m1):
            prev=cohorts[t-1,i]; curr=np.zeros(n); curr[0]=newc[t,i]
            for age in range(1,n): rate=churn_y1_m[i] if age<12 else churn_post_m[i]; curr[age]=prev[age-1]*(1-rate)
            cohorts[t,i]=curr
    active = cohorts.sum(axis=2)
    # product adoption
    adopt=np.zeros((n,len(prod)))
    for j in range(len(prod)):
        r=prod.loc[j]; idx=(pd.to_datetime(r['Start Date']).year-START_DATE.year)*12+pd.to_datetime(r['Start Date']).month-1
        prep=int(r['Prep mo']); ys=r[['Ad1','Ad2','Ad3','Ad4','Ad5']].astype(float).values; arr=np.zeros(n)
        for yi in range(5): s=idx+prep+yi*12; e=min(s+12,n);
            if s<n: arr[s:e]=ys[yi]/12
        adopt[:,j]=np.cumsum(arr)
    # efficiency multipliers
    eff_active=np.zeros((n,len(effp)))
    for j in range(len(effp)):
        r=effp.loc[j]; idx=(pd.to_datetime(r['Start Date']).year-START_DATE.year)*12+pd.to_datetime(r['Start Date']).month-1
        dur=int(r['Duration']); eff_active[idx:idx+dur,j]=1
    cac_eff=np.ones(n); csc_eff=np.ones(n); tech_cost_eff=np.ones(n)
    for j in range(len(effp)):
        mrow=effp.loc[j,['CAC Mult','CSC Mult','TechCost Mult']].astype(float)
        mask=eff_active[:,j]==1; cac_eff[mask]*=mrow[0]; csc_eff[mask]*=mrow[1]; tech_cost_eff[mask]*=mrow[2]
    # prepare dataframes
    dates=pd.DatetimeIndex(dates)
    rev_mkt=pd.DataFrame(0,index=dates,columns=markets)
    cos_df=pd.DataFrame(0,index=dates,columns=markets)
    acq_df=pd.DataFrame(0,index=dates,columns=markets)
    serv_df=pd.DataFrame(0,index=dates,columns=markets)
    tech_avail=np.zeros(n); tech_req=np.zeros(n)
    gna_series=np.zeros(n); cash_series=np.zeros(n)
    # init
    gna_month=p['hq_gna']/12; cash_series[0]=p['cash']; tech_avail[0]=p['tech_units']
    # run months
    for t in range(n):
        # base metric arrays
        arpu_base=np.concatenate([base['ARPU'],newm['ARPU']]).astype(float)
        cac_base=np.concatenate([base['CAC'],newm['CAC']]).astype(float)
        csc_base=np.concatenate([base['CSC'],newm['CSC']]).astype(float)
        arpu=arpu_base.copy(); cac=cac_base.copy(); csc=csc_base.copy()
        # apply product effects
        for j in range(len(prod)):
            frac=adopt[t,j]; r=prod.loc[j]
            arpu += frac*(r['ARPU Mult']-1)*arpu_base
            cac  += frac*(r['CAC Mult']-1)*cac_base
            csc  += frac*(r['CSC Mult']-1)*csc_base
        cac *= cac_eff[t]; csc *= csc_eff[t]
        # compute revenue and costs per market
        act = active[t]; newc_t=newc[t]
        rev_mkt.iloc[t] = act*arpu/12
        cos_df.iloc[t] = rev_mkt.iloc[t]*p['cos']
        acq_df.iloc[t] = newc_t*cac
        serv_df.iloc[t] = act*csc/12
        # record G&A
        gna_series[t] = gna_month
        # tech availability growth
        if t>0:
            rev_growth = (rev_mkt.iloc[t].sum()-rev_mkt.iloc[t-1].sum())/max(rev_mkt.iloc[t-1].sum(),1)
            gna_month *= (1 + p['gna_share']*rev_growth)
            tech_avail[t] = tech_avail[t-1] * (1 + p['tech_share']*rev_growth)
        # tech requirement (HQ share only here; extend with projects if needed)
        tech_req[t] = tech_avail[t]*p['hq_share']
        # update cash
        total_rev = rev_mkt.iloc[t].sum()
        total_cost = cos_df.iloc[t].sum() + acq_df.iloc[t].sum() + serv_df.iloc[t].sum() + gna_series[t] + tech_req[t]*p['tech_cost']*tech_cost_eff[t]
        cash_series[t] = (cash_series[t-1] + total_rev - total_cost) if t>0 else cash_series[0]
    # aggregated costs
    costs_agg = pd.DataFrame({
        'CoS': cos_df.sum(axis=1),
        'Acquisition': acq_df.sum(axis=1),
        'Servicing': serv_df.sum(axis=1),
        'G&A': gna_series,
        'Tech': tech_req*p['tech_cost']*tech_cost_eff
    }, index=dates)
    # metrics
    newc_sum = newc.sum(axis=1)
    denom = np.where(newc_sum==0, 1, newc_sum)
    metrics_df = pd.DataFrame({
        'CAC': acq_df.sum(axis=1)/denom,
        'ARPU': rev_mkt.sum(axis=1)/active.sum(axis=1)*12,
        'Churn Y1': (churn_y1_m*active).sum(axis=1)/active.sum(axis=1),
        'Churn Post': (churn_post_m*active).sum(axis=1)/active.sum(axis=1)
    }, index=dates)
    return rev_mkt, costs_agg, pd.DataFrame({'available':tech_avail,'required':tech_req}, index=dates), cash_series, metrics_df, cos_df, acq_df, serv_df

# ─────────────────────────────────────────────────────────────────────────────
# Run & Display
# ─────────────────────────────────────────────────────────────────────────────
if st.button('Run Simulation'):
    rev_mkt, costs_agg, tech_df, cash_series, metrics_df, cos_df, acq_df, serv_df = simulate(inputs)
    # revenue filter
    st.subheader('Revenue by Market')
    sel = st.multiselect('Markets', rev_mkt.columns.tolist(), default=rev_mkt.columns.tolist())
    st.line_chart(rev_mkt[sel])
    # costs by market
    st.subheader('Costs by Market')
    total_costs_mkt = cos_df + acq_df + serv_df
    cost_sel = st.multiselect('Cost markets', total_costs_mkt.columns.tolist(), default=total_costs_mkt.columns.tolist())
    st.area_chart(total_costs_mkt[cost_sel])
    # aggregated costs
    st.subheader('Aggregated Costs Breakdown')
    st.area_chart(costs_agg)
    # cash
    st.subheader('Cash Balance')
    st.line_chart(cash_series)
    # tech
    st.subheader('Tech Capacity')
    st.line_chart(tech_df)
    # metrics
    st.subheader('Key Metrics')
    st.line_chart(metrics_df)

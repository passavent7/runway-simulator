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
# App Title and How-It-Works
# ─────────────────────────────────────────────────────────────────────────────
st.title('Runway & Financial Simulator for Business Neobank')
if st.button('How it works'):
    st.markdown('## How it works')
    st.markdown(
        '''
Define your base parameters, existing markets, new markets, new products and efficiency projects in the tables below. Then click **Run Simulation** to project month-by-month over five years:

- Revenue by market
- Costs (CoS, Acquisition, Servicing) by market + aggregated G&A and Tech
- Tech capacity vs. requirement
- Cash balance
- Key metrics (CAC, ARPU, NRR, churn)
'''    )

# ─────────────────────────────────────────────────────────────────────────────
# Input Sections
# ─────────────────────────────────────────────────────────────────────────────
# 1) Base parameters
st.header('Base Parameters')
col1, col2 = st.columns(2)
with col1:
    cash_start = st.number_input('Cash in bank (USD)', 70000000, step=1000000)
    cos_pct = st.slider('Cost of Sales (% of Revenue)', 0.0, 1.0, 0.4)
    hq_gna_start = st.number_input('HQ G&A annual (USD)', 15000000, step=500000)
    hq_share = st.slider('HQ tech bandwidth share (%)', 0.0, 1.0, 0.5)
with col2:
    tech_units_start = st.number_input('Tech units/month', 50)
    tech_unit_cost_start = st.number_input('Tech unit cost (USD)', 30000)
    gna_growth_share = st.slider('HQ G&A growth share', 0.0, 1.0, 0.5)
    tech_growth_share = st.slider('Tech capacity growth share', 0.0, 1.0, 1.0)

# 2) Existing markets
st.header('Existing Markets (fixed rows)')
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

# Gather inputs
inputs = dict(
    base=base_df.to_dict('list'),
    new=new_df.to_dict('list'),
    prod=prod_df.to_dict('list'),
    eff=eff_df.to_dict('list'),
    params=dict(cash=cash_start, cos=cos_pct,
                hq_gna=hq_gna_start, hq_share=hq_share,
                gna_share=gna_growth_share,
                tech_units=tech_units_start,
                tech_cost=tech_unit_cost_start,
                tech_share=tech_growth_share)
)
if save_btn and new_name:
    save_scenario(new_name, inputs)
    st.sidebar.success(f"Saved '{new_name}'")

# ─────────────────────────────────────────────────────────────────────────────
# Simulation Logic
# ─────────────────────────────────────────────────────────────────────────────
def simulate(inputs):
    # Unpack
    base = pd.DataFrame(inputs['base'])
    newm = pd.DataFrame(inputs['new'])
    prod = pd.DataFrame(inputs['prod'])
    effp = pd.DataFrame(inputs['eff'])
    p = inputs['params']
    # Timeline
    dates = pd.date_range(start=START_DATE, periods=MONTHS, freq='MS')
    n = MONTHS
    # Combine markets
    m0 = len(base)
    m1 = len(newm)
    markets = list(base['Market']) + list(newm['Market'])
    # Churn rates
    churn_y1 = np.concatenate([base['Churn Y1'], newm['Churn Y1']]).astype(float)
    churn_post = np.concatenate([base['Churn Post Y1'], newm['Churn Post Y1']]).astype(float)
    churn_y1_m = 1 - (1 - churn_y1)**(1/12)
    churn_post_m = 1 - (1 - churn_post)**(1/12)
    # Existing cohorts
    cohorts = np.zeros((n, m0+m1, n))
    for i in range(m0):
        cohorts[0,i,12] = base.at[i,'Existing Clients']
    # New clients schedule
    newc = np.zeros((n, m0+m1))
    # base markets
    for i in range(m0):
        yrs = base.loc[i,['New Y1','New Y2','New Y3','New Y4','New Y5']].astype(float).values
        for yi in range(5):
            newc[yi*12:(yi+1)*12,i] = yrs[yi]/12
    # new markets
    for i in range(m1):
        r = newm.loc[i]
        idx = (pd.to_datetime(r['Start Date']).year-START_DATE.year)*12 + (pd.to_datetime(r['Start Date']).month-1)
        prep = int(r['Prep mo'])
        yrs = r[['New Y1','New Y2','New Y3','New Y4','New Y5']].astype(float).values
        for yi in range(5):
            s = idx+prep+yi*12
            e = min(s+12,n)
            if s<n: newc[s:e,m0+i] = yrs[yi]/12
    # Simulate cohorts
    for t in range(1,n):
        for i in range(m0+m1):
            # age
            prev = cohorts[t-1,i]
            curr = np.zeros(n)
            curr[0] = newc[t,i]
            for age in range(1,n):
                rate = churn_y1_m[i] if age<12 else churn_post_m[i]
                curr[age] = prev[age-1]*(1-rate)
            cohorts[t,i] = curr
    active = cohorts.sum(axis=2)
    # Products adoption
    adopt = np.zeros((n,len(prod)))
    for j in range(len(prod)):
        r = prod.loc[j]
        idx = (pd.to_datetime(r['Start Date']).year-START_DATE.year)*12+(pd.to_datetime(r['Start Date']).month-1)
        prep = int(r['Prep mo'])
        yrs = r[['Ad1','Ad2','Ad3','Ad4','Ad5']].astype(float).values
        arr = np.zeros(n)
        for yi in range(5):
            s = idx+prep+yi*12; e=min(s+12,n)
            if s<n: arr[s:e]=yrs[yi]/12
        adopt[:,j]=np.cumsum(arr)
    # Efficiency multipliers
    eff_active = np.zeros((n,len(effp)))
    for j in range(len(effp)):
        r=effp.loc[j]; idx=(pd.to_datetime(r['Start Date']).year-START_DATE.year)*12+(pd.to_datetime(r['Start Date']).month-1)
        dur=int(r['Duration']); eff_active[idx:idx+dur,j]=1
    cac_eff=csc_eff=tech_cost_eff=np.ones(n)
    for j in range(len(effp)):
        mrow=effp.loc[j,['CAC Mult','CSC Mult','TechCost Mult']].astype(float)
        cac_eff*=np.where(eff_active[:,j],mrow[0],1)
        csc_eff*=np.where(eff_active[:,j],mrow[1],1)
        tech_cost_eff*=np.where(eff_active[:,j],mrow[2],1)
    # Pre-alloc
    dates=pd.DatetimeIndex(dates)
    rev_mkt=pd.DataFrame(0,index=dates,columns=markets)
    cost_cos=cost_acq=cost_serv=pd.DataFrame(0,index=dates,columns=markets)
    tech_avail=tech_req=np.zeros(n)
    gna_month=p['hq_gna']/12; cash=np.zeros(n)
    cash[0]=p['cash']
    # Run
    for t in range(n):
        # effective rates
        arpu_base=np.concatenate([base['ARPU'],newm['ARPU']]).astype(float)
        cac_base=np.concatenate([base['CAC'],newm['CAC']]).astype(float)
        csc_base=np.concatenate([base['CSC'],newm['CSC']]).astype(float)
        arpu=arpu_base.copy(); cac=cac_base.copy(); csc=csc_base.copy()
        # apply products
        for j in range(len(prod)):
            frac=adopt[t,j]
            r=prod.loc[j]
            arpu+=frac*(r['ARPU Mult']-1)*arpu_base
            cac+=frac*(r['CAC Mult']-1)*cac_base
            csc+=frac*(r['CSC Mult']-1)*csc_base
        cac*=cac_eff[t]; csc*=csc_eff[t]
        # counts
        act=active[t]; newc_t=newc[t]
        # revenue
        rev_mkt.iloc[t]=act*arpu/12
        # costs per market
        cost_cos.iloc[t]=rev_mkt.iloc[t]*p['cos']
        cost_acq.iloc[t]=newc_t*cac
        cost_serv.iloc[t]=act*csc/12
        # G&A
        if t>0:
            gr=(rev_mkt.iloc[t].sum()-rev_mkt.iloc[t-1].sum())/max(rev_mkt.iloc[t-1].sum(),1)
            gna_month*=1+p['gna_share']*gr
        # tech
        tech_avail[t]=tech_avails if (tech_avails:=tech_avail[t-1] if t>0 else p['tech_units']) and False else (p['tech_units'] if t==0 else tech_avail[t-1]*(1+p['tech_share']*gr))
        # TODO: compute tech_req similar to costs
        tech_req[t]=tech_avail[t]*p['hq_share']
        # cash update
        total_rev=rev_mkt.iloc[t].sum(); total_cost=cost_cos.iloc[t].sum()+cost_acq.iloc[t].sum()+cost_serv.iloc[t].sum()+gna_month+tech_req[t]*p['tech_cost']*tech_cost_eff[t]
        cash[t]=cash[t-1] + total_rev - total_cost if t>0 else cash[0]
    # aggregate
    costs_agg=pd.DataFrame({
        'CoS':cost_cos.sum(axis=1),'Acquisition':cost_acq.sum(axis=1),'Servicing':cost_serv.sum(axis=1),
        'G&A':gna_month,'Tech':tech_req*p['tech_cost']*tech_cost_eff
    },index=dates)
    tech_df=pd.DataFrame({'available':tech_avail,'required':tech_req},index=dates)
    metrics=pd.DataFrame({
        'CAC':cost_acq.sum(axis=1)/newc.sum(axis=1).replace(0,1),
        'ARPU':rev_mkt.sum(axis=1)/active.sum(axis=1)*12,
        'Churn Y1':(churn_y1_m*active).sum(axis=1)/active.sum(axis=1),
        'Churn Post':(churn_post_m*active).sum(axis=1)/active.sum(axis=1)
    },index=dates)
    return rev_mkt, costs_agg, tech_df, cash, metrics, cost_cos, cost_acq, cost_serv

# ─────────────────────────────────────────────────────────────────────────────
# Run & Display
# ─────────────────────────────────────────────────────────────────────────────
if st.button('Run Simulation'):
    rev_mkt, costs_agg, tech_df, cash_series, metrics_df, cos_df, acq_df, serv_df = simulate(inputs)
    # revenue filter
    st.subheader('Revenue by Market')
    sel=st.multiselect('Markets', rev_mkt.columns.tolist(), default=rev_mkt.columns.tolist())
    st.line_chart(rev_mkt[sel])
    # costs per market filter
    st.subheader('Costs by Market (CoS+Acq+Serv)')
    total_costs_mkt = cos_df+acq_df+serv_df
    st.multiselect('Cost markets', total_costs_mkt.columns.tolist(), default=total_costs_mkt.columns.tolist(), key='cost_sel')
    st.area_chart(total_costs_mkt[st.session_state.cost_sel])
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

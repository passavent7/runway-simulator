import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import altair as alt

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Utilities
# ─────────────────────────────────────────────────────────────────────────────
START_DATE = pd.Timestamp('2025-07-01')  # simulation start date
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
# Sidebar: Scenario Management & Smoothing
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title('Runway & Financial Simulator')
scenarios = load_scenarios()
choice = st.sidebar.selectbox('Load scenario', ['(new)'] + list(scenarios.keys()))
scenario_data = scenarios.get(choice)
new_name = st.sidebar.text_input('Save scenario as')
smoothing_window = st.sidebar.slider('Rolling average window (months)', 1, 12, 3)
if st.sidebar.button('Save') and new_name:
    # Save current inputs
    save_scenario(new_name, st.session_state['inputs'])
    st.sidebar.success(f"Saved '{new_name}'")

# ─────────────────────────────────────────────────────────────────────────────
# App Header & How It Works
# ─────────────────────────────────────────────────────────────────────────────
st.title('Runway & Financial Simulator')
with st.expander('How it works'):
    prompt_path = Path('instructions.md')
    if prompt_path.exists():
        st.markdown(prompt_path.read_text())
    else:
        st.warning('Please add an instructions.md with full prompt details.')

# ─────────────────────────────────────────────────────────────────────────────
# Input: Base Parameters
# ─────────────────────────────────────────────────────────────────────────────
st.header('Base Parameters')
col1, col2 = st.columns(2)
with col1:
    cash_start = st.number_input('Cash in bank (USD)', 70_000_000)
    cos_pct = st.slider('CoS % of Revenue', 0.0, 1.0, 0.4)
    hq_gna = st.number_input('HQ G&A annual (USD)', 15_000_000)
    hq_share = st.slider('HQ tech share', 0.0, 1.0, 0.5)
with col2:
    tech_units = st.number_input('Tech units/mo', 50)
    tech_cost = st.number_input('Tech unit cost (USD)', 30_000)
    gna_share = st.slider('G&A growth share', 0.0, 1.0, 0.5)
    tech_share = st.slider('Tech growth share', 0.0, 1.0, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Table Editor
# ─────────────────────────────────────────────────────────────────────────────
def table_editor(key, cols, init=None):
    df = pd.DataFrame(columns=cols)
    if init is not None:
        df.loc[0] = init
    return st.data_editor(df, key=key, num_rows='dynamic')

# ─────────────────────────────────────────────────────────────────────────────
# Input: Existing Markets
# ─────────────────────────────────────────────────────────────────────────────
st.header('Existing Markets')
base_cols = [
    'Market','Existing Clients','CAC','ARPU','NRR','Churn Y1','Churn Post',
    'New1','New2','New3','New4','New5','CSC','Tech/mo','G&A/yr'
]
base_df = pd.DataFrame([
    ['Singapore',20000,500,1500,1.0,0.3,0.1,6000,6000,6000,6000,6000,200,4,2_000_000],
    ['Hong-Kong',10000,500,1500,1.0,0.3,0.05,1000,3000,6000,10000,15000,200,4,2_000_000]
], columns=base_cols)
base_df = st.data_editor(base_df, key='base', num_rows='fixed')

# ─────────────────────────────────────────────────────────────────────────────
# Input: New Markets
# ─────────────────────────────────────────────────────────────────────────────
st.header('New Markets')
new_cols = [
    'Market','Start','Prep mo','Prep Tech/mo','Prep G&A/mo','CAC','ARPU',
    'Churn Y1','Churn Post','New1','New2','New3','New4','New5','CSC',
    'M1','M2','M3','M4','M5','G1','G2','G3','G4','G5'
]
new_init = [
    'United States','2025-08-01',0,0,0,1000,2000,0.3,0.1,
    1000,3000,10000,25000,50000,300,3,4,5,6,7,1000000,2000000,3000000,4000000,5000000
]
new_df = table_editor('new', new_cols, new_init)

# ─────────────────────────────────────────────────────────────────────────────
# Input: New Products
# ─────────────────────────────────────────────────────────────────────────────
st.header('New Products')
prod_cols = [
    'Product','Start','Prep mo','Prep Tech/mo','Prep G&A/mo','CAC Mult','ARPU Mult',
    'Churn1 Mult','ChurnP Mult','CSC Mult','Ad1','Ad2','Ad3','Ad4','Ad5',
    'M1','M2','M3','M4','M5','G1','G2','G3','G4','G5'
]
prod_init = [
    'AI Accounting','2025-07-01',6,2,20000,0.95,1.3,0.9,0.9,1.2,
    0.02,0.05,0.1,0.2,0.3,3,3,3,3,3,250000,500000,750000,1000000,1500000
]
prod_df = table_editor('prod', prod_cols, prod_init)

# ─────────────────────────────────────────────────────────────────────────────
# Input: Efficiency Projects
# ─────────────────────────────────────────────────────────────────────────────
st.header('Efficiency Projects')
eff_cols = ['Project','Start','Duration','Tech/mo','CAC Mult','CSC Mult','TechCost Mult']
eff_df = table_editor('eff', eff_cols)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Filters
# ─────────────────────────────────────────────────────────────────────────────
markets = list(base_df['Market']) + list(new_df['Market'])
sel_mkt = st.sidebar.multiselect('Markets', markets, default=markets)
products = list(prod_df['Product'])
sel_prod = st.sidebar.multiselect('Products', products, default=products)
eff_list = list(eff_df['Project'])
sel_eff = st.sidebar.multiselect('Efficiency', eff_list, default=eff_list)

# ─────────────────────────────────────────────────────────────────────────────
# Gather Inputs
# ─────────────────────────────────────────────────────────────────────────────
inputs = {
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
st.session_state['inputs'] = inputs

# ─────────────────────────────────────────────────────────────────────────────
# Simulation Function
# ─────────────────────────────────────────────────────────────────────────────
def simulate(inp):
    base = pd.DataFrame(inp['base'])
    newm = pd.DataFrame(inp['new'])
    prod = pd.DataFrame(inp['prod'])
    effp = pd.DataFrame(inp['eff'])
    p = inp['params']
    # Timeline
    dates = pd.date_range(START_DATE, periods=MONTHS, freq='MS')
    n = MONTHS
    m0 = len(base)
    m1 = len(newm)
    markets = list(base['Market']) + list(newm['Market'])
    # Monthly churn
    churn_y1 = np.concatenate([base['Churn Y1'].astype(float), newm['Churn Y1'].astype(float)])
    churn_post = np.concatenate([base['Churn Post'].astype(float), newm['Churn Post'].astype(float)])
    cy1 = 1 - (1 - churn_y1) ** (1/12)
    cp = 1 - (1 - churn_post) ** (1/12)
    # Initialize cohorts
    cohorts = np.zeros((n, m0 + m1, n))
    for i in range(m0): cohorts[0, i, 12] = base.at[i, 'Existing Clients']
    # New clients schedule
    newc = np.zeros((n, m0 + m1))
    for i in range(m0):
        ys = base.loc[i, ['New1','New2','New3','New4','New5']].astype(float).values
        for yi in range(5): newc[yi*12:(yi+1)*12, i] = ys[yi] / 12
    for i in range(m1):
        r = newm.loc[i]
        idx = (pd.to_datetime(r['Start']).year - START_DATE.year)*12 + (pd.to_datetime(r['Start']).month - 1)
        prep = int(r['Prep mo'])
        ys = r[['New1','New2','New3','New4','New5']].astype(float).values
        for yi in range(5):
            s = idx + prep + yi*12; e = min(s + 12, n)
            if s < n: newc[s:e, m0 + i] = ys[yi] / 12
    # Simulate cohorts
    for t in range(1, n):
        for i in range(m0 + m1):
            prev = cohorts[t-1, i]
            curr = np.zeros(n)
            curr[0] = newc[t, i]
            for age in range(1, n):
                rate = cy1[i] if age < 12 else cp[i]
                curr[age] = prev[age-1] * (1 - rate)
            curr[n-1] += prev[n-1] * (1 - cp[i])
            cohorts[t, i] = curr
    active = cohorts.sum(axis=2)
    # Product adoption
    adopt = np.zeros((n, len(prod)))
    for j in range(len(prod)):
        r = prod.loc[j]
        idx = (pd.to_datetime(r['Start']).year - START_DATE.year)*12 + (pd.to_datetime(r['Start']).month - 1)
        prep = int(r['Prep mo'])
        ys = r[['Ad1','Ad2','Ad3','Ad4','Ad5']].astype(float).values
        arr = np.zeros(n)
        for yi in range(5):
            s = idx + prep + yi*12; e = min(s + 12, n)
            if s < n:
                arr[s:e] = ys[yi] / 12
        adopt[:, j] = np.cumsum(arr)
    # Efficiency multipliers
    eff_active = np.zeros((n, len(effp)))
    for j in range(len(effp)):
        r = effp.loc[j]
        idx = (pd.to_datetime(r['Start']).year - START_DATE.year)*12 + (pd.to_datetime(r['Start']).month - 1)
        dur = int(r['Duration'])
        eff_active[idx:idx+dur, j] = 1
    cac_eff = np.ones(n); csc_eff = np.ones(n); tc_eff = np.ones(n)
    for j in range(len(effp)):
        mrow = effp.loc[j, ['CAC Mult','CSC Mult','TechCost Mult']].astype(float).values
        mask = eff_active[:, j] == 1
        cac_eff[mask] *= mrow[0]
        csc_eff[mask] *= mrow[1]
        tc_eff[mask]  *= mrow[2]
    # Prepare P&L containers
    idxs = pd.DatetimeIndex(dates)
    rev_mkt   = pd.DataFrame(0, index=idxs, columns=markets)
    cos_df    = pd.DataFrame(0, index=idxs, columns=markets)
    acq_df    = pd.DataFrame(0, index=idxs, columns=markets)
    serv_df   = pd.DataFrame(0, index=idxs, columns=markets)
    tech_av   = np.zeros(n); tech_req  = np.zeros(n)
    gna_ser   = np.zeros(n); cash_ser  = np.zeros(n)
    # Initial values
    gna_month = p['hq_gna'] / 12
    cash_ser[0] = p['cash']
    tech_av[0]  = p['tech_units']
    # Monthly loop
    for t in range(n):
        base_arpu = np.concatenate([base['ARPU'], newm['ARPU']]).astype(float)
        base_cac  = np.concatenate([base['CAC'], newm['CAC']]).astype(float)
        base_csc  = np.concatenate([base['CSC'], newm['CSC']]).astype(float)
        arpu = base_arpu.copy(); cac = base_cac.copy(); csc = base_csc.copy()
        for j in range(len(prod)):
            frac = adopt[t, j]
            r = prod.loc[j]
            arpu += frac * (r['ARPU Mult'] - 1) * base_arpu
            cac  += frac * (r['CAC Mult'] - 1) * base_cac
            csc  += frac * (r['CSC Mult'] - 1) * base_csc
        cac *= cac_eff[t]; csc *= csc_eff[t]
        act = active[t]; nc = newc[t]
        rev_mkt.iloc[t] = act * arpu / 12
        cos_df.iloc[t]  = rev_mkt.iloc[t] * p['cos']
        acq_df.iloc[t]  = nc * cac
        serv_df.iloc[t] = act * csc / 12
        gna_ser[t]      = gna_month
        if t > 0:
            rev_prev = rev_mkt.iloc[t-1].sum(); rev_cur = rev_mkt.iloc[t].sum()
            growth   = (rev_cur - rev_prev) / max(rev_prev, 1)
            gna_month *= (1 + p['gna_share'] * growth)
            tech_av[t] = tech_av[t-1] * (1 + p['tech_share'] * growth)
        tech_req[t]    = tech_av[t] * p['hq_share']
        total_rev      = rev_mkt.iloc[t].sum()
        total_cost     = (cos_df.iloc[t].sum() + acq_df.iloc[t].sum() + serv_df.iloc[t].sum() + gna_ser[t] + tech_req[t] * p['tech_cost'] * tc_eff[t])
        cash_ser[t]    = cash_ser[t-1] + total_rev - total_cost if t > 0 else cash_ser[0]
    costs_agg = pd.DataFrame({
        'CoS': cos_df.sum(axis=1),
        'Acquisition': acq_df.sum(axis=1),
        'Servicing': serv_df.sum(axis=1),
        'G&A': gna_ser,
        'Tech': tech_req * p['tech_cost'] * tc_eff
    }, index=idxs)
    tech_df = pd.DataFrame({'available': tech_av, 'required': tech_req}, index=idxs)
    # Churn & metrics
    denom = np.where(newc.sum(axis=1)==0,1,newc.sum(axis=1))
    metrics_df = pd.DataFrame({
        'CAC': acq_df.sum(axis=1)/denom,
        'ARPU': rev_mkt.sum(axis=1)/active.sum(axis=1)*12,
        'CSC': (serv_df.sum(axis=1)/active.sum(axis=1))*12,
        'Churn Y1': ((cy1*active).sum(axis=1)/active.sum(axis=1)),
        'Churn Post': ((cp*active).sum(axis=1)/active.sum(axis=1))
    }, index=idxs)
    return rev_mkt, costs_agg, tech_df, cash_ser, metrics_df, cos_df, acq_df, serv_df, active

# ─────────────────────────────────────────────────────────────────────────────
# Run Simulation & Display
# ─────────────────────────────────────────────────────────────────────────────
if st.button('Run Simulation'):
    rev_mkt, costs_agg, tech_df, cash_ser, metrics_df, cos_df, acq_df, serv_df, active = simulate(inputs)

    # Revenue by Market (stacked area)
    st.subheader('Revenue by Market')
    rev_df = rev_mkt[sel_mkt].reset_index().melt(id_vars='index', var_name='Market', value_name='Revenue')
    rev_chart = alt.Chart(rev_df).mark_area().encode(
        x=alt.X('index:T', title='Date'),
        y=alt.Y('Revenue:Q', title='Revenue (USD)', stack='zero'),
        color=alt.Color('Market:N', title='Market')
    ).properties(width='container', height=300)
    st.altair_chart(rev_chart, use_container_width=True)

    # Costs by Market (smoothed)
    st.subheader('Costs by Market')
    raw_cost = cos_df + acq_df + serv_df
    smooth_cost = raw_cost.rolling(window=smoothing_window, min_periods=1).mean()
    cost_df = smooth_cost[sel_mkt].reset_index().melt(id_vars='index', var_name='Market', value_name='Cost')
    cost_chart = alt.Chart(cost_df).mark_area().encode(
        x=alt.X('index:T', title='Date'),
        y=alt.Y('Cost:Q', title='Cost (USD)', stack='zero'),
        color=alt.Color('Market:N', title='Market')
    ).properties(width='container', height=300)
    st.altair_chart(cost_chart, use_container_width=True)

    # Aggregated Costs Breakdown
    st.subheader('Aggregated Costs Breakdown')
    agg_df = costs_agg.reset_index().melt(id_vars='index', var_name='Cost Type', value_name='Amount')
    agg_chart = alt.Chart(agg_df).mark_area().encode(
        x=alt.X('index:T', title='Date'),
        y=alt.Y('Amount:Q', title='Amount (USD)', stack='zero'),
        color=alt.Color('Cost Type:N', title='Cost Type')
    ).properties(width='container', height=300)
    st.altair_chart(agg_chart, use_container_width=True)

    # Cash Balance
    st.subheader('Cash Balance')
    st.line_chart(cash_ser)

    # Tech Capacity Breakdown
    st.subheader('Tech Capacity Breakdown')
    dates_idx = tech_df.index
    breakdown = pd.DataFrame({'HQ': tech_df['required']}, index=dates_idx)
    for i, market in enumerate(base_df['Market']): breakdown[market] = base_df.loc[i, 'Tech/mo']
    for i, market in enumerate(new_df['Market']):
        row = new_df.loc[i]
        dt = pd.to_datetime(row['Start'])
        start_idx = (dt.year - START_DATE.year)*12 + (dt.month - START_DATE.month)
        arr = np.zeros(len(dates_idx))
        prep = int(row['Prep mo'])
        if prep>0 and start_idx>=0:
            end_prep = min(start_idx+prep, len(arr)); arr[start_idx:end_prep] = row['Prep Tech/mo']
        for yi in range(1,6):
            m = row[f'M{yi}']; s = start_idx + prep + (yi-1)*12; e = min(s+12, len(arr))
            if 0 <= s < len(arr): arr[s:e] += m
        breakdown[market] = arr
    for j, prod_name in enumerate(prod_df['Product']):
        row = prod_df.loc[j]; dt = pd.to_datetime(row['Start'])
        start_idx = (dt.year - START_DATE.year)*12 + (dt.month - START_DATE.month)
        arr = np.zeros(len(dates_idx)); prep = int(row['Prep mo'])
        if prep>0 and start_idx>=0:
            end_prep = min(start_idx+prep, len(arr)); arr[start_idx:end_prep] = row['Prep Tech/mo']
        for yi in range(1,6): m = row[f'M{yi}']; s = start_idx+prep+(yi-1)*12; e = min(s+12,len(arr))
            if 0<=s<len(arr): arr[s:e]+=m
        breakdown[prod_name] = arr
    for k, proj in enumerate(eff_df['Project']):
        row = eff_df.loc[k]; start_idx = (pd.to_datetime(row['Start']).year-START_DATE.year)*12+(pd.to_datetime(row['Start']).month-1)
        dur = int(row['Duration']); arr = np.zeros(len(dates_idx)); end = min(start_idx+dur,len(arr))
        if start_idx < len(arr): arr[start_idx:end] = row['Tech/mo']
        breakdown[proj] = arr
    tech_long = breakdown.reset_index().melt(id_vars='index', var_name='Component', value_name='Units')
    tech_area = alt.Chart(tech_long).mark_area().encode(x='index:T', y='Units:Q', color='Component:N')
    avail_df = pd.DataFrame({'index': dates_idx, 'Available': tech_df['available']})
    avail_line = alt.Chart(avail_df).mark_line(size=2).encode(x='index:T', y='Available:Q')
    st.altair_chart(alt.layer(tech_area, avail_line), use_container_width=True)

    # Key Metrics
    st.subheader('Key Metrics')
    st.line_chart(metrics_df[['CAC','ARPU','CSC']])

    # Churn Metrics
    st.subheader('Churn Metrics')
    st.line_chart(metrics_df[['Churn Y1','Churn Post']])

    # Total Active Clients
    st.subheader('Total Active Clients')
    st.line_chart(active.sum(axis=1))

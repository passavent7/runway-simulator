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
if choice != '(new)':
    scenario_data = scenarios[choice]
else:
    scenario_data = None
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
base_markets_df = st.experimental_data_editor(
    base_markets_df, num_rows='fixed', key='base_markets',
    help='Core markets where we are live. Churn and new clients per year.'
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
# prefill United States
new_markets_df.loc[0] = [
    'United States', '2025-08-01', 0,
    0, 0,
    1000, 2000, 0.3, 0.1,
    1000,3000,10000,25000,50000,
    300,
    3,4,5,6,7,
    1_000_000,2_000_000,3_000_000,4_000_000,5_000_000
]
new_markets_df = st.experimental_data_editor(
    new_markets_df, key='new_markets',
    help='Launch parameters for new regions. New clients per year spread evenly.'
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
new_products_df = st.experimental_data_editor(
    new_products_df, key='new_products',
    help='Parameters for product launches. Adoption as fraction of base clients.'
)

# 5) Efficiency Projects
st.header('Efficiency Projects')
eff_cols = [
    'Project','Start Date','Duration (mo)',
    'Tech Units/mo','CAC Mult','CSC Mult','Tech Unit Cost Mult'
]
eff_df = pd.DataFrame(columns=eff_cols)
eff_df = st.experimental_data_editor(
    eff_df, key='efficiency', help='Bets to lower CAC, CSC, and tech cost.'
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
    # Parse inputs
    base_df = pd.DataFrame(inputs['base'])
    new_df = pd.DataFrame(inputs['new_markets'])
    prod_df = pd.DataFrame(inputs['new_products'])
    eff_df = pd.DataFrame(inputs['efficiency'])
    p = inputs['params']

    # Timeline
    dates = pd.date_range(start=START_DATE, periods=MONTHS, freq='MS')
    n = MONTHS

    # All markets list
    markets = list(base_df['Market']) + list(new_df['Market'])
    m0 = len(base_df)
    m1 = len(new_df)

    # Prepare churn monthly rates and existing-client cohorts
    churn_y1_m = []  # per market
    churn_post_m = []
    existing_cohorts = []  # initial cohort vector for each market
    for i in range(m0):
        a = base_df.at[i, 'Churn Y1']
        b = base_df.at[i, 'Churn Post Y1']
        churn_y1_m.append(1 - (1 - a)**(1/12))
        churn_post_m.append(1 - (1 - b)**(1/12))
        # put all existing clients into age=12 bucket
        vec = np.zeros(n)
        vec[12] = base_df.at[i, 'Existing Clients']
        existing_cohorts.append(vec)
    for i in range(m1):
        a = new_df.at[i, 'Churn Y1']
        b = new_df.at[i, 'Churn Post Y1']
        churn_y1_m.append(1 - (1 - a)**(1/12))
        churn_post_m.append(1 - (1 - b)**(1/12))
        existing_cohorts.append(np.zeros(n))
    churn_y1_m = np.array(churn_y1_m)
    churn_post_m = np.array(churn_post_m)

    # Build new-clients schedule per market
    new_clients = np.zeros((n, m0 + m1))
    # base markets: spread Y1-Y5 evenly
    for i in range(m0):
        yrs = base_df.loc[i, ['New Clients Y1','New Clients Y2','New Clients Y3','New Clients Y4','New Clients Y5']].values.astype(float)
        for yi in range(5):
            per_month = yrs[yi] / 12
            start = yi*12
            end = start + 12
            new_clients[start:end, i] += per_month
    # new markets: after prep
    for i in range(m1):
        row = new_df.loc[i]
        prep = int(row['Prep Duration (mo)'])
        start_date = pd.to_datetime(row['Start Date'])
        start_idx = (start_date.year - START_DATE.year)*12 + (start_date.month-1)
        yrs = row[['New Clients Y1','New Clients Y2','New Clients Y3','New Clients Y4','New Clients Y5']].values.astype(float)
        for yi in range(5):
            per_month = yrs[yi] / 12
            s = start_idx + prep + yi*12
            e = s + 12
            if s < n:
                new_clients[s:min(e,n), m0+i] += per_month

    # Active cohorts per market: shape (n, markets, n_age)
    active_cohorts = np.zeros((n, m0 + m1, n))
    for i in range(m0 + m1):
        active_cohorts[0, i] = existing_cohorts[i]
        # add new clients at t0 if any
        active_cohorts[0, i, 0] += new_clients[0, i]
    # Simulate cohorts over time
    for t in range(1, n):
        for i in range(m0 + m1):
            prev = active_cohorts[t-1, i]
            curr = np.zeros(n)
            # new clients
            curr[0] = new_clients[t, i]
            # age previous cohorts
            for age in range(1, n):
                prev_count = prev[age-1]
                rate = churn_y1_m[i] if age < 12 else churn_post_m[i]
                curr[age] = prev_count * (1 - rate)
            active_cohorts[t, i] = curr
    # Sum to get active clients per market
    active = active_cohorts.sum(axis=2)  # shape (n, markets)

    # Prepare adoption fractions per product over time
    prod_start = []
    prod_prep = []
    adop_rates = []  # list of arrays shape (n,) for each product
    for j in range(len(prod_df)):
        row = prod_df.loc[j]
        start_date = pd.to_datetime(row['Start Date'])
        start_idx = (start_date.year - START_DATE.year)*12 + (start_date.month-1)
        prep = int(row['Prep Duration (mo)'])
        # adoption per year
        yrs = row[['Adopt Y1','Adopt Y2','Adopt Y3','Adopt Y4','Adopt Y5']].values.astype(float)
        arr = np.zeros(n)
        for yi in range(5):
            rate = yrs[yi]
            s = start_idx + prep + yi*12
            e = s + 12
            if s < n:
                arr[s:min(e,n)] = rate/12  # monthly increment
        adop_rates.append(np.cumsum(arr))
        prod_start.append(start_idx + prep)
        prod_prep.append(prep)

    # Efficiency project multipliers
    eff_active = np.zeros((n, len(eff_df)))
    for j in range(len(eff_df)):
        row = eff_df.loc[j]
        sd = pd.to_datetime(row['Start Date'])
        start_idx = (sd.year - START_DATE.year)*12 + (sd.month-1)
        dur = int(row['Duration (mo)'])
        eff_active[start_idx:start_idx+dur, j] = 1
    # Compute combined efficiency multipliers each month
    cac_eff = np.ones(n)
    csc_eff = np.ones(n)
    tech_cost_eff = np.ones(n)
    for j in range(len(eff_df)):
        mults = eff_df.loc[j, ['CAC Mult','CSC Mult','Tech Unit Cost Mult']].astype(float)
        cac_eff *= np.where(eff_active[:,j]==1, mults[0], 1)
        csc_eff *= np.where(eff_active[:,j]==1, mults[1], 1)
        tech_cost_eff *= np.where(eff_active[:,j]==1, mults[2], 1)

    # Initialize arrays for metrics
    total_rev = np.zeros(n)
    total_cos = np.zeros(n)
    total_gna = np.zeros(n)
    total_acq = np.zeros(n)
    total_serv = np.zeros(n)
    tech_avail = np.zeros(n)
    tech_req = np.zeros(n)
    cash = np.zeros(n)
    cac_series = np.zeros(n)
    arpu_series = np.zeros(n)
    churn_y1_series = np.zeros(n)
    churn_post_series = np.zeros(n)
    nrr_series = np.zeros(n)

    # Tech & G&A starting
    tech_avail[0] = p['tech_units_start']
    gna_month = p['hq_gna_start'] / 12
    cash[0] = p['cash_start']

    # Pre-calc base ARPU & CAC & CSC per market
    base_arpu = np.concatenate([base_df['ARPU'].values, new_df['ARPU'].values]).astype(float)
    base_cac = np.concatenate([base_df['CAC'].values, new_df['CAC'].values]).astype(float)
    base_csc = np.concatenate([base_df['CSC'].values, new_df['CSC'].values]).astype(float)
    hq_share = p['hq_share']
    cos_pct = p['cos_pct']

    # Loop months
    for t in range(n):
        # Revenue & metrics
        # effective ARPU/CAC/Churn per market by adoption
        adopt_mult_arpu = np.ones(m0+m1)
        adopt_mult_cac = np.ones(m0+m1)
        adopt_churn_y1 = np.zeros(m0+m1)
        adopt_churn_post = np.zeros(m0+m1)
        adopt_mult_csc = np.ones(m0+m1)
        for j in range(len(prod_df)):
            frac = adop_rates[j][t]
            row = prod_df.loc[j]
            adapt_arpu = row['ARPU Mult']
            adapt_cac = row['CAC Mult']
            adapt_ch1 = row['Churn Y1 Mult']
            adapt_chp = row['Churn Post Y1 Mult']
            adapt_csc = row['CSC Mult']
            adopt_mult_arpu += frac*(adapt_arpu-1)
            adopt_mult_cac += frac*(adapt_cac-1)
            adopt_mult_csc += frac*(adapt_csc-1)
            # for churn, reduce:
            adopt_churn_y1 += frac*(row['Churn Y1 Mult'] - 1)
            adopt_churn_post += frac*(row['Churn Post Y1 Mult'] - 1)
        # final churn rates per market
        churn_y1_eff = np.clip(churn_y1_m * (1 + adopt_churn_y1), 0, 1)
        churn_post_eff = np.clip(churn_post_m * (1 + adopt_churn_post), 0, 1)

        # apply efficiency multipliers
        cac_eff_t = cac_eff[t]
        csc_eff_t = csc_eff[t]
        tech_cost_eff_t = tech_cost_eff[t]

        # compute market-level ARPU/CAC/CSC
        arpu_t = base_arpu * adopt_mult_arpu
        cac_t = base_cac * adopt_mult_cac * cac_eff_t
        csc_t = base_csc * (1 + adopt_mult_csc-1) * csc_eff_t  # combine product + eff

        active_t = active[t]
        # monthly revenue
        rev_mkt = active_t * arpu_t / 12
        total_rev[t] = rev_mkt.sum()
        # acquisition cost this month
        new_tot = new_clients[t].sum()
        cost_acq = (new_clients[t] * cac_t).sum()
        total_acq[t] = cost_acq
        # COS
        total_cos[t] = cos_pct * total_rev[t]
        # Servicing cost
        total_serv[t] = (active_t * csc_t / 12).sum()

        # G&A: HQ + markets + products
        # HQ G&A
        if t>0:
            rev_growth = (total_rev[t] - total_rev[t-1]) / max(total_rev[t-1], 1)
            gna_month *= (1 + p['gna_growth_share'] * rev_growth)
        cost_gna_markets = (base_df['Ongoing G&A annual'].sum() + new_df.filter(like='Ongoing G&A Y').sum(axis=1).sum() + prod_df.filter(like='Ongoing G&A Y').sum(axis=1).sum()) / 12
        total_gna[t] = gna_month + cost_gna_markets

        # Tech capacity growth
        if t>0:
            rev_growth = (total_rev[t] - total_rev[t-1]) / max(total_rev[t-1], 1)
            tech_avail[t] = tech_avail[t-1] * (1 + p['tech_growth_share'] * rev_growth)
        # Tech requirement: HQ + projects
        req = tech_avail[t] * hq_share
        # add prep & maintain usages
        # base markets tech
        req += (base_df['Tech Units/mo'].sum())
        # new markets: prep & maintain
        for i in range(m1):
            row = new_df.loc[i]
            start_idx = (pd.to_datetime(row['Start Date']).year - START_DATE.year)*12 + (pd.to_datetime(row['Start Date']).month-1)
            prep = int(row['Prep Duration (mo)'])
            if start_idx <= t < start_idx+prep:
                req += row['Prep Tech/mo']
            # maintenance years
            for yi in range(5):
                s = start_idx + prep + yi*12
                e = s + 12
                col = f'Maint Tech Y{yi+1}'
                if s <= t < min(e, n):
                    req += row[col]
        # products: prep & maintain
        for j in range(len(prod_df)):
            row = prod_df.loc[j]
            start_idx = (pd.to_datetime(row['Start Date']).year - START_DATE.year)*12 + (pd.to_datetime(row['Start Date']).month-1)
            prep = int(row['Prep Duration (mo)'])
            if start_idx <= t < start_idx+prep:
                req += row['Prep Tech/mo']
            for yi in range(5):
                s = start_idx + prep + yi*12
                e = s + 12
                col = f'Maint Tech Y{yi+1}'
                if s <= t < min(e, n):
                    req += row[col]
        tech_req[t] = req

        # Tech cost
        total_tech_cost = req * p['tech_unit_cost_start'] * tech_cost_eff_t

        # Cash update
        total_cost = total_cos[t] + total_gna[t] + total_acq[t] + total_serv[t] + total_tech_cost
        if t>0:
            cash[t] = cash[t-1] + total_rev[t] - total_cost
        # metrics: CAC = cost_acq / new_tot, ARPU avg, churn avg, NRR
        cac_series[t] = cost_acq / max(new_tot, 1)
        arpu_series[t] = total_rev[t] / max(active_t.sum(), 1) * 12
        churn_y1_series[t] = (churn_y1_eff * active_t).sum() / max(active_t.sum(),1)
        churn_post_series[t] = (churn_post_eff * active_t).sum() / max(active_t.sum(),1)
        # NRR monthly
        if t>0:
            retained_rev = total_rev[t] - (new_tot * arpu_series[t] / 12)
            nrr_series[t] = retained_rev / max(total_rev[t-1],1)
        else:
            nrr_series[t] = 1

    # Build DataFrames/Series
    rev_by_mkt = pd.DataFrame(active * (base_arpu * adopt_mult_arpu) / 12, index=dates, columns=markets)
    rev_tot = pd.Series(total_rev, index=dates)
    costs_df = pd.DataFrame({
        'CoS': total_cos,
        'G&A': total_gna,
        'Acquisition': total_acq,
        'Servicing': total_serv,
        'Tech': tech_req * p['tech_unit_cost_start'] * tech_cost_eff
    }, index=dates)
    tech_df = pd.DataFrame({
        'available': tech_avail,
        'required': tech_req
    }, index=dates)
    cash_s = pd.Series(cash, index=dates)
    metrics_df = pd.DataFrame({
        'CAC': cac_series,
        'ARPU': arpu_series,
        'NRR': nrr_series,
        'Churn Y1': churn_y1_series,
        'Churn Post Y1': churn_post_series,
        'CSC (annual per client avg)': (total_serv*12)/active.sum(axis=1)
    }, index=dates)

    return {
        'dates': dates,
        'revenue_by_market': rev_by_mkt,
        'revenue': rev_tot,
        'costs_breakdown': costs_df,
        'tech': tech_df,
        'cash': cash_s,
        'metrics': metrics_df
    }

# ─────────────────────────────────────────────────────────────────────────────
# Run & Display
# ─────────────────────────────────────────────────────────────────────────────
if st.button('Run Simulation'):
    with st.spinner('Simulating...'):
        results = simulate(current_inputs)
    st.success('Simulation complete!')

    # Tech capacity warning
    if (results['tech']['required'] > results['tech']['available']).any():
        st.error('⚠️ Tech bandwidth exceeded in some months')

    # Revenue by market
    st.subheader('Revenue by Market')
    all_markets = results['revenue_by_market'].columns.tolist()
    sel = st.multiselect('Select markets', all_markets, default=all_markets)
    st.line_chart(results['revenue_by_market'][sel])

    # Total Revenue & Cash
    st.subheader('Total Revenue & Cash Balance')
    st.line_chart(pd.DataFrame({
        'Revenue': results['revenue'],
        'Cash': results['cash']
    }))

    # Costs Breakdown
    st.subheader('Monthly Costs Breakdown')
    st.area_chart(results['costs_breakdown'])

    # Tech capacity vs requirement
    st.subheader('Tech Bandwidth')
    st.line_chart(results['tech'])

    # Key Metrics
    st.subheader('Key Metrics')
    st.dataframe(results['metrics'])

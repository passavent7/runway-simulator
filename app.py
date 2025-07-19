import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Default Parameters ---
st.title("Startup Runway & Project Simulator")

st.sidebar.header("💰 Core Business Inputs")
cash_on_hand = st.sidebar.number_input("Initial Cash on Hand ($M)", value=50.0, step=1.0)
burn_rate = st.sidebar.number_input("Core Monthly Burn ($M)", value=1.3, step=0.1)
arr = st.sidebar.number_input("Current ARR ($M)", value=70.0, step=1.0)
core_growth_rate = st.sidebar.slider("Core ARR Growth Rate (YoY %)", 0, 100, 30)
tech_bandwidth_total = 100  # assume 100 units = 100%
tech_reserved_core = 50

st.sidebar.header("📦 Project Parameters")
project_types = {
    "Small Bet": dict(cost=0.12, duration=6, prob=0.5, payoff=0.5, tech=5, growth=50),
    "Medium Bet": dict(cost=0.24, duration=12, prob=0.5, payoff=1.0, tech=10, growth=50),
    "Big Bet": dict(cost=0.48, duration=24, prob=0.5, payoff=2.0, tech=20, growth=50)
}

project_plan = {}

for p_type in project_types:
    n = st.sidebar.number_input(f"# {p_type}s", 0, 10, 1, key=p_type)
    project_plan[p_type] = n

st.sidebar.header("🛠 Simulation Settings")
months = st.sidebar.slider("Simulation Duration (months)", 6, 60, 36)

def simulate():
    monthly_cash = []
    monthly_revenue = []
    monthly_burn = []
    tech_used = []
    
    cash = cash_on_hand
    revenue = arr / 12
    
    # Track project states
    projects = []
    for p_type, count in project_plan.items():
        for i in range(count):
            p = project_types[p_type].copy()
            p['type'] = p_type
            p['start_month'] = 0
            p['active'] = True
            p['month'] = 0
            projects.append(p)

    for month in range(months):
        burn = burn_rate
        tech_this_month = tech_reserved_core
        rev_this_month = revenue * ((1 + core_growth_rate/100) ** (month/12))

        # project loop
        for p in projects:
            if not p['active']:
                continue
            if p['month'] < p['duration']:
                burn += p['cost']
                tech_this_month += p['tech']
                p['month'] += 1
            elif p['month'] == p['duration']:
                if p['prob'] >= 1.0 or st.sidebar.checkbox(f"Force Success: {p['type']}", False):
                    rev_this_month += p['payoff'] / 12 * ((1 + p['growth']/100) ** (month/12))
                p['active'] = False

        if tech_this_month > tech_bandwidth_total:
            st.error(f"⚠️ Month {month+1}: Tech bandwidth exceeded ({tech_this_month}%)")

        cash -= burn
        if cash < 0:
            st.warning(f"💸 Cash out in month {month+1}")
            break

        monthly_cash.append(cash)
        monthly_burn.append(burn)
        monthly_revenue.append(rev_this_month)
        tech_used.append(tech_this_month)

    return monthly_cash, monthly_burn, monthly_revenue, tech_used

# --- Run Simulation ---
if st.button("Run Simulation"):
    cash, burn, rev, tech = simulate()
    months_range = list(range(1, len(cash)+1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months_range, y=cash, mode='lines+markers', name='Cash Remaining ($M)'))
    fig.add_trace(go.Scatter(x=months_range, y=burn, mode='lines', name='Monthly Burn ($M)'))
    fig.add_trace(go.Scatter(x=months_range, y=rev, mode='lines', name='Monthly Revenue ($M)'))
    st.plotly_chart(fig, use_container_width=True)

    export_df = pd.DataFrame({
        'Month': months_range,
        'Cash Remaining ($M)': cash,
        'Monthly Burn ($M)': burn,
        'Monthly Revenue ($M)': rev,
        'Tech Bandwidth Used (%)': tech[:len(months_range)]
    })
    
    st.download_button("📤 Export Scenario as CSV", export_df.to_csv(index=False), file_name="runway_simulation.csv")

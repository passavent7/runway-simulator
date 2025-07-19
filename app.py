import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

# --- App Title ---
st.title("📊 Startup Runway & Project Simulator")

# --- Sidebar: Core Inputs ---
st.sidebar.header("💰 Core Business Inputs")
cash_on_hand = st.sidebar.number_input("Initial Cash on Hand ($M)", value=50.0, step=1.0)
burn_rate = st.sidebar.number_input("Core Monthly Burn ($M)", value=1.3, step=0.1)
arr = st.sidebar.number_input("Current ARR ($M)", value=70.0, step=1.0)
core_growth_rate = st.sidebar.slider("Core ARR Growth Rate (YoY %)", 0, 100, 30)
tech_bandwidth_total = 100
tech_reserved_core = 50
months = st.sidebar.slider("Simulation Duration (months)", 6, 60, 36)

# --- Default Project Types ---
def get_project_types():
    return {
        "Small Bet": dict(cost=0.12, duration=6, prob=0.5, payoff=0.5, tech=5, growth=50),
        "Medium Bet": dict(cost=0.24, duration=12, prob=0.5, payoff=1.0, tech=10, growth=50),
        "Big Bet": dict(cost=0.48, duration=24, prob=0.5, payoff=2.0, tech=20, growth=50)
    }

project_types = get_project_types()

# --- Scenario Save/Load ---
scenario_json = st.sidebar.text_area("📥 Paste Scenario JSON to Load (optional)")

project_df = pd.DataFrame(columns=["Project Name", "Type", "Start Month"])

if scenario_json:
    try:
        scenario = json.loads(scenario_json)
        project_df = pd.DataFrame(scenario['projects'])
        cash_on_hand = scenario['cash']
        burn_rate = scenario['burn']
        arr = scenario['arr']
        core_growth_rate = scenario['growth']
        months = scenario['months']
        st.success("Scenario loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load scenario: {e}")

# --- Project Planning Table ---
st.subheader("🧩 Project Planning Table")
def default_projects():
    return pd.DataFrame([
        {"Project Name": "Project A", "Type": "Small Bet", "Start Month": 0},
        {"Project Name": "Project B", "Type": "Medium Bet", "Start Month": 3},
        {"Project Name": "Project C", "Type": "Big Bet", "Start Month": 6},
    ])

if project_df.empty:
    project_df = default_projects()

edited_df = st.data_editor(
    project_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={"Type": st.column_config.SelectboxColumn(options=list(project_types.keys()))}
)

# --- Save Scenario ---
if st.button("💾 Download Scenario as JSON"):
    scenario = {
        "cash": cash_on_hand,
        "burn": burn_rate,
        "arr": arr,
        "growth": core_growth_rate,
        "months": months,
        "projects": edited_df.to_dict(orient="records")
    }
    st.download_button("📥 Download JSON", json.dumps(scenario, indent=2), file_name="scenario.json")

# --- Simulation Logic ---
def simulate():
    monthly_cash = []
    monthly_revenue = []
    monthly_burn = []
    tech_used = []
    
    cash = cash_on_hand
    revenue = arr / 12
    
    project_instances = []
    for _, row in edited_df.iterrows():
        p_type = project_types[row['Type']].copy()
        p_type['name'] = row['Project Name']
        p_type['type'] = row['Type']
        p_type['start_month'] = int(row['Start Month'])
        p_type['active'] = True
        p_type['month'] = 0
        project_instances.append(p_type)

    for month in range(months):
        burn = burn_rate
        tech_this_month = tech_reserved_core
        rev_this_month = revenue * ((1 + core_growth_rate/100) ** (month/12))

        for p in project_instances:
            if not p['active'] or month < p['start_month']:
                continue
            if p['month'] < p['duration']:
                burn += p['cost']
                tech_this_month += p['tech']
                p['month'] += 1
            elif p['month'] == p['duration']:
                if p['prob'] >= 1.0:
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
if st.button("▶️ Run Simulation"):
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

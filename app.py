import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Startup Runway Simulator", layout="wide")
st.title("🚀 Startup Runway Simulator")

# --- Core Parameters ---
st.sidebar.header("Core Settings")
cash_on_hand = st.sidebar.number_input("Cash on hand ($M)", value=50.0, step=1.0) * 1e6
core_burn = st.sidebar.number_input("Core monthly burn ($M)", value=1.3, step=0.1) * 1e6
core_arr = st.sidebar.number_input("Core ARR ($M)", value=70.0, step=1.0) * 1e6
core_growth = st.sidebar.slider("Core ARR Growth (YoY %)", min_value=0, max_value=100, value=30)
tech_capacity = st.sidebar.slider("Total Tech Bandwidth (%)", min_value=0, max_value=100, value=100)
lead_capacity = st.sidebar.slider("Total Leadership Bandwidth (%)", min_value=0, max_value=100, value=100)

# --- Bet Definitions ---
st.sidebar.header("Bet Definitions")
project_types = {
    "Small Bet": {"cost": 120_000, "duration": 6, "success_prob": 0.5, "payoff": 500_000, "tech": 5, "lead": 5, "growth": 50},
    "Medium Bet": {"cost": 240_000, "duration": 6, "success_prob": 0.5, "payoff": 1_000_000, "tech": 10, "lead": 10, "growth": 50},
    "Big Bet": {"cost": 480_000, "duration": 6, "success_prob": 0.5, "payoff": 2_000_000, "tech": 20, "lead": 20, "growth": 50},
}

# --- Scenario Load ---
with st.sidebar.expander("Load Scenario JSON"):
    scenario_json = st.text_area("Paste scenario JSON here")
    if st.button("Load Scenario"):
        try:
            parsed = json.loads(scenario_json)
            cash_on_hand = parsed['cash']
            core_burn = parsed['burn']
            core_arr = parsed['arr']
            core_growth = parsed['core_growth']
            tech_capacity = parsed['tech_capacity']
            lead_capacity = parsed['lead_capacity']
            st.success("Scenario loaded!")
        except:
            st.error("Invalid JSON")

# --- Editable Project Table ---
st.subheader("📋 Project Planning")
def default_df():
    return pd.DataFrame([
        {"Project": "New Market A", "Type": "Small Bet", "Start Month": 1},
        {"Project": "New Product B", "Type": "Medium Bet", "Start Month": 3},
    ])

project_df = st.session_state.get("project_df", default_df())

edited_df = st.data_editor(
    project_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Type": st.column_config.SelectboxColumn("Type", options=list(project_types.keys())),
        "Start Month": st.column_config.NumberColumn("Start Month", min_value=0, step=1),
    }
)

st.session_state["project_df"] = edited_df

# --- Run Simulation ---
def simulate():
    cash = []
    burn = []
    revenue = []
    tech_used = []
    lead_used = []

    projects = []
    for _, row in edited_df.iterrows():
        p_type_name = row['Type']
        if p_type_name not in project_types:
            st.warning(f"Skipping unknown project type: {p_type_name}")
            continue

        p_type = project_types[p_type_name].copy()
        projects.append({**p_type, "start": int(row['Start Month'])})

    max_months = 60
    current_cash = cash_on_hand
    current_arr = core_arr

    for month in range(max_months):
        monthly_burn = core_burn
        tech_this_month = 0
        lead_this_month = 0
        cash_in = 0

        for p in projects:
            if p['start'] <= month < p['start'] + p['duration']:
                monthly_burn += p['cost']
                tech_this_month += p['tech']
                lead_this_month += p['lead']

            if month == p['start'] + p['duration']:
                cash_in += p['payoff'] * p['success_prob']

        current_cash = current_cash - monthly_burn + (cash_in + current_arr / 12)
        current_arr *= (1 + core_growth / 12 / 100)

        cash.append(current_cash)
        burn.append(monthly_burn)
        revenue.append(current_arr / 12)
        tech_used.append(tech_this_month)
        lead_used.append(lead_this_month)

    return cash, burn, revenue, tech_used, lead_used

cash, burn, rev, tech_used, lead_used = simulate()

# --- Charts ---
st.subheader("📊 Financial Projections")
st.line_chart({"Cash Balance": cash, "Burn": burn, "Revenue": rev})
st.line_chart({"Tech Used (%)": tech_used, "Leadership Used (%)": lead_used})

# --- Save Scenario ---
with st.sidebar.expander("Save Scenario"):
    scenario = {
        "cash": cash_on_hand,
        "burn": core_burn,
        "arr": core_arr,
        "core_growth": core_growth,
        "tech_capacity": tech_capacity,
        "lead_capacity": lead_capacity
    }
    st.code(json.dumps(scenario, indent=2), language="json")

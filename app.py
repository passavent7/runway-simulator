import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

# -----------------------
# Helper functions
# -----------------------

def annual_to_monthly_rate(annual_rate):
    """Convert an annual rate (e.g. 0.10) to equivalent monthly rate."""
    return (1 + annual_rate) ** (1/12) - 1


def load_or_default(csv_uploader, columns):
    """
    Read uploaded CSV or return empty DataFrame with specified columns.
    """
    if csv_uploader is not None:
        df = pd.read_csv(csv_uploader, parse_dates=["start_date"])
        return df
    else:
        return pd.DataFrame(columns=columns)

# -----------------------
# Streamlit App
# -----------------------

def main():
    st.title("Neobank Financial Simulator")

    # How it works
    if st.sidebar.button("How it works"):
        st.header("How It Works & Limitations")
        st.markdown(
            """
            **Overview**
            - Projects 72 months of monthly revenue, costs, and cash runway for a Series C neobank.
            - Models existing markets plus customizable new-market, new-product, and efficiency projects.

            **Inputs**
            1. Base metrics: ARR, profit margin, OPEX, cash on hand.
            2. Unit economics: ARPU, CAC, NRR, and churn assumptions.
            3. Existing markets: enter parameters like launch date, prep costs, unit metrics, and client additions.
            4. Projects: tables for new markets, products, and efficiency initiatives (editable directly or via CSV).

            **Simulation**
            - Derives starting client count = ARR / ARPU.
            - Applies monthly churn (Yr1 vs Yr>1) and growth per market.
            - Calculates revenue = clients * ARPU/12, gross profit = revenue * margin.
            - Calculates burn = OPEX/12 – gross profit + project costs.
            - Tracks cash balance month by month until depletion.

            **Limitations**
            - Project impacts are placeholders; you’ll need to add logic for ongoing costs and revenue from projects.
            - Assumes linear prep-cost allocation and flat unit economics outside project scope.
            - Does not model CAPEX or non-operating income/expenses.

            """
        )
        return

    st.markdown(
        "Simplify scenario planning by editing inputs in the sidebar and clicking **Run Simulation**."
    )

    # Sidebar inputs
    with st.sidebar:
        st.header("1. Simulation Settings")
        today = date.today()
        start_date = st.date_input("Start date", today)
        months = st.slider("Months to simulate", 1, 120, 72)

        st.markdown("---")
        st.header("2. Base Metrics & Unit Economics")
        arr = st.number_input("Current ARR ($)", 0.0, 1e9, 70_000_000.0, step=1_000_000.0)
        profit_margin = st.slider("Gross profit margin (%)", 0.0, 100.0, 60.0) / 100
        opex = st.number_input("Current OPEX ($/year)", 0.0, 1e9, 60_000_000.0, step=1_000_000.0)
        cash0 = st.number_input("Current cash ($)", 0.0, 1e9, 70_000_000.0, step=1_000_000.0)
        arpu = st.number_input("ARPU ($/year)", 0.0, 100_000.0, 1500.0)
        cac = st.number_input("CAC ($)", 0.0, 100_000.0, 500.0)
        nrr = st.slider("NRR (%)", 0.0, 300.0, 100.0) / 100
        churn_first = st.slider("Churn Yr1 (%)", 0.0, 100.0, 40.0) / 100
        churn_later = st.slider("Churn Yr>1 (%)", 0.0, 100.0, 5.0) / 100

        st.markdown("---")
        st.header("3. Existing Markets")
        existing_markets_template = pd.DataFrame({
            "name": ["Singapore", "Hong Kong"],
            "start_date": [today, today],
            "prep_months": [0, 0],
            "tech_prep_cost": [0.0, 0.0],
            "g_and_a_prep_cost": [0.0, 0.0],
            "cac": [cac, cac],
            "arpu": [arpu, arpu],
            "nrr": [nrr, nrr],
            "new_clients_y1": [0, 0],
            "new_clients_y2": [0, 0],
            "new_clients_y3": [0, 0],
            "new_clients_y4": [0, 0],
            "new_clients_y5": [0, 0],
            "cust_serv_cost": [0.0, 0.0],
            "ongoing_tech_cost": [0.0, 0.0],
            "ongoing_g_and_a": [0.0, 0.0],
        })
        with st.expander("Edit existing-market parameters"):
            existing_markets = st.data_editor(
                existing_markets_template,
                use_container_width=True,
                key="existing_markets"
            )

        st.markdown("---")
        st.header("4. Projects")
        # Define columns
        market_cols = [
            "name","start_date","prep_months","tech_prep_cost","g_and_a_prep_cost",
            "cac","arpu","nrr","new_clients_y1","new_clients_y2","new_clients_y3",
            "new_clients_y4","new_clients_y5","cust_serv_cost","ongoing_tech_cost","ongoing_g_and_a"
        ]
        product_cols = [
            "name","start_date","prep_months","tech_prep_cost","g_and_a_prep_cost",
            "cac_mul","adoption_y1","adoption_y2","adoption_y3","adoption_y4","adoption_y5",
            "arpu_mul","nrr_mul","csc_mul",
            "ongoing_tech_y1","ongoing_tech_y2","ongoing_tech_y3","ongoing_tech_y4","ongoing_tech_y5",
            "ongoing_g_and_a_y1","ongoing_g_and_a_y2","ongoing_g_and_a_y3",
            "ongoing_g_and_a_y4","ongoing_g_and_a_y5"
        ]
        eff_cols = [
            "name","start_date","prep_months","tech_prep_cost","g_and_a_prep_cost",
            "cac_mul","csc_mul",
            "ongoing_tech_y1","ongoing_tech_y2","ongoing_tech_y3",
            "ongoing_tech_y4","ongoing_tech_y5",
            "ongoing_g_and_a_y1","ongoing_g_and_a_y2","ongoing_g_and_a_y3",
            "ongoing_g_and_a_y4","ongoing_g_and_a_y5"
        ]

        # New Markets
        st.subheader("New-Market Launches")
        new_markets_csv = st.file_uploader("Upload CSV for new markets", type="csv")
        new_markets_df = load_or_default(new_markets_csv, market_cols)
        new_markets = st.data_editor(new_markets_df, use_container_width=True, key="new_markets_editor")

        # New Products
        st.subheader("New-Product Launches")
        new_products_csv = st.file_uploader("Upload CSV for new products", type="csv")
        new_products_df = load_or_default(new_products_csv, product_cols)
        new_products = st.data_editor(new_products_df, use_container_width=True, key="new_products_editor")

        # Efficiency Projects
        st.subheader("Efficiency Initiatives")
        eff_csv = st.file_uploader("Upload CSV for efficiency projects", type="csv")
        eff_projects_df = load_or_default(eff_csv, eff_cols)
        eff_projects = st.data_editor(eff_projects_df, use_container_width=True, key="eff_projects_editor")

        st.markdown("---")
        run_button = st.button("Run Simulation")

    if run_button:
        # Simulation logic placeholder
        st.success("Simulation logic to be implemented with project impacts.")

if __name__ == "__main__":
    main()

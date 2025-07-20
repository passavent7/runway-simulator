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
    st.markdown(
        """
        Simulate 72 months of revenue, costs, and runway for a Series C neobank, including projects for new markets,
        new products, and efficiency initiatives. Use the sidebar to configure base metrics and upload project CSVs.
        """
    )

    # Sidebar sections
    with st.sidebar:
        st.header("1. Simulation Settings")
        today = date.today()
        start_date = st.date_input("Start date", today)
        months = st.slider("Months to simulate", min_value=1, max_value=120, value=72)

        st.markdown("---")
        st.header("2. Base Metrics")
        arr = st.number_input("Current ARR ($)", min_value=0.0, value=70_000_000.0, step=1_000_000.0)
        profit_margin = st.slider("Gross profit margin (%)", 0.0, 100.0, 60.0) / 100
        opex = st.number_input("Current OPEX ($)", min_value=0.0, value=60_000_000.0, step=1_000_000.0)
        cash0 = st.number_input("Current cash ($)", min_value=0.0, value=70_000_000.0, step=1_000_000.0)

        st.markdown("---")
        st.header("3. Unit Economics")
        arpu = st.number_input("ARPU ($/year)", min_value=0.0, value=1500.0)
        cac = st.number_input("CAC ($)", min_value=0.0, value=500.0)
        nrr = st.slider("NRR (%)", 0.0, 300.0, 100.0) / 100
        churn_first = st.slider("Churn Yr1 (%)", 0.0, 100.0, 40.0) / 100
        churn_later = st.slider("Churn Yr>1 (%)", 0.0, 100.0, 5.0) / 100

        st.markdown("---")
        st.header("4. Market Breakdown")
        market_template = pd.DataFrame({
            "market": ["Singapore", "Hong Kong"],
            "rev_share_pct": [90, 10],
            "growth_y1_pct": [10, 100],
            "growth_y2_pct": [10, 100],
            "growth_y3_pct": [10, 50],
            "growth_y4_pct": [10, 50],
            "growth_y5_pct": [10, 50],
        })
        st.markdown("Edit your market parameters below:")
        with st.expander("Market table"):
            market_df = st.experimental_data_editor(
                market_template,
                use_container_width=True,
                key="market_editor"
            )

        st.markdown("---")
        st.header("5. Project CSV Uploads")
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

        new_markets = load_or_default(st.file_uploader("New markets CSV", type="csv"), market_cols)
        new_products = load_or_default(st.file_uploader("New products CSV", type="csv"), product_cols)
        eff_projects = load_or_default(st.file_uploader("Efficiency projects CSV", type="csv"), eff_cols)

        run_button = st.button("Run Simulation")

    # Run simulation when requested
    if run_button:
        # Build date index
        dates = pd.date_range(start_date, periods=months, freq='M')
        df = pd.DataFrame(index=dates)

        # Initialize series
        df['clients'] = np.nan
        df['revenue'] = np.nan
        df['gross_profit'] = np.nan
        df['burn'] = np.nan
        df['cash'] = np.nan
        df['arpu'] = arpu
        df['cac'] = cac
        df['nrr'] = nrr
        df['csc'] = 0.0

        base_clients = arr / arpu

        for i, current_date in enumerate(dates):
            # Determine churn
            monthly_churn = churn_first/12 if i < 12 else churn_later/12
            if i == 0:
                clients = base_clients
            else:
                clients = prev_clients * (1 - monthly_churn)
                # Apply market growth
                growth_rate = 0
                for _, m in market_df.iterrows():
                    share = m.rev_share_pct/100
                    if i < 12:
                        ann = m.growth_y1_pct/100
                    elif i < 24:
                        ann = m.growth_y2_pct/100
                    elif i < 36:
                        ann = m.growth_y3_pct/100
                    elif i < 48:
                        ann = m.growth_y4_pct/100
                    else:
                        ann = m.growth_y5_pct/100
                    growth_rate += share * annual_to_monthly_rate(ann)
                clients *= (1 + growth_rate)

            prev_clients = clients
            df.at[current_date, 'clients'] = clients

            # Revenue & profit
            rev = clients * (arpu/12)
            profit = rev * profit_margin

            # Base burn = OPEX/12 - gross profit
            base_burn = opex/12 - profit

            # Placeholder: add project impacts
            project_burn = 0
            project_rev = 0

            total_burn = base_burn + project_burn
            df.at[current_date, 'revenue'] = rev
            df.at[current_date, 'gross_profit'] = profit
            df.at[current_date, 'burn'] = total_burn

            # Cash runway
            if i == 0:
                df.at[current_date, 'cash'] = cash0 - total_burn
            else:
                df.at[current_date, 'cash'] = df.iloc[i-1].cash - total_burn

        st.header("Simulation Results")
        st.subheader("Top‑Line & Profit")
        st.line_chart(df[['revenue','gross_profit']])
        st.subheader("Burn & Cash")
        st.line_chart(df[['burn','cash']])
        st.subheader("Blended Metrics")
        st.line_chart(df[['arpu','cac','nrr','csc']])

        st.download_button("Download simulation data as CSV",
                            df.to_csv().encode('utf-8'),
                            file_name="simulation.csv")

if __name__ == "__main__":
    main()

Initial Prompt: 

I am the CFO of a series C business neobank. I want to build a simulator of my company's financials to understand what my revenue, costs and runway will look like based on different scenarios. 
Simulator to be hosted on a webpage with streamlit, coded in Python.
 
We are contemplating a few projects to grow faster. Typically 3 kinds of projects:
- Launching our core product (a "Business Account" for SMEs) in new markets. To be able to serve more SMEs.
- Launching new Products (in all markets where we operate). Always focused on SMEs -> Goal is to lower our CAC, grow ARPU and decrease churn across markets where we'll operate.
- Efficiency increase projects - to lower our CAC, lower our CSC (Customer Servicing Costs), and lower our cost per Tech bandwidth unit.

Each of these 3 types of projects has different types of parameters.
For new markets: 
- Market name
- Project start date
- Time needed to prepare the launch (in months)
- Tech Bandwidth units to prepare launch (per month)
- G&A Cost for launch preparation (per month)
- CAC ($)
- ARPU ($)
- Churn Year 1
- Churn post Year 1 
- New clients in Year 1, 2, 3, 4, 5 (linear progression each month):
- CSC ($ per client per year)
- Tech Bandwidth units for maintenance (per month) - Y1, Y2, Y3, Y4, Y5
- Ongoing G&A cost ($ per year) - Y1, Y2, Y3, Y4, Y5

For new Products:
- Product name
- Project start date
- Time needed to prepare the launch (in months)
- Tech Bandwidth units during launch preparation (per month)
- G&A cost for launch preparation (per month)
- CAC multiplier (as new products are expected to reduce our overall CAC - applies to all clients)
- ARPU multiplier (as new products are expected to increase our overall ARPU - applies only to clients using the product)
- Churn Year 1 multiplier (as new products are expected to reduce it - applies only to clients using the product)
- Churn post Year 1 multiplier (as new products are expected to reduce it - applies only to clients using the product)
- CSC multiplier (as new products are expected to increase our overall CSC - applies only to clients using the product)
- Adoption Y1, Y2, Y3, Y4, Y5 (percentage of existing client base, spread evenly across markets, linear progression each month)
- Tech Bandwidth units for maintenance (units per month) - Y1, Y2, Y3, Y4, Y5
- Ongoing G&A cost ($ per year) - Y1, Y2, Y3, Y4, Y5

For new Efficiency Increase Projects:
- Project name:
- Project start date:
- Duration (in months):
- Tech Bandwidth units required per month:
- CAC multiplier (as Efficiency increase projects reduce our overall CAC)
- CSC multiplier (as Efficiency projects reduce overall CSC; applies to our entire client base)
- Tech bandwidth unit multiplier (as Efficiency projects reduce the cost per Tech Bandwidth unit)

In the simulator, for each of the 3 types of project, I want to be able to input a list of projects in a table. Table columns should be the required parameters for those projects. Choose self explanatory names for columns of the table - but add explainers on each, to help clear any possible doubt.
After the simulation is run, it should be possible to edit the parameters of any project easily, and re-run the simulation.

I the output of the simulator, I want to see, on different graphs, the month on month evolution of the following for next 5 years:
- Revenue
- Costs 
- Total Tech bandwidth units available
- Total Tech bandwidth units required
- Cash in the bank 
- All key metrics: CAC, ARPU, NRR, CSC, Churn year 1, Churn post year 1 (especially as the launch of new products and efficiency projects impact them)

For Revenue and Cost, have views showing contribution of each Market and Product.
For Costs, have a view showing breakdown by type of cost (CoS, G&A, Tech, Acquisition, Servicing)
For all graphs, show aggregate over all markets - but the interface should allow to select only some markets. For costs breakdown, allocate 'Headquarters' costs to a standalone, non Revenue generating 'Headquarters' market 

Base data for simulations:

- Cash in the bank: $70 million
- Cost of Sales (we assume it is always the same for simplicity): 40% of Revenue
- Headquarter G&A costs: $15 million per year. 
- Headquarter G&A cost growth rate (as a percentage of Revenue growth rate, to calculate for each month): 50%
- Available Tech Bandwidth units (per month): 50 
- Tech bandwidth unit cost: $30,000  
- Tech bandwidth required by Headquarter (bandwidth required for Product infra maintenance and enhancements; expressed as a percentage of total available Tech Bandwidth units, to calculate for each month): 50% 
- Total available Tech bandwidth units growth rate (expressed as a percentage of total Revenue's growth rate, to calculate for each month):  100%

Key numbers in existing Markets: 
Singapore:
- Existing clients: 20000
- CAC: $500
- ARPU: $1500
- NRR: 100%
- Clients churn Year 1: 30% 
- Clients churn Post Year 1: 10%
- New clients in Year 1, 2, 3, 4, 5: 6000, 6000, 6000, 6000, 6000
- CSC: $200
- Tech Bandwidth units required per month: 4
- Ongoing G&A cost: $2M per year, every year

Hong-Kong:
- Existing clients: 1000
- CAC: $500
- ARPU: $1500
- Clients churn Year 1: 30% 
- Clients churn Post Year 1: 5%
- New clients in Year 1, 2, 3, 4, 5: 1000, 3000, 6000, 10000, 15000
- CSC: $200
- Tech Bandwidth units required per month: 4
- Ongoing G&A cost: $2M per year, every year

All those base numbers should also be editable in the simulator, via tables.

Also Prefill the new markets table with this market: 
- Market name: United States
- Project start date: August 2025
- Time needed to prepare the launch (in months): 0
- Tech Bandwidth units to prepare launch (per month): 0
- G&A Cost for launch preparation (per month): 0
- CAC ($): 1000
- ARPU ($): 2000
- Clients churn Year 1: 30% 
- Clients churn Post Year 1: 10%
- New clients in Year 1, 2, 3, 4, 5: 1000, 3000, 10000, 25000, 50000
- CSC ($ per client per year): 300
- Tech Bandwidth units for maintenance (per month) - Y1, Y2, Y3, Y4, Y5: 3, 4, 5, 6, 7
- Ongoing G&A cost ($ per year) - Y1, Y2, Y3, Y4, Y5: 1000000, 2000000, 3000000, 4000000, 5000000

Also prefill the new Products table with this Product: 
- Product name: AI Accounting
- Project start date: July 2025
- Time needed to prepare the launch (in months): 6
- Tech Bandwidth units during launch preparation (per month): 2
- G&A cost for launch preparation (per month): $20000
- CAC multiplier: 0.95
- ARPU multiplier: 1.3
- Churn Year 1 multiplier: 0.9
- Churn post Year 1 multiplier: 0.9
- CSC multiplier: 1.2
- Adoption Y1, Y2, Y3, Y4, Y5 (percentage of existing client base, spread evenly across markets, linear progression month on month): 2, 5, 10, 20, 30
- Tech Bandwidth units for maintenance Y1, Y2, Y3, Y4, Y5: 3, 3, 3, 3, 3
- Ongoing G&A cost ($ per year) - Y1, Y2, Y3, Y4, Y5: 250000, 500000, 750000, 1000000, 1500000 

For now no need to think about exports or imports in JSON/.csv, we do everything online
Start simulation from July 2025.
Calculate churn monthly from yearly numbers given in input.
Allow to save all scenarios / each time we run a simulation
Assume no other inflows/outflows aside from operating P&L and capex from projects.
Allow to go over Tech budget capacity, but pls indicate clearly at the top of the dashboard if we're above capacity

Do let me know first if anything is not clear - important to clarify everything before building!

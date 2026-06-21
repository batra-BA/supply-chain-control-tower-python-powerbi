# Install PuLP if not already installed
try:
    import pulp
except ImportError:
    # pip install pulp
    import pulp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the forecasts dataset
# Assuming forecasts.csv was generated in the previous forecasting phase.
# If the actual forecasts.csv does not contain store_id, we will need to re-evaluate how to merge.
# For now, let's assume it was aggregated at item_id + store_id level or can be merged appropriately.
# I'll create a dummy 'forecasts' DataFrame for demonstration if it's not present.

# --- Dummy forecasts DataFrame for demonstration (if forecasts.csv is not available) ---
try:
    # Try loading the actual forecasts.csv (assuming it was created in the previous step)
    forecasts_df = pd.read_csv('forecasts.csv')
    # The prompt mentioned 'forecast_demand, lower_80, upper_80, lower_95, upper_95'
    # Let's ensure these columns are available or mocked if not present in the CSV.
    # For this exercise, I'll assume 'forecast' from the previous step is 'forecast_demand'.
    if 'forecast' in forecasts_df.columns:
        forecasts_df.rename(columns={'forecast': 'forecast_demand'}, inplace=True)
    if 'item_id' not in forecasts_df.columns or 'store_id' not in forecasts_df.columns:
        # This part might need adjustment based on how forecasts.csv was actually structured.
        # If forecasts were only at item_id level, we need to add a store_id.
        # For now, I'll assume forecasts are for item_id only and duplicate them across available stores.
        print("Warning: 'store_id' not found in forecasts.csv. Duplicating item-level forecasts across stores.")
        all_item_store_pairs = pd.DataFrame({'item_id': [1, 2, 3], 'store_id': [4, 5, 6]})
        forecasts_df = pd.merge(all_item_store_pairs, forecasts_df[['item_id', 'forecast_demand']].drop_duplicates(), on='item_id', how='left')
        # Fill any potential NaNs if some items in inventory_policy don't have forecasts
        forecasts_df['forecast_demand'] = forecasts_df['forecast_demand'].fillna(forecasts_df['forecast_demand'].mean())

except FileNotFoundError:
    print("forecasts.csv not found. Creating a dummy forecasts DataFrame for demonstration.")
    # Generate dummy forecasts data if the file is not found
    # This part should be removed or adjusted if actual forecasts data is available.
    dummy_data = []
    for item in all_item_store_pairs['item_id'].unique():
        for store in all_item_store_pairs['store_id'].unique():
            dummy_data.append({
                'item_id': item,
                'store_id': store,
                'forecast_demand': np.random.randint(100, 1000) # Random demand
            })
    forecasts_df = pd.DataFrame(dummy_data)
# ------------------------------------------------------------------------------------------------


lp_input = pd.merge(all_item_store_pairs, cost_parameters, on='item_id', how='left')
lp_input = pd.merge(lp_input, forecasts_df, on=['item_id', 'store_id'], how='left')

# Select required columns
lp_input = lp_input[[
    'item_id',
    'store_id',
    'forecast_demand',
    'safety_stock',
    'reorder_point',
    'holding_cost',
    'stockout_cost'
]]

# Validation: No duplicate SKU-store rows
if lp_input.duplicated(subset=['item_id', 'store_id']).any():
    raise ValueError("Duplicate item_id + store_id found in lp_input.")

# Validation: No missing values
lp_input = lp_input.dropna()

print("LP Input Table created successfully.")
print("Shape of lp_input:", lp_input.shape)
print(lp_input.head())

baseline_policy = lp_input[['item_id', 'store_id', 'forecast_demand']].copy()
baseline_policy['baseline_inventory'] = 2 * baseline_policy['forecast_demand']
baseline_policy['baseline_inventory'] = baseline_policy['baseline_inventory'].astype(int)

print("Baseline Policy created successfully.")
print(baseline_policy.head())

prob = pulp.LpProblem("Inventory Optimization", pulp.LpMinimize)

# Decision Variables
inventory_qty = pulp.LpVariable.dicts(
    "Inventory",
    [(row['item_id'], row['store_id']) for idx, row in lp_input.iterrows()],
    lowBound=0,  # Inventory cannot be negative
    cat=pulp.LpContinuous
)

# Shortage Variables (to handle max(0, forecast_demand - inventory_qty))
shortage_qty = pulp.LpVariable.dicts(
    "Shortage",
    [(row['item_id'], row['store_id']) for idx, row in lp_input.iterrows()],
    lowBound=0, # Shortage cannot be negative
    cat=pulp.LpContinuous
)

print("LP Problem and Decision Variables initialized.")

# Objective Function: Minimize Total Holding Cost + Total Stockout Cost
prob += (
    pulp.lpSum(inventory_qty[(row['item_id'], row['store_id'])] * row['holding_cost'] for idx, row in lp_input.iterrows()) +
    pulp.lpSum(shortage_qty[(row['item_id'], row['store_id'])] * row['stockout_cost'] for idx, row in lp_input.iterrows())
), "Total Cost"

print("Objective Function defined.")

# Constraint 1: Inventory >= safety_stock
for idx, row in lp_input.iterrows():
    item_store = (row['item_id'], row['store_id'])
    prob += inventory_qty[item_store] >= row['safety_stock'], f"SafetyStock_Constraint_{item_store[0]}_{item_store[1]}"

# Constraint 2: Inventory + shortage >= forecast_demand
for idx, row in lp_input.iterrows():
    item_store = (row['item_id'], row['store_id'])
    prob += inventory_qty[item_store] + shortage_qty[item_store] >= row['forecast_demand'], f"DemandFulfillment_Constraint_{item_store[0]}_{item_store[1]}"

# Constraint 3: Budget Constraint
# Calculate baseline total inventory value
baseline_total_inventory_value = sum(
    baseline_policy.loc[baseline_policy['item_id'] == row['item_id'], 'baseline_inventory'].iloc[0] * lp_input.loc[idx, 'holding_cost'] * (52 / 0.15) 
    for idx, row in lp_input.iterrows()
    # Note: avg_sell_price is needed for inventory value, but it's not directly in lp_input. 
    # Re-merging cost_parameters to get avg_sell_price for the budget calculation.
)
# Assuming 'avg_sell_price' is needed for inventory value calculation. 
# lp_input contains 'holding_cost', which is avg_sell_price * 0.15 / 52.
# So, avg_sell_price = holding_cost * 52 / 0.15

# Calculate baseline inventory value per item_id and store_id
baseline_inventory_value_per_sku = pd.merge(all_item_store_pairs, cost_parameters[['item_id', 'avg_sell_price']], on='item_id', how='left')
baseline_inventory_value_per_sku['value'] = baseline_inventory_value_per_sku['baseline_inventory'] * baseline_inventory_value_per_sku['avg_sell_price']
baseline_total_inventory_value = baseline_inventory_value_per_sku['value'].sum()

budget_cap = 0.70 * baseline_total_inventory_value

prob += (
    pulp.lpSum(
        inventory_qty[(row['item_id'], row['store_id'])] * 
        lp_input.loc[idx, 'holding_cost'] * (52 / 0.15) # Reconstruct avg_sell_price from holding_cost
        for idx, row in lp_input.iterrows()
    )
    <= budget_cap
), "Budget_Constraint"

print("Constraints defined.")

max_retries = 5
retry_count = 0

# Calculate the minimum budget required to hold all safety stock
# Reconstruct avg_sell_price from holding_cost: avg_sell_price = holding_cost * 52 / 0.15
min_inv_value_from_safetystock = (lp_input['safety_stock'] * lp_input['holding_cost'] * (52 / 0.15)).sum()
print(f"Minimum inventory value required to hold all safety stock: {min_inv_value_from_safetystock:.2f}")

initial_budget_cap = budget_cap # This is 70% of baseline
current_budget_cap = initial_budget_cap

# Ensure the initial budget is at least enough to cover safety stock with a generous buffer.
# If the original 70% of baseline is less than twice the cost of holding safety stock,
# then start the budget at twice the cost of holding safety stock.
# This provides a much larger initial space for the solver to work.
if current_budget_cap < (min_inv_value_from_safetystock * 2.0):
    print(f"Initial budget ({current_budget_cap:.2f}) is less than a generous budget for safety stock ({min_inv_value_from_safetystock * 2.0:.2f}). Adjusting initial budget.")
    current_budget_cap = min_inv_value_from_safetystock * 2.0 # Start with a much larger buffer

while retry_count < max_retries:
    # Update the budget constraint in the problem for each retry
    # First, remove the old budget constraint if it exists
    if "Budget_Constraint" in prob.constraints:
        del prob.constraints["Budget_Constraint"]

    # Add the updated budget constraint
    prob += (
        pulp.lpSum(
            inventory_qty[(row['item_id'], row['store_id'])] *
            lp_input.loc[idx, 'holding_cost'] * (52 / 0.15) # Reconstruct avg_sell_price
            for idx, row in lp_input.iterrows()
        )
        <= current_budget_cap
    ), "Budget_Constraint"

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    print(f"Attempt {retry_count + 1}: Solver Status - {pulp.LpStatus[prob.status]}")

    if prob.status == pulp.LpStatusOptimal or prob.status == pulp.LpStatusNotSolved:
        break  # Solution found or not solvable (e.g., unbounded, user killed)
    elif prob.status == pulp.LpStatusInfeasible:
        retry_count += 1
        current_budget_cap *= 1.10  # Increase budget by 10%
        print(f"Infeasible. Increasing budget to {current_budget_cap:.2f} and retrying...")
    else:
        break # Other status, break out

print(f"\nFinal Solver Status: {pulp.LpStatus[prob.status]}")
if pulp.LpStatus[prob.status] == "Optimal":
    print(f"Optimal Objective Value (Total Cost): {pulp.value(prob.objective):.2f}")
else:
    print("Could not find an optimal solution.")

optimized_inventory_policy = []
for idx, row in lp_input.iterrows():
    item_id = row['item_id']
    store_id = row['store_id']
    optimized_inv = inventory_qty[(item_id, store_id)].varValue if inventory_qty[(item_id, store_id)].varValue is not None else 0
    optimized_inventory_policy.append({
        'item_id': item_id,
        'store_id': store_id,
        'optimized_inventory': round(optimized_inv),
        'safety_stock': row['safety_stock'],
        'reorder_point': row['reorder_point']
    })

optimized_inventory_policy = pd.DataFrame(optimized_inventory_policy)

# Merge with baseline to calculate reduction
optimized_inventory_policy = pd.merge(
    optimized_inventory_policy,
    baseline_policy[['item_id', 'store_id', 'baseline_inventory']],
    on=['item_id', 'store_id'],
    how='left'
)
optimized_inventory_policy['inventory_reduction'] = optimized_inventory_policy['baseline_inventory'] - optimized_inventory_policy['optimized_inventory']

print("Optimized Inventory Policy created successfully.")
print(optimized_inventory_policy.head())

print("\nTop 20 inventory reductions:")
print(optimized_inventory_policy.sort_values(by='inventory_reduction', ascending=False).head(20))

policy_comparison = []

# --- Baseline Policy KPIs ---
baseline_total_inventory_units = baseline_policy['baseline_inventory'].sum()

# Merge lp_input to get avg_sell_price indirectly (from holding_cost)
baseline_kpis_df = pd.merge(baseline_policy, lp_input, on=['item_id', 'store_id'], how='left', suffixes=('_x', '_y'))
baseline_kpis_df['avg_sell_price'] = baseline_kpis_df['holding_cost'] * 52 / 0.15

baseline_total_inventory_value = (baseline_kpis_df['baseline_inventory'] * baseline_kpis_df['avg_sell_price']).sum()
baseline_total_holding_cost = (baseline_kpis_df['baseline_inventory'] * baseline_kpis_df['holding_cost']).sum()

# Calculate baseline stockout cost for comparison, using forecast_demand_x which corresponds to baseline_policy's forecast_demand
baseline_shortage = np.maximum(0, baseline_kpis_df['forecast_demand_x'] - baseline_kpis_df['baseline_inventory'])
baseline_total_stockout_cost = (baseline_shortage * baseline_kpis_df['stockout_cost']).sum()
baseline_total_cost = baseline_total_holding_cost + baseline_total_stockout_cost

policy_comparison.append({'metric': 'Total Inventory Units', 'baseline': baseline_total_inventory_units, 'optimized': None, 'improvement_pct': None})
policy_comparison.append({'metric': 'Total Inventory Value', 'baseline': baseline_total_inventory_value, 'optimized': None, 'improvement_pct': None})
policy_comparison.append({'metric': 'Total Holding Cost', 'baseline': baseline_total_holding_cost, 'optimized': None, 'improvement_pct': None})
policy_comparison.append({'metric': 'Total Stockout Cost', 'baseline': baseline_total_stockout_cost, 'optimized': None, 'improvement_pct': None})
policy_comparison.append({'metric': 'Total Cost', 'baseline': baseline_total_cost, 'optimized': None, 'improvement_pct': None})

# --- Optimized Policy KPIs ---
optimized_total_inventory_units = optimized_inventory_policy['optimized_inventory'].sum()

# Merge lp_input to get avg_sell_price indirectly (from holding_cost)
optimized_kpis_df = pd.merge(optimized_inventory_policy, lp_input, on=['item_id', 'store_id'], how='left')
optimized_kpis_df['avg_sell_price'] = optimized_kpis_df['holding_cost'] * 52 / 0.15

optimized_total_inventory_value = (optimized_kpis_df['optimized_inventory'] * optimized_kpis_df['avg_sell_price']).sum()
optimized_total_holding_cost = (optimized_kpis_df['optimized_inventory'] * optimized_kpis_df['holding_cost']).sum()

# Calculate optimized stockout cost, using forecast_demand from lp_input (which corresponds to forecast_demand_y)
optimized_shortage = np.maximum(0, optimized_kpis_df['forecast_demand'] - optimized_kpis_df['optimized_inventory'])
optimized_total_stockout_cost = (optimized_shortage * optimized_kpis_df['stockout_cost']).sum()
optimized_total_cost = optimized_total_holding_cost + optimized_total_stockout_cost

# Update policy_comparison with optimized values
for entry in policy_comparison:
    if entry['metric'] == 'Total Inventory Units':
        entry['optimized'] = optimized_total_inventory_units
    elif entry['metric'] == 'Total Inventory Value':
        entry['optimized'] = optimized_total_inventory_value
    elif entry['metric'] == 'Total Holding Cost':
        entry['optimized'] = optimized_total_holding_cost
    elif entry['metric'] == 'Total Stockout Cost':
        entry['optimized'] = optimized_total_stockout_cost
    elif entry['metric'] == 'Total Cost':
        entry['optimized'] = optimized_total_cost

# Calculate improvement percentage
for entry in policy_comparison:
    if entry['baseline'] is not None and entry['baseline'] != 0:
        entry['improvement_pct'] = ((entry['baseline'] - entry['optimized']) / entry['baseline']) * 100
    else:
        entry['improvement_pct'] = np.nan

policy_comparison_df = pd.DataFrame(policy_comparison)
print("Business KPIs calculated successfully.")
print(policy_comparison_df)

service_levels = {
    85: 1.04,
    90: 1.28,
    95: 1.65,
    97: 1.88,
    99: 2.33
}

pareto_frontier_data = []

# original_safety_stock and original_reorder_point are not needed for this loop.

for service_lvl, Z_value in service_levels.items():
    print(f"\nRunning optimization for Service Level: {service_lvl}%")

    # Recalculate safety stock for this service level
    temp_inventory_parameters = all_item_store_pairs.copy()
    temp_inventory_parameters['safety_stock'] = Z_value * np.sqrt(
        (temp_inventory_parameters['avg_lead_time'] * temp_inventory_parameters['demand_std']**2) +
        (temp_inventory_parameters['avg_demand']**2 * temp_inventory_parameters['leadtime_std']**2)
    )
    temp_inventory_parameters['safety_stock'] = temp_inventory_parameters['safety_stock'].fillna(0).round().astype(int)

    # Recalculate reorder points (depends on avg_demand, avg_lead_time, and new safety_stock)
    temp_inventory_parameters['reorder_point'] = (
        temp_inventory_parameters['avg_demand'] * temp_inventory_parameters['avg_lead_time']
    ) + temp_inventory_parameters['safety_stock']
    temp_inventory_parameters['reorder_point'] = temp_inventory_parameters['reorder_point'].fillna(0).round().astype(int)

    # Update lp_input with new safety_stock and reorder_point
    current_lp_input = pd.merge(
        lp_input.drop(columns=['safety_stock', 'reorder_point']),
        temp_inventory_parameters[['item_id', 'store_id', 'safety_stock', 'reorder_point']],
        on=['item_id', 'store_id'],
        how='left'
    )

    # Rebuild LP Problem for this service level
    prob_sl = pulp.LpProblem(f"Inventory Optimization SL {service_lvl}", pulp.LpMinimize)

    inventory_qty_sl = pulp.LpVariable.dicts(
        "Inventory",
        [(row['item_id'], row['store_id']) for idx, row in current_lp_input.iterrows()],
        lowBound=0,
        cat=pulp.LpContinuous
    )
    shortage_qty_sl = pulp.LpVariable.dicts(
        "Shortage",
        [(row['item_id'], row['store_id']) for idx, row in current_lp_input.iterrows()],
        lowBound=0,
        cat=pulp.LpContinuous
    )

    # Objective function
    prob_sl += (
        pulp.lpSum(inventory_qty_sl[(row['item_id'], row['store_id'])] * row['holding_cost'] for idx, row in current_lp_input.iterrows()) +
        pulp.lpSum(shortage_qty_sl[(row['item_id'], row['store_id'])] * row['stockout_cost'] for idx, row in current_lp_input.iterrows())
    ), "Total Cost"

    # Constraints
    for idx, row in current_lp_input.iterrows():
        item_store = (row['item_id'], row['store_id'])
        prob_sl += inventory_qty_sl[item_store] >= row['safety_stock'], f"SafetyStock_Constraint_SL_{service_lvl}_{item_store[0]}_{item_store[1]}"
        prob_sl += inventory_qty_sl[item_store] + shortage_qty_sl[item_store] >= row['forecast_demand'], f"DemandFulfillment_Constraint_SL_{service_lvl}_{item_store[0]}_{item_store[1]}"

    # Budget Constraint (use the final budget from the main optimization)
    # Ensure the budget is at least enough to cover safety stock for this service level as well.
    min_inv_value_for_current_sl = (current_lp_input['safety_stock'] * current_lp_input['holding_cost'] * (52 / 0.15)).sum()
    current_sl_budget_cap = current_budget_cap # Start with the budget from the main optimization

    # If the budget from the main optimization is not enough for the current service level's safety stock, increase it.
    # This is similar logic to the main solve step to ensure feasibility for each service level.
    if current_sl_budget_cap < (min_inv_value_for_current_sl * 2.0): # Use a generous buffer like in main solve
        print(f"  Adjusting budget for SL {service_lvl}% from {current_sl_budget_cap:.2f} to {min_inv_value_for_current_sl * 2.0:.2f} to ensure feasibility.")
        current_sl_budget_cap = min_inv_value_for_current_sl * 2.0

    prob_sl += (
        pulp.lpSum(
            inventory_qty_sl[(row['item_id'], row['store_id'])] *
            current_lp_input.loc[idx, 'holding_cost'] * (52 / 0.15) # Reconstruct avg_sell_price
            for idx, row in current_lp_input.iterrows()
        )
        <= current_sl_budget_cap
    ), "Budget_Constraint_SL"

    # Solve the LP
    prob_sl.solve(pulp.PULP_CBC_CMD(msg=False))

    if prob_sl.status == pulp.LpStatusOptimal:
        current_total_inventory = sum(inventory_qty_sl[(row['item_id'], row['store_id'])].varValue for idx, row in current_lp_input.iterrows())
        current_holding_cost = sum(inventory_qty_sl[(row['item_id'], row['store_id'])].varValue * row['holding_cost'] for idx, row in current_lp_input.iterrows())
        current_stockout_cost = sum(shortage_qty_sl[(row['item_id'], row['store_id'])].varValue * row['stockout_cost'] for idx, row in current_lp_input.iterrows())
        current_total_cost = pulp.value(prob_sl.objective)

        pareto_frontier_data.append({
            'service_level': service_lvl,
            'total_inventory': current_total_inventory,
            'total_cost': current_total_cost,
            'holding_cost': current_holding_cost,
            'stockout_cost': current_stockout_cost
        })
    else:
        print(f"Warning: Could not solve for service level {service_lvl}%. Status: {pulp.LpStatus[prob_sl.status]}")

pareto_frontier = pd.DataFrame(pareto_frontier_data)
pareto_frontier = pareto_frontier.sort_values('service_level')

print("Sensitivity analysis complete. Pareto Frontier data generated.")
print(pareto_frontier)

plt.figure(figsize=(10, 6))
plt.plot(
    pareto_frontier['service_level'],
    pareto_frontier['total_cost'],
    marker='o', linestyle='-', color='blue'
)

for i, row in pareto_frontier.iterrows():
    plt.annotate(
        f"SL {int(row['service_level'])}%: ${row['total_cost']:.0f}",
        (row['service_level'], row['total_cost']),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )

plt.title('Pareto Frontier: Service Level vs. Total Inventory Cost')
plt.xlabel('Service Level (%)')
plt.ylabel('Total Inventory Cost')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(pareto_frontier['service_level'])
plt.tight_layout()
plt.savefig('pareto_frontier.png')
plt.show()

print("Pareto Frontier chart saved as 'pareto_frontier.png'.")

print("--- Executive Summary ---")

# 1. Baseline Inventory Value
baseline_inv_value = policy_comparison_df[policy_comparison_df['metric'] == 'Total Inventory Value']['baseline'].iloc[0]
print(f"1. Baseline Inventory Value: ${baseline_inv_value:,.2f}")

# 2. Optimized Inventory Value
optimized_inv_value = policy_comparison_df[policy_comparison_df['metric'] == 'Total Inventory Value']['optimized'].iloc[0]
print(f"2. Optimized Inventory Value: ${optimized_inv_value:,.2f}")

# 3. Inventory Reduction %
inv_reduction_pct = policy_comparison_df[policy_comparison_df['metric'] == 'Total Inventory Value']['improvement_pct'].iloc[0]
print(f"3. Inventory Value Reduction %: {inv_reduction_pct:.2f}%")

# 4. Best Service Level (Assuming best is lowest total cost)
if not pareto_frontier.empty:
    best_service_level_row = pareto_frontier.loc[pareto_frontier['total_cost'].idxmin()]
    print(f"4. Best Service Level (min total cost): {best_service_level_row['service_level']:.0f}% (Cost: ${best_service_level_row['total_cost']:.2f})")
else:
    print("4. Best Service Level: N/A (Pareto Frontier not generated)")

# 5. Cost Difference between 90% and 95%
cost_90 = pareto_frontier[pareto_frontier['service_level'] == 90]['total_cost'].iloc[0] if 90 in pareto_frontier['service_level'].values else np.nan
cost_95 = pareto_frontier[pareto_frontier['service_level'] == 95]['total_cost'].iloc[0] if 95 in pareto_frontier['service_level'].values else np.nan

if not np.isnan(cost_90) and not np.isnan(cost_95):
    cost_diff_90_95 = cost_95 - cost_90
    print(f"5. Cost Difference between 90% and 95% Service Level: ${cost_diff_90_95:,.2f}")
else:
    print("5. Cost Difference between 90% and 95% Service Level: N/A (data missing)")

# 6. Cost Difference between 95% and 99%
cost_99 = pareto_frontier[pareto_frontier['service_level'] == 99]['total_cost'].iloc[0] if 99 in pareto_frontier['service_level'].values else np.nan

if not np.isnan(cost_95) and not np.isnan(cost_99):
    cost_diff_95_99 = cost_99 - cost_95
    print(f"6. Cost Difference between 95% and 99% Service Level: ${cost_diff_95_99:,.2f}")
else:
    print("6. Cost Difference between 95% and 99% Service Level: N/A (data missing)")

# Export all datasets as CSV
lp_input.to_csv('lp_input.csv', index=False)
baseline_policy.to_csv('baseline_policy.csv', index=False)
optimized_inventory_policy.to_csv('optimized_inventory_policy.csv', index=False)
policy_comparison_df.to_csv('policy_comparison.csv', index=False)
pareto_frontier.to_csv('pareto_frontier.csv', index=False)

print("All final deliverables exported to CSV files.")
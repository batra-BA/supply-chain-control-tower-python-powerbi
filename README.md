Dashboard --> https://chainos-dashboard.netlify.app/
# Project Title

# 🚀 ChainOS: ABCXYZ-Driven Supply Chain Control Tower for Demand Forecasting, Inventory Optimization & Reverse Logistics

# Brief One Line Summary

An end-to-end supply chain analytics system that uses **ABCXYZ segmentation** to prioritize SKUs, forecast demand, optimize inventory, recover value from returns, and flag operational exceptions through an integrated control tower.

---

<img width="2752" height="1536" alt="image" src="https://github.com/user-attachments/assets/76484261-0047-463b-9dd9-5afe5373b13a" />

---


# 📋 Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Tools and Technologies](#tools-and-technologies)
5. [Methods](#methods)
6. [Key Insights](#key-insights)
7. [Dashboard / Model / Output](#dashboard--model--output)
8. [How to Run this Project](#how-to-run-this-project)
9. [Results & Conclusion](#results--conclusion)
10. [Future Work](#future-work)
11. [Author & Contact](#author--contact)

---

# Overview

**ChainOS** is a consulting-style supply chain decision support project built on the M5 Forecasting dataset.  
The project was designed to simulate how a real supply chain team would move from raw transactional data to business action.

The core idea is simple:

- not every SKU deserves the same level of attention,
- not every product behaves the same,
- and not every exception should be handled the same way.

That is why the project begins with **ABCXYZ segmentation**.

### Why ABCXYZ matters

- **ABC analysis** identifies the highest-value SKUs based on revenue contribution.
- **XYZ analysis** identifies the demand behavior of SKUs based on variability.
- Together, ABCXYZ helps determine:
  - which SKUs need accurate forecasting,
  - which SKUs need tighter inventory control,
  - which SKUs can be managed with simpler rules,
  - and which SKUs should be monitored most closely in the control tower.

This project combines:
- segmentation,
- forecasting,
- inventory optimization,
- supplier analysis,
- reverse logistics,
- root cause analysis,
- and dashboarding

into one integrated supply chain system.

# Problem Statement

Supply chains often fail because organizations treat all products equally.

In reality:
- some SKUs generate most of the revenue,
- some SKUs are highly volatile,
- some suppliers are unreliable,
- some exceptions repeat for the same root cause,
- and some returned inventory can still be monetized.

The problem ChainOS solves is:

**How can a supply chain team prioritize the right SKUs, forecast demand, optimize inventory, recover return value, and detect exceptions early enough to act?**

This project addresses that problem by building a full decision pipeline from data to action.

# Dataset

The project uses the **M5 Forecasting dataset**, including:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

From these raw files, the project generates multiple engineered datasets, including:

- `weekly_demand`
- `abcxyz_segments`
- `forecasts`
- `forecast_metrics`
- `inventory_parameters`
- `safety_stock_table`
- `inventory_policy`
- `optimized_inventory_policy`
- `supplier_deliveries`
- `returns_data`
- `returns_clean`
- `returns_decisions`
- `inventory_adjustments`
- `returns_kpis`
- `exceptions_log`
- `action_queue`
- `control_tower_dashboard`
- `ci_triggers`

### Data preparation logic

The original sales data was transformed from daily demand to weekly demand so it could be used for supply chain planning.  
Then the SKUs were prioritized using **ABCXYZ segmentation**, so the project focused on products that matter most from both a value and demand-behavior perspective.

# Tools and Technologies

- **Python**
- **Pandas**
- **NumPy**
- **Statsmodels**
- **Pmdarima**
- **PuLP**
- **Matplotlib**
- **Plotly**
- **Google Colab**
- **Power BI**
- **HTML / CSS / JavaScript** for the dashboard front end
- **GitHub** for project hosting and version control

# Methods

## 1) ABCXYZ Segmentation
SKUs were classified by:
- **ABC**: revenue contribution
- **XYZ**: demand variability

This was used to decide planning priority.

## 2) Weekly Demand Engineering
Daily M5 sales data was aggregated into weekly demand so it could support planning and forecasting.

## 3) Forecasting
Time-series models were used to forecast demand for priority SKUs.  
Forecast quality was measured using:
- MAE
- RMSE
- MAPE
- prediction intervals

## 4) Inventory Optimization
Safety stock and reorder points were estimated using:
- average demand
- demand variability
- lead-time variability
- service level assumptions

A linear programming model was then used to improve inventory policy.

## 5) Supplier Simulation
Since the M5 dataset does not contain procurement history, supplier delivery behavior was simulated to create realistic lead-time uncertainty.

## 6) Reverse Logistics
Returns were simulated and classified into:
- Grade A: Restock
- Grade B: Redistribute
- Grade C: Write-off

A disposition model was used to maximize recovery value.

## 7) Control Tower & RCA
Operational exceptions were detected automatically and mapped to root causes such as:
- forecast underestimate
- supplier delay
- inventory misallocation
- demand spike
- service level failure

## 8) Dashboarding
The final outputs were presented through a multi-page control tower dashboard.

# Key Insights

## ABCXYZ Insights
- A small group of SKUs contributed a disproportionate amount of revenue.
- Stable SKUs and volatile SKUs required different planning logic.
- Forecasting effort should be concentrated on high-value, high-uncertainty products.

## Forecasting Insights
- Some SKUs were highly predictable.
- Others showed intermittent demand and larger forecast errors.
- Confidence intervals were essential for inventory decisions.

## Inventory Insights
- Safety stock must reflect both demand uncertainty and lead-time uncertainty.
- Reorder points are more useful than static inventory rules.
- Tight budget constraints significantly affect service level.

## Supplier Insights
- Lead-time variation is a major driver of stockout risk.
- Delay rates differ across suppliers and should be monitored continuously.

## Reverse Logistics Insights
- Returns should not be treated as pure loss.
- A meaningful share of returned inventory can be recovered.
- Disposition decisions have direct financial impact.

## Control Tower Insights
- Many exceptions can be traced back to a small number of recurring root causes.
- Early detection is more useful than late reporting.
- Recommended actions should be operational, not just descriptive.

# Dashboard / Model / Output

The final dashboard includes five major views:

## 1) Executive Overview
Shows:
- Forecast accuracy
- Fill rate
- Inventory value
- Recovery value
- Total exceptions
- High-priority alerts
- business value generated

## 2) Inventory Control Tower
Shows:
- Safety stock
- Reorder point
- Inventory heatmap
- Stockout risk
- Coverage distribution

## 3) Demand & Forecasting
Shows:
- Actual vs forecast
- Forecast accuracy by SKU
- Prediction intervals
- Forecast risk leaderboard

## 4) Supplier & Reverse Logistics
Shows:
- Supplier delay rate
- Lead-time distribution
- Recovery funnel
- Returns grade distribution
- Recovery value trend

## 5) Control Tower & RCA
Shows:
- Exception distribution
- Root cause Pareto
- Action queue
- Priority alerts
- Continuous improvement triggers

### Key outputs generated
- Forecast tables
- Inventory policy tables
- Supplier performance tables
- Returns recovery tables
- Exception logs
- RCA action queue
- Control tower KPI tables
- Power BI-ready exports

# How to Run this Project

## Step 1: Open the notebooks
Open the project notebooks in **Google Colab** or **Jupyter Notebook**.

## Step 2: Load the raw M5 dataset
Place the raw files in the expected data folder:
- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

## Step 3: Run the pipeline in order
The recommended flow is:

1. Data preparation  
2. ABCXYZ segmentation  
3. Weekly demand creation  
4. Forecasting  
5. Inventory optimization  
6. Supplier simulation  
7. Reverse logistics  
8. Exception detection  
9. RCA engine  
10. Dashboard export

## Step 4: Open the dashboard
Launch the dashboard files and inspect:
- executive summary
- inventory control tower
- demand analytics
- supplier & reverse logistics
- control tower & RCA

# Results & Conclusion

ChainOS demonstrates how supply chain analytics can move beyond forecasting into actual decision support.

The project shows how to:
- prioritize high-value SKUs using **ABCXYZ**
- forecast demand with time-series models
- build inventory policies with safety stock and reorder points
- model supplier uncertainty
- recover value from returns
- detect recurring operational exceptions
- recommend corrective actions
- and present all of it through an executive dashboard

### Final takeaway
This project is not just a model.  
It is a **supply chain operating framework** that connects analytics with business action.

# Future Work

Possible next improvements:

- Multi-echelon inventory optimization
- Supplier segmentation by risk tier
- Automated replenishment alerts
- Scenario-based stress testing
- Demand sensing with external variables
- Streamlit deployment
- Live Power BI refresh pipeline
- More advanced ABCXYZ rules by category or store

# Author & Contact

**Author:** Kabir Batra  
**Focus Areas:** Supply Chain Analytics, Forecasting, Inventory Optimization, Operations Research, Power BI

LinkedIn: LinkedIn.com/kabir-batra-profile
Email: kbatra2005@gmail.com

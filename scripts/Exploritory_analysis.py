import pandas as pd
import matplotlib as md
import numpy as np

calendar = pd.read_csv('../data/raw/calendar.csv')
sales    = pd.read_csv('../data/raw/sales_train_validation.csv')
prices   = pd.read_csv('../data/raw/sell_prices.csv')

# Group by item_id and week, then check if all prices are equal across stores
price_check = (prices.groupby
 (['item_id', 'wm_yr_wk'])['sell_price']
    .nunique()
    .reset_index(name='unique_price_count')
)

# Filter where prices differ across stores
different_prices = price_check[price_check['unique_price_count'] > 1]

print(f"Total item-week combinations with differing prices: {len(different_prices)}")
print(different_prices.head(100))

price_check = (
    prices.groupby(['item_id', 'wm_yr_wk'])['sell_price']
    .nunique()
    .reset_index(name='unique_price_count')
)

# Filter where prices differ across stores
different_prices = price_check[price_check['unique_price_count'] > 1]

# Merge back to see which stores have those differing prices
stores_with_diff_prices = prices.merge(
    different_prices,
    on=['item_id', 'wm_yr_wk'],
    how='inner'
).sort_values(['item_id', 'wm_yr_wk', 'store_id'])

print("Stores with differing prices for the same item-week combination:")
print(stores_with_diff_prices.head(10))

duplicate_check = (
    prices
    .groupby(
        ["item_id", "store_id", "wm_yr_wk"]
    )
    .size()
    .reset_index(name="count")
)

duplicates = duplicate_check[
    duplicate_check["count"] > 1
]

print("Duplicate combinations:", len(duplicates))
duplicates.head()

states_count = sales.groupby('state_id')['state_id'].count().reset_index(name='count')
print (states_count)

CA_sales = sales[sales['state_id'] == 'CA']

stores = CA_sales.groupby('store_id')['store_id'].count().reset_index(name='count')

Processed_data = CA_sales.loc[CA_sales['store_id'].isin(['CA_1', 'CA_2'])]
(Processed_data).to_parquet('../data/processed/processed_data.parquet')

calendar.head()

calendar_subset = calendar.drop(columns=['weekday', 'wday', 'month', 'year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI'])

prices.head()

Prices_stores_CA = prices[prices['store_id'].isin(['CA_1', 'CA_2'])]
Prices_stores_CA.head()

stores_with_diff_prices = Prices_stores_CA.groupby('store_id')['store_id'].count().reset_index(name='count')
Prices_stores_CA.to_parquet('../data/processed/Prices_stores_CA.parquet')
import os
import pandas as pd # type: ignore

# ensures we've got the necessary data file
assert os.path.exists("data/sales.csv"), "Sales data file is missing!"

# ################################################################ #
#                            CHECK CSV                             #
# ################################################################ #

# retrieve the data
data: pd.DataFrame = pd.read_csv('data/sales.csv') # type: ignore

print("CSV Data:")
print(data)
print(f"\nShape: {data.shape[0]} rows, {data.shape[1]} columns")

# DataFrame.shape returns tuple(total rows, total columns)

# ################################################################ #
#                      COMPUTES TOTAL PRICES                       #
# ################################################################ #

data['total'] = data['quantity'] * data['price']
print("\nWith totals:")
print(data)

# make directory 
os.makedirs('output', exist_ok=True)

# ################################################################ #
#                              EXPORT                              #
# ################################################################ #

# JSON
data.to_json('output/sales_data.json', orient='records', indent=2) # type: ignore

# EXCEL
data.to_excel('output/sales_data.xlsx', index=False) # type: ignore

# CSV
data.to_csv('output/sales_with_totals.csv', index=False)

# ################################################################ #
#                       MORE STRUCTURED DATA                       #
# ################################################################ #

# 1. Group by product
#    - DataFrame.groupby combine rows which has the same value of 'product'
#    - The groupped data from the other colume has yet to be processed. Pandas just holding it for now
# 2. Aggregate the grouped data
#    - this is where we tell pandas how to process those groupped data
#    - using predefined string shortcut that pandas recognizes:
#    - 'sum' to sum up the values
#    - 'first' to just take the first value from the group
#    - e.g.: 'quantity': 'sum': for each group, take all the quantity values and sum them.
#    - e,g.: 'price': 'first': for each group, take the first price value.

simplified: pd.DataFrame = data.groupby('product', as_index=False).agg({ # type: ignore
    'quantity': 'sum',
    'price': 'first',
})

# compute total price for each product
simplified['total'] = simplified['quantity'] * simplified['price']

# reorder columns
simplified = simplified[['product', 'quantity', 'price', 'total']] # type: ignore

print("\nSimplified Data:")
print(simplified)
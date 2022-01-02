# Task1 Data Processing

import pandas as pd
pd.set_option("display.max.columns", None)
pd.set_option("display.width", 500)
# pd.set_option("display.max.rows", None)
pd.set_option("display.expand_frame_repr", False)
from mlxtend.frequent_patterns import apriori, association_rules

## read dataset
df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

## reading frequently used functions from py
from helpers.helpers import check_df, retail_data_prep, \
    replace_with_thresholds, outlier_thresholds

check_df(df)
df = retail_data_prep(df)

## picking up Germany
df_deu = df[df["Country"]=="Germany"]
check_df()

## creating invoices
### amount of goods bought in a transaction
df_deu.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)
### reshaping table: goods as variables, bundles(invoices) in columns, pivoting with unstack()
df_deu.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]
### filling NAs with 0 because the good does not exist in that bundle
df_deu.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]
### filling the amount of existing goods with 1
df_deu.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).\
    unstack().fillna(0).\
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5] # applymap() applies to entire dataset

### defining the the process as function

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

## goods frequently bought together
deu_inv_pro_df = create_invoice_product_df(df_deu, id = True)

## checking the IDs and description for a possible mistake
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


# Task 2 Association Rules Based on German Customers
### grouping frequently boguht items
frequent_itemsets = apriori(deu_inv_pro_df, min_support=0.01, use_colnames=True)
### sorting frequently bought items
frequent_itemsets.sort_values("support", ascending=False).head(20)
### finding association rules => association_rules()
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head(20)
rules.sort_values("lift", ascending=False).head(20) # good if lift is greater than 1

# Task 3 Accessing the Good through IDs
## name of the good by ID
check_id(df_deu, 21987)
check_id(df_deu, 23235)
check_id(df_deu, 22747)


# Task 4 Good Recommendations for Customers

def arl_recommender(rules_df, product_id, rec_count=5):
    sorted_rules = rules_df.sort_values("lift", ascending=False) # rules in df based on
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]): ## eumerate helps to get index and value
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules, 21987, 1)
arl_recommender(rules, 23235, 1)
arl_recommender(rules, 22747, 1)

# Task 5 Names of Recommended Goods

check_id(df_deu, 21989)

check_id(df_deu, 23243)

check_id(df_deu, 22746)

######################

ef retail_data_prep(dataframe):
    dataframe.drop(dataframe[dataframe["StockCode"] == "POST"].index, inplace=True) #### Guncelle!!!
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe
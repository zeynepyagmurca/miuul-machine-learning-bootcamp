# Dataset Summary:
# Number of Variables: 8
# Number of Observations: 541,909
# Size: 45.6 MB
# Variable Descriptions:
# InvoiceNo: Invoice number (If the code starts with 'C', it indicates the transaction has been canceled.)
# StockCode: Product code (Unique for each product)
# Description: Product name
# Quantity: Quantity of the product
# InvoiceDate: Invoice date
# UnitPrice: Invoice price
# CustomerID: Unique customer ID

# Importing Libraries
import numpy as np
import pandas as pd
from numba.parfors.parfor import replace_functions_checkers_map
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from sympy import primitive
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import warnings

#Pandas Output Configuration
pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',50)
pd.set_option('display.float_format',lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")

#Importing Dataset
df = pd.read_excel("online_retail_II.xlsx")
df = df.copy()


# Functions Required for Exploratory Data Analysis and Feature Engineering
def check_data(dataframe, head=5):
    print("##### SHAPE ####")
    print(dataframe.shape)
    print("#### INFO ####")
    print(dataframe.info())
    print("#### DESCRIBE ####")
    print(dataframe.describe([0.01, 0.1, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]))
    print("#### NA VALUES ####")
    print(dataframe.isnull().sum())
    print("#### FIRST {} ROWS ####".format(head))
    print(dataframe.head(head))

def select_country(dataframe, country):
    new_dataframe = dataframe.loc[dataframe["Country"] == country]
    return new_dataframe

def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any (axis = None):
        return True
    else:
        return False

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_tresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


######## Exploratory Data Analysis and Feature Engineering ########

check_data(df)

# Filtering and Removing Invalid Transactions
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df["Quantity"] > 0)]
df.dropna(inplace=True)

df.describe([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

# Identifying Products with Zero Price
df[df["Price"] == 0]["StockCode"].unique()

df = df[df["StockCode"] != "M"]

invalid_codes = df[df["StockCode"].astype(str).str.contains(r"[a-zA-Z]{3,}")]["StockCode"].unique().tolist()
invalid_codes

df[df["StockCode"].isin(invalid_codes)].groupby(["StockCode"]).agg({"Invoice": "nunique",
                                                                    "Quantity": "sum",
                                                                    "Price": "sum",
                                                                    "Customer ID": "nunique"})

df = df[~df["StockCode"].isin(invalid_codes)].reset_index(drop=True)

check_data(df)

df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()

df.describe([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

max_invoice_date = df["InvoiceDate"].max()
today_date = (max_invoice_date + dt.timedelta (days=2))

today_date
max_invoice_date

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda date:(today_date -date.max()).days,#recency
                                     "Invoice":"nunique",#frequency
                                     "TotalPrice":"sum"})#monetary

rfm.head()

rfm.columns = ["Recency", "Frequency", "Monetary"]

rfm = rfm[(rfm["Monetary"] > 0) & (rfm["Frequency"] > 0 )]

rfm.describe([0.01, 0.1, 0.25, 0.5, 0.75, 0.90, 0.99])

for col in rfm.columns:
    print(col, check_outlier(rfm, col))

for col in rfm.columns:
    replace_with_tresholds(rfm, col)

rfm.describe()

rfm["Recency"].hist(bins= 20)
plt.title("Recency")
plt.show()

rfm["Frequency"].hist(bins= 20)
plt.title("Frequency")
plt.show()

for col in ["Recency", "Frequency"]:
    rfm[f"LOG_{col}"] = np.log1p(rfm[col])

rfm.head()

sc = StandardScaler()
sc.fit(rfm[["LOG_Recency", "LOG_Frequency"]])
scaled_rf = sc.transform(rfm[["LOG_Recency", "LOG_Frequency"]])
scaled_rf

scaled_df = pd.DataFrame(index = rfm.index, columns=["LOG_Recency", "LOG_Frequency"],data=scaled_rf)
scaled_df

scaled_df.describe()

######## Model Development ########

# K-Means Clustering

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=30)
elbow.fit(scaled_df)
elbow.show();

k_=elbow.elbow_value_
print(k_)

k_means = KMeans(n_clusters=k_, random_state=99).fit(scaled_df)

segments = k_means.labels_
segments

rfm["KMeans_Segments"] = segments
rfm.head()

rfm.groupby("KMeans_Segments").agg({"Recency":["mean","median","count"],
                                    "Frequency":["mean","median","count"],
                                    "Monetary":["mean","median","count"]})

plt.figure(figsize = (20,10))
sns.boxplot(x="KMeans_Segments", y="Monetary", data=rfm)
plt.show();


# Hierarchical Clustering

hc_complete = linkage(scaled_df, 'complete')
plt.figure(figsize=(7,5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
                  truncate_mode="lastp",
                  p=10,
                  show_contracted=True,
                  leaf_font_size=10)

plt.axhline(y=1.2, color='r', linestyle='--')
plt.show()

hc=AgglomerativeClustering(n_clusters=6)
segments = hc.fit_predict(scaled_df)
rfm["Hierarchy_Segments"] = segments

rfm.groupby("Hierarchy_Segments").agg({"Recency":["mean","median","count"],
                                       "Frequency":["mean","median","count"],
                                       "Monetary":["mean","median","count"]})

plt.figure(figsize=(20,10))
sns.boxplot(x="Hierarchy_Segments", y="Monetary", data=rfm)
plt.show();

# IMPORT PACKAGES:
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
%matplotlib inline


# READ DATA
df = pd.read_csv(
    '/Users/clairesarraille/git-repos/ph2finproj/dsc-phase-2-project-main/data/kc_house_data.csv')


# REDUCE FEATURES:
df = df.drop(['id', 'date', 'view', 'sqft_above', 'sqft_basement', 'yr_renovated',
             'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'], axis=1)
df.head()


# CONVERT PRICE AND LOT SIZE TO READABLE FORMAT:
df['price_millions'] = df['price'] / 1000000  # price in millions
df['acres_lot'] = df['sqft_lot'] / 43560  # lot size in acres


# Frequency Histogram - price_millions:
ax = df.hist(column='price_millions',
             bins='auto',
             grid=False,
             figsize=(12, 12))

ax = ax[0]
for x in ax:

    x.set_title("Distribution of Home Sale Price", size=20)

    # Set x-axis label
    x.set_xlabel("Sale Price of Home", labelpad=15, weight='bold', size=10)
    x.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}' + ' Million'))

    # Set y-axis label
    x.set_ylabel("Number of Homes Sold", labelpad=15, weight='bold', size=10)
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))


# Find Fields with NaN values:
df.isnull().sum(


# Outliers Using Boxplots:
sns.boxplot(x=df['price_millions'])
sns.boxplot(x=df['acres_lot'])
sns.boxplot(x=df['bedrooms'])
sns.boxplot(x=df['bathrooms'])


# Outliers Using Sorting:
df.sort_values('price_millions', ascending=False).head(15)
df.sort_values('bedrooms', ascending=False).head(10)
df.sort_values('bathrooms', ascending=False).head(10)
df.sort_values('sqft_living', ascending=False).head(10)
df.sort_values('acres_lot', ascending=False).head(10)


# Descriptive Statistics Using .describe():
format_dict={'bedrooms': '{:.2f}', 'bathrooms': '{:.2f}', 'floors': '{:.2f}', 'sqft_living': '{:20,.2f}',
    'sqft_lot': '{:20,.2f}', 'acres_lot': '{:.2f}',  'price': '${:20,.0f}', 'yr_built': '{:.0f}'}
df[['price', 'yr_built', 'bedrooms', 'bathrooms', 'floors', 'sqft_living',
    'sqft_lot', 'acres_lot']].describe().style.format(format_dict)


# Deal with "waterfront" Missing Values:
df['waterfront']=df['waterfront'].fillna(0)  # Fill NaN values with 0
df['waterfront']=df["waterfront"].astype(int)  # Cast float value as integer


# Write Data to Pickle:
with open('df_data.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)


# Read Data from Pickle:
with open('df_data.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    df=pickle.load(f)


# Scatter Matrix Distribution:
df_scatter=df.drop(['price', 'condition', 'sqft_lot',
                   'waterfront', 'grade', 'yr_built', 'floors'], axis=1)
pd.plotting.scatter_matrix(df_scatter, figsize=(12, 12))
plt.show()


# Density Histogram - price_millions by waterfront:
# Create separate dfs:
water_df=df.loc[df['waterfront'] == 1]
no_water_df=df.loc[df['waterfront'] == 0]
# Plot:
plt.figure(figsize=(12, 8))
binsize=10
water_df.price_millions.plot.hist(
    bins=binsize, density=True,  alpha=0.6, label="Waterfront Prices in millions");
no_water_df.price_millions.plot.hist(
    bins=binsize, density=True, alpha=0.6, label='Non-Waterfront Prices in millions');
plt.legend()
plt.show()
# With KDE curve:
plt.figure(figsize=(12, 8))
sns.distplot(water_df.price_millions, kde=True)
sns.distplot(no_water_df.price_millions, kde=True)
plt.title('Comparing House Prices: Waterfront vs. Non-Waterfront')
plt.show()


# Density Histogram - Simple price_millions
plt.figure(figsize=(12, 8))
sns.distplot(df.price_millions, kde=True)
plt.title('Density Histogram of Housing Prices in Kings County')
plt.show()


# Create a Sensible df for modeling:
df_model=df=df.drop(['acres_lot', 'price'], axis=1)




# Check for Multicollinearity:
# Correlation Heatmap:
correlations=df.corr()
plt.figure(figsize=(12, 12))  # Set size of figure
ax=sns.heatmap(correlations, cmap="Greens", annot=True,
               fmt='.2f', cbar_kws={"shrink": .77}, square=True)
ax.set(title='King County House Sales Dataset Correlation Heatmap')
bottom, top=ax.get_ylim()
ax=ax.set_ylim(bottom + 0.5, top - 0.5)
plt.xticks(rotation=70)
plt.show()


# View Correlations - if a column has a high VIF value, may want to drop it from model
correlations["price"].sort_values(ascending=False)
correlations["sqft_living"].sort_values(ascending=False)


# Scatterplot of sqft_living vs. price
ax=sns.scatterplot(x="sqft_living", y="price", data=df)
ax.set_title("House Price vs. Square Footage")
ax.set_xlabel("Living Area in square feet")
ax.set_ylabel("House Sale Price")


# Scatterplot, sqft_living vs. price - waterfront vs. none
sns.lmplot(x="sqft_living", y="price", data=df, hue="waterfront")


# Normalize Numeric Data - using min-max feature scaling
# copy the data
df_min_max_scaled=df.copy()

# apply normalization techniques
for column in df_min_max_scaled.columns:
	df_min_max_scaled[column]=(df_min_max_scaled[column] - df_min_max_scaled[column].min()) / \
	                           (df_min_max_scaled[column].max() - \
	                            df_min_max_scaled[column].min())

# view normalized data
print(df_min_max_scaled)

# the independent variables set
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'grade',
        'waterfront', 'floors', 'sqft_lot', 'yr_built', 'condition']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(
    X.values, i) for i in range(len(X.columns))]

print(vif_data.sort_values(by=['VIF'], ascending=False))

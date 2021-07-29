# Linear Regression Attempt #1

# Predictive Regression Model:
# Data is divided into features dataset and target dataset:
X = df.drop('price', axis=1)
y = df['price']

# Divide into test and training:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Standardized feature matrix
X_std = StandardScaler().fit_transform(X_train)

# Linear Regression
lr = LinearRegression()

# Evaluate model with cross-validation
cvs = cross_val_score(estimator=lr, X=X_train,
                      y=y_train,
                      cv=10, scoring='r2')
print('CV score: %.3f ± %.3f' % (cvs.mean(), cvs.std()))

# I'm not understanding why it shows that each bedroom accounts for -$41,932 drop in price
lr.fit(X_train, y_train)
coef_list = list(lr.coef_)
name_list = list(X_train.columns)
pd.Series(coef_list, index=name_list)


# Linear Regression Attempt #2:

# Predictive Regression Model:
# Data is divided into features dataset and target dataset:
X = df[['sqft_living', 'bathrooms', 'bedrooms',
        'floors', 'sqft_lot', 'waterfront']]
y = df['price']

# Divide into test and training:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Standardized feature matrix
X_std = StandardScaler().fit_transform(X_train)

# Linear Regression
lr = LinearRegression()

# Evaluate model with cross-validation
cvs = cross_val_score(estimator=lr, X=X_train,
                      y=y_train,
                      cv=10, scoring='r2')
print('CV score: %.3f ± %.3f' % (cvs.mean(), cvs.std()))

lr.fit(X_train, y_train)
coef_list = list(lr.coef_)
name_list = list(X_train.columns)
pd.Series(coef_list, index=name_list)


# Linear Regression Attempt #3:

# Predictive Regression Model:
# Data is divided into features dataset and target dataset:
X = df_min_max_scaled[['sqft_living', 'bathrooms',
                       'bedrooms', 'floors', 'sqft_lot']]
y = df_min_max_scaled['price']

# Divide into test and training:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Standardized feature matrix
X_std = StandardScaler().fit_transform(X_train)

# Linear Regression
lr = LinearRegression()

# Evaluate model with cross-validation
cvs = cross_val_score(estimator=lr, X=X_train,
                      y=y_train,
                      cv=10, scoring='r2')
print('CV score: %.3f ± %.3f' % (cvs.mean(), cvs.std()))

lr.fit(X_train, y_train)
coef_list = list(lr.coef_)
name_list = list(X_train.columns)
pd.Series(coef_list, index=name_list)

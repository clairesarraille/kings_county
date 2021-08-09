from xgboost import XGBRegressor

xgb = XGBRegressor()

xgb.fit(X_train, y_train)

xgb_predict_y_train = xgb.predict(X_train)


print('The R squared value is: ' +
      str(metrics.r2_score(y_train, xgb_predict_y_train)))


plt.scatter(y_train, xgb_predict_y_train)
plt.xlabel('price')
plt.ylabel('predicted price')
plt.show()


plt.scatter(xgb_predict_y_train, y_train - xgb_predict_y_train)
plt.xlabel('predicted')
plt.ylabel('residuals')

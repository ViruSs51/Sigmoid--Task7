from sklearn.naive_bayes import GaussianNB
# Creating the model object.
gauss = GaussianNB()
# Fitting the model.
gauss.fit(X_train_scaled, y_train)
# Making predictions on new samples.
y_pred = gauss.predict(X_test_scaled)

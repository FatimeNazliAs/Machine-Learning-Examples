import numpy as np
from sklearn.linear_model import LinearRegression

#Now, provide the values for independent variable X −
X=np.array([[1,1],[1,2],[2,2],[2,3]])
#Next, the value of dependent variable y can be calculated
y=np.dot(X,np.array([1,2]))+3

#create a linear regression object
regr=LinearRegression(fit_intercept=True,normalize=True,copy_X=True,n_jobs=2).fit(X,y)
regr.predict(np.array([[3,5]]))

# get the coefficient of determination of the prediction
regr.score(X,y)
# estimate the coefficients by using attribute named ‘coef’
regr.coef_
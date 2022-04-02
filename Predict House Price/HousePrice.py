import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split



#load the boston housing data set
from sklearn.datasets import load_boston
boston=load_boston()


#transform data set to data frame
#x values are independent variable also known as features
#y values ara dependent variable also known as target

df_x=pd.DataFrame(boston.data,columns=boston.feature_names)
df_y=pd.DataFrame(boston.target)

#get statics from data set
df_x.describe()

#linear regressin model
reg=linear_model.LinearRegression()

#data 67% training and 33% testing
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.33,random_state=42)


#train the model
reg.fit(x_train,y_train)

#print coefficents for each feature/column of our model
print(reg.coef_) #f(x)=mx+b=y m is coefficent


y_pred=reg.predict(x_test)
print(y_pred)

#print actual values
print(y_test)

#check model performance wiht mean squared error
print(np.mean((y_pred-y_test)**2))

#check model with sklearn.metrices
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))


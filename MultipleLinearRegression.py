import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

df = pd.read_csv("FuelConsumptionCo2.csv")

# print(df)

cdf = df[["ENGINESIZE","FUELCONSUMPTION_COMB","CYLINDERS","CO2EMISSIONS"]]
print(cdf)

plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color="blue")
plt.xlabel("ENGINE SIZE")
plt.ylabel("CO2 EMISSIONS")
plt.show()

import sklearn.linear_model as l

regr = l.LinearRegression()
msk = np.random.rand(len(cdf))<0.8
train = cdf[msk]
test = cdf[~msk]

x = np.asanyarray(train[["ENGINESIZE","FUELCONSUMPTION_COMB","CYLINDERS"]])
y = np.asanyarray(train[["CO2EMISSIONS"]])
regr.fit(x,y)

print("Intercept: ",regr.intercept_)
print("Coefficients: ",regr.coef_)

test_x = np.asanyarray(test[["ENGINESIZE","FUELCONSUMPTION_COMB","CYLINDERS"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])
test_y_ = regr.predict(test_x)

print("Mean Squared Error: %.2f"%np.mean((test_y-test_y_)**2))
print("Variance score: %.2f"%regr.score(test_x,test_y)) 



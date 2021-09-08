import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.datasets import load_boston
import seaborn as sns
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
from flask import Flask,request,jsonify

app = Flask(__name__)


boston = load_boston()
boston.keys()
bos = pd.DataFrame(boston.data,columns = boston.feature_names)
boston.DESCR

bos['MEDV'] = boston.target

bos.isnull().sum()

report = ProfileReport(bos)
report.to_widgets()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.displot(bos['MEDV'],bins=30)
plt.show()
corr_matrix = bos.corr().round(2)
sns.heatmap(corr_matrix,annot = True)

features = ['LSTAT', 'RM']
target = bos['MEDV']
for i, col in enumerate(features):
    plt.subplot(1, 2, i + 1)
    plt.scatter(x=bos[col], y=target)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')

X = pd.DataFrame(np.c_[bos['LSTAT'], bos['RM']],columns = ['LSTAT','RM'])

Y = bos['MEDV']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 5)
lr = LinearRegression()
lr.fit(X_train,Y_train)

y_train_pred = lr.predict(X_train)
error = np.sqrt(mean_squared_error(Y_train,y_train_pred))

score = lr.score(X_train,Y_train)

def adj_r2(x,y):
    r2 = lr.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adj_r2

y_test = np.array(Y_test).reshape(-1,1)
y_test_pred = lr.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test,y_test_pred))

test_score = lr.score(X_test,Y_test)
adj_r2(X_test,Y_test)
pickle.dump(lr,open('assign2.sav','wb'))
model = pickle.load(open('assign2.sav','rb'))
model.predict([[4.98,6.575]])

@app.route('/predict',methods = ['GET','POST'])
def pred():
    RM = float(request.json['RM'])
    LSTAT = float(request.json['LSTAT'])
    price = model.predict([[RM,LSTAT]])
    return jsonify('Predicted price {}:'.format(price[0]))


if __name__ == '__main__':
    app.run()



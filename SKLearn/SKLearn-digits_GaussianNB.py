from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

digits=load_digits()

X=digits.data
y=digits.target

x_dtrain,x_dtest,y_dtrain,y_dtest=train_test_split(X,y,random_state=0)

d_model=GaussianNB()

d_model.fit(x_dtrain,y_dtrain)
y_dmodel=d_model.predict(x_dtest)

print(accuracy_score(y_dtest,y_dmodel))
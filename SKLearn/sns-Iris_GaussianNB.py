import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

iris=sns.load_dataset('iris')

X_iris=iris.drop('species',axis=1)
X_iris.shape

y_iris=iris['species']
y_iris.shape

x_train,x_test,y_train,y_test=train_test_split(X_iris,y_iris,random_state=1)

model=GaussianNB()

model.fit(x_train,y_train)
y_model=model.predict(x_test)

print(accuracy_score(y_model,y_test))
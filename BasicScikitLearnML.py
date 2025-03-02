#%% import desired data--------------------------------------------------------
import seaborn as sns
iris=sns.load_dataset('iris')
iris.head()

# %% plot for visualization-----------------------------------------------------
sns.pairplot(iris,hue='species',size=1.5)
# %% process data and split for training and validation---------------------
X_iris=iris.drop('species',axis=1)
X_iris.shape
y_iris=iris['species']
y_iris.shape

import sklearn
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X_iris,y_iris,random_state=1)
# %% import model structure--------------------------------------------------
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)
y_model=model.predict(x_test)
# %%test with validation-------------------------------------------------------
from sklearn.metrics import accuracy_score
accuracy_score(y_model,y_test)
#%%USING DIMENSIONALITY REDUCTION AND USING GAUSSIAN MIXTURE MODEL CLASSIFICATION
from sklearn.decomposition import PCA
model=PCA(n_components=2)
model.fit(X_iris)#run PCA
x_reduced=model.transform(X_iris)#apply PCA onto data set

iris['PCA1']=x_reduced[:,0]
iris['PCA2']=x_reduced[:,1]
sns.lmplot(x='PCA1',y='PCA2',hue='species',data=iris,fit_reg=False)
# %%
from sklearn.mixture import GaussianMixture
model=GaussianMixture(n_components=3,covariance_type='full')
model.fit(X_iris)
y_gmm=model.predict(X_iris)
iris['cluster']=y_gmm
sns.lmplot(x='PCA1',y='PCA2',data=iris,hue='species',col='cluster',fit_reg=False)#2D PCA projection for visual sense, but still using all four features
# %%HANDWRITTEN DIGIT TASK BELOW------------!!!!!!
#%%
from sklearn.datasets import load_digits
digits=load_digits()
digits.images.shape

import matplotlib.pyplot as plt
fig,axes=plt.subplots(10,10,figsize=(8,8),subplot_kw={'xticks':[],'yticks':[]},gridspec_kw=dict(hspace=0.1,wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i],cmap='binary',interpolation='nearest')
    ax.text(0.05,0.05,str(digits.target[i]),transform=ax.transAxes,color='green')
# %%
X=digits.data
Y=digits.target
# %%reduce dimensions using Manifold learning----------------------------------
from sklearn.manifold import Isomap
iso=Isomap(n_components=2)
iso.fit(X)
data_projected=iso.transform(X)
data_projected.shape

plt.scatter(data_projected[:,0],data_projected[:,1],c=Y,edgecolors='none',alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral',10))
plt.colorbar(label='digit label',ticks=range(10))
plt.clim(-0.5,9.5)
# %%
from sklearn.naive_bayes import GaussianNB
x_dtrain,x_dtest,y_dtrain,y_dtest=train_test_split(X,Y,random_state=0)
d_model=GaussianNB()
model.fit(x_dtrain,y_dtrain)
y_dmodel=model.predict(x_dtest)
accuracy_score(y_dmodel,y_dtest)
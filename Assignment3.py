'''
Part 1
Create a circle dataset, and apply a SVM on it.
'''

import matplotlib.pyplot as plt
import numpy as np
##make points that fit on a circle

import math
pi = math.pi

###The sklearn doesn't let us control the noise of the two and number of points, but a single one.
#sklearn.datasets.make_circles(n_samples=100, *, shuffle=True, noise=None,

meanshift_0 = 5
meanshift_1 = 3

radius_inner = 2
var_inner = 1
num_inner = 499

radius_outer = 5
var_outer = 0.5
num_outer = 499

def PointsInCircum(r,n):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]

a = PointsInCircum(radius_inner,num_inner)
a = np.array(a)
a = a +np.random.normal(0,var_inner,a.shape)
#shift it by some value "meanshift_x"
a[:,0] = a[:,0]+meanshift_0
a[:,1] = a[:,1]+meanshift_1

plt.scatter(a[:,0],a[:,1])
plt.show()

#add noise

#set these as class 0
classlist_a = np.zeros(a.shape[0])

##create a smaller circle
b = PointsInCircum(radius_outer,num_outer)
b = np.array(b)
b = b +np.random.normal(0,var_outer,b.shape)
#shift it by some value "meanshift_x"
b[:,0] = b[:,0]+meanshift_0
b[:,1] = b[:,1]+meanshift_1
plt.scatter(b[:,0],b[:,1])
plt.show()

classlist_b = np.ones(b.shape[0])

#combine the two, and shuffle it before our classifier.
print(a.shape)
print(b.shape)
X = np.concatenate((a,b))
print(X.shape)

Y = np.concatenate((classlist_a, classlist_b))
print(Y.shape)
print(X[:,0].shape)
ax = plt.subplot()
ax.scatter(X[:,0],X[:,1],c=Y.squeeze())
plt.title('Two classes of circles')
plt.xlabel('x_0')
plt.ylabel('x_1')
plt.show()

#####we should split into train and test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#if using entire model
#X_train = X
#y_train = Y
###############now we design a classifier####################
from sklearn import svm
clf = svm.SVC(kernel='rbf', C=1, random_state=0, gamma='scale') #000)
clf.fit(X_train, y_train)

###output the accuracy
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test, y_pred)
print(cf)
accuracy = np.trace(cf)/y_pred.shape[0]
print('accuracy', accuracy)

################VISUALIZE THE RESULTS#######################
Z1 = clf.decision_function(X)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()

'''
Part 2
A simply case of plug into SVM and output result
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#for windows, download the zip file of graphviz.
#https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft
#https://graphviz.gitlab.io/_pages/Download/Download_windows.html
#then replace the path where you put it.
import os
from sklearn.metrics import confusion_matrix

os.environ["PATH"] += os.pathsep + 'D:/Documents/Software/graphviz-2.38/release/bin'

in_file = '../data/COVID-19_formatted_dataset.csv'   #this is excel file so we use different reader. install slrd

df = pd.read_csv(in_file)
print('read success')
print(df['SARS-Cov-2 exam result'])  #okay now we have our data.
codes = {'negative':0, 'positive':1}
labels = df['SARS-Cov-2 exam result'].map(codes)
print(labels)  #lets explore history

####select the features we will study...we will ignore age and obviously label.
cols = list(np.array([1]))+list(np.arange(3,6))
input_data = df.iloc[:,cols].to_numpy()
y = np.expand_dims(labels.to_numpy(),axis=1)
#Split train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_data, y, test_size=0.2, random_state=0)

#lets remove a few of the X values.

#SVM , class_weight='balanced' means more positives
from sklearn import svm
clf = svm.SVC(kernel='rbf', C=5, random_state=0, probability=True)
#train the tree
clf = clf.fit(X_train, y_train)
#plot the tree
feature_names={0:'age',1:'hematocrit',2:'hemo',3:'platelets',4:'mean_platelet_volume',5:'rbc',6:'lymphocytes',7:'MCHC',8:'leukocytes',9:'basophils',10:'MCH',11:'eosinophils',12:'MCV',13:'monocytes',14:'rdw'}
c_names={0: 'healthy', 1: 'covid_19'}

#test the tree
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = clf.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for DT = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for DT = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))
print(cm_test)

#for the test case
X_patient = X_test[0].reshape(1,-1)    #assume this is already binned
y_out = clf.predict_proba(X_patient.reshape(1,-1))
print('predicted first test sample')
print(y_out)

####find a test case where its probbly is not 100%, and explain how this is computed.
y_out = clf.predict_proba(X_test)
print(y_out[0:4])
print('actual', y_test[0:4])
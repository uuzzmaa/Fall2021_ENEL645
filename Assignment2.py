'''
On the COVID-19 dataset, to start with.

Notes: install graphviz, give code for visualization
*explain the visualization what it means*
teach about the decision boundaries for (3 features) on iris. and how it can be obtained from the graph too.
https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html


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
cols = cols = list(np.array([1]))+list(np.arange(3,17))
input_data = df.iloc[:,cols]
y = np.expand_dims(labels.to_numpy(),axis=1)
#Split train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_data.to_numpy(), y, test_size=0.2, random_state=0)

print(X_train[0])


#In decision tree, if we compute entropy per feature, it doesnt matter if we normalize or not..
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth =4, random_state=0)
#train the tree
clf = clf.fit(X_train, y_train)
#plot the tree
feature_names={0:'age',1:'hematocrit',2:'hemo',3:'platelets',4:'mean_platelet_volume',5:'rbc',6:'lymphocytes',7:'MCHC',8:'leukocytes',9:'basophils',10:'MCH',11:'eosinophils',12:'MCV',13:'monocytes',14:'rdw'}
c_names={0: 'healthy', 1: 'covid_19'}
tree.plot_tree(clf, feature_names=feature_names, class_names=c_names)
plt.show()

'''
#lets define some properties..
feature_names={0:'age',1:'hematocrit',2:'hemo',3:'platelets',4:'mean_platelet_volume',5:'rbc',6:'lymphocytes',7:'MCHC',8:'leukocytes',9:'basophils',10:'MCH',11:'eosinophils',12:'MCV',13:'monocytes',14:'rdw'}
c_names={0: 'healthy', 1: 'covid_19'}
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, class_names=c_names) # , feature_names=feature_names
graph = graphviz.Source(dot_data)
graph.render("covid_tree")  #output will be a pdf with the tree in high quality. plt.show will make plot blurry.
'''
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

'''
DECISION BOUNDARY
Interpreting and visualizing a decision tree's decision boundary
https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html#sphx-glr-auto-examples-tree-plot-iris-dtc-py
This base code is from the tutorial of sklearn (plotting parts)
# I just modified to fit our data and depth, and explain one part of the subplot for X0 and X1

https://towardsdatascience.com/understanding-decision-trees-once-and-for-all-2d891b1be579

'''
from sklearn.tree import DecisionTreeClassifier, plot_tree


# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

print(labels)  #lets explore history


dataset = X_train  # we only take the first two features.
target = y_train

#####to run the decision boundary for just X[0] and X[1]
X = dataset[:,[0,1]]  #lets just take the first 2 features and make a decision tree. Then we can visualize boundary.
y = target
clf = DecisionTreeClassifier(max_depth=5).fit(X, y)
#plot the DT
c_names = {0:'healthy', 1:'covid'}
plot_tree(clf, filled=True,class_names=c_names)
plt.show()
plt.figure()

#plot the boundaries of DT
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

#basically, just run the decision classifier at each point of (x[0],x[1]) minimum to max.
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

plt.xlabel('X[0]')
plt.ylabel('X[1]')

# Plot the training points
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color,
                cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.title("Decision surface using feature 0 and 1")
#plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()


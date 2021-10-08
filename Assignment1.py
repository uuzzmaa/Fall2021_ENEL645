
import pandas as pd
from pomegranate import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix as cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import accuracy_score

import seaborn as sns

import matplotlib.pyplot as plt   # for plotting

import pandas as pd  # for data manipulation
import networkx as nx  # for drawing graphs
# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
############################################################
def bin2str(binary):
    return 'Negative' if binary == 0 else 'Positive'
############################################################
np.random.seed(0)  #makes the random numbers predictable, when we run, we have a similar result

#############################################################
# ------------------------------------------------------------
# ----------------C1 Preprocessing -------------------
# ------------------------------------------------------------
# a. Read the data from csv file
data = pd.read_csv('COVID-19_formatted_dataset.csv')  # read cvs file via pandas library


y = data['SARS-Cov-2 exam result'] # Y is my target variable


data.drop(data.columns[[0, 2]], axis=1, inplace=True)
#total_rows=len(data.axes[0]);  print(total_rows)
#total_cols=len(data.axes[1]); print(total_cols)

# we cannot work with strings... so we change target variable to 0 or 1
# c. Map the label
y = y.replace('negative', 0)
y = y.replace('positive', 1)
#print(y)
# d. form a numpy array:
x = data.values
y = y.values

# e. Split into training, testing sets
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

df = pd.read_csv('COVID-19_formatted_dataset.csv')

df = df.drop(['Unnamed: 0'], axis=1)

y_1 = df['SARS-Cov-2 exam result']
y_1 = df.loc[df['SARS-Cov-2 exam result'] == 'negative', 'SARS-Cov-2 exam result'] = 0
y_1 = df.loc[df['SARS-Cov-2 exam result'] == 'positive', 'SARS-Cov-2 exam result'] = 1
df.head(100)

df.to_numpy()
columns = list(df.columns)
features = set(columns) - set(['SARS-Cov-2 exam result'])
X_1, y_1 = df[features], df['SARS-Cov-2 exam result'].to_frame()
X_1, y_1 = X_1.astype('int'), y_1.astype('int')
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=0)

# ---------------------------------------------------------
# 3. histogram:
for i in features:
    # sns.displot(df, x=i, hue = df['SARS-Cov-2 exam result'], kind = "kde")
    sns.histplot(df, x=i, hue=df['SARS-Cov-2 exam result'])
    plt.show()


# ------------------------------------------------------------
# ------------------- Bayesian Networks---------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# 1.  Discretization:
disc_enc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
x_train_binned = disc_enc.fit_transform(x_train)
x_test_binned = disc_enc.transform(x_test)

train_data = np.concatenate([x_train_binned, y_train.reshape(-1, 1)], axis=1)
test_data = np.concatenate([x_test_binned, np.empty_like(y_test).reshape(-1, 1)], axis=1)
test_data[:, -1] = np.nan

print('---------------------------------------------------------')
print('Bayesian Network:\n')

print('Finding best DAG...')
model = BayesianNetwork.from_samples(train_data, algorithm='exact')
result = np.array(model.predict(test_data)).astype(int)
prediction = result[:, -1]

print('Confusion Matrix for features selected data:\n', cm(y_test, prediction), '\n')
print('test score: {:4.2f}%'.format(accuracy_score(y_test, prediction) * 100))

#Prediction probability on 5 arbitrary samples:
ind = np.random.randint(0, len(x_test), 5)
test_5sample = np.concatenate([x_test_binned[ind], y_test[ind].reshape(-1, 1)], axis=1)
test_5sample[:, -1] = np.nan
probs = model.predict_proba(test_5sample)

for i in range(5):
    print('Sample {}(actual: {}): Positive Probability: {:4.2f} %'
          .format(i, bin2str(y_test[ind[i]]), probs[i][-1].parameters[0][1] * 100))

# find the probability of having Covid-19 given input sample
sample = [[3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 2.0, 3.0, 1.0, 0.0, None]]
sample_result = np.array(model.predict(sample)).astype(int)
sample_prob = model.predict_proba(sample)
print('\nSample probability to be positive Covid-19 is {:4.2f}%'.format(sample_prob[0][-1].parameters[0][1]*100))
print('So sample is tagged as {}'.format(bin2str(sample_result[0, -1])))


# 3. Building Bayesian network and displaying the probabilities
# Read in the covid data csv
df = pd.read_csv('COVID-19_formatted_dataset.csv', encoding='utf-8')

# Create bands for variables that we want to use in the model
df['Hematocrit'] = df['Hematocrit'].apply(lambda x: 'high' if x > 0 else 'low')

df['Hemoglobin'] = df['Hemoglobin'].apply(lambda x: 'high' if x > 0 else 'low')
df['Lymphocytes'] = df['Lymphocytes'].apply(lambda x: 'high' if x > 0 else 'low')


# Show a snaphsot of data
# print(df)

# This function helps to calculate probability distribution, which goes into BBN (note, can handle up to 2 parents)
def probs(data, child, parent1=None, parent2=None):
    if parent1 == None:
        # Calculate probabilities
        prob = pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(
            -1).tolist()
    elif parent1 != None:
        # Check if child node has 1 parent or 2 parents
        if parent2 == None:
            # Caclucate probabilities
            prob = pd.crosstab(data[parent1], data[child], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()
        else:
            # Caclucate probabilities
            prob = pd.crosstab([data[parent1], data[parent2]], data[child], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else:
        print("Error in Probability Frequency Calculations")
    return prob


# Create nodes by using our earlier function to automatically calculate probabilities
hema = BbnNode(Variable(0, 'hema', ['low', 'high']), probs(df, child='Hematocrit'))
hemo = BbnNode(Variable(1, 'hemo', ['low', 'high']), probs(df, child='Hemoglobin', parent1='Hematocrit'))
lym = BbnNode(Variable(2, 'lym', ['low', 'high']), probs(df, child='Lymphocytes'))
covidpred = BbnNode(Variable(3, 'covidpred', ['No', 'Yes']),
                    probs(df, child='SARS-Cov-2 exam result', parent1='Hemoglobin', parent2='Lymphocytes'))

# Create Network
bbn = Bbn() \
    .add_node(hema) \
    .add_node(hemo) \
    .add_node(lym) \
    .add_node(covidpred) \
    .add_edge(Edge(hema, hemo, EdgeType.DIRECTED)) \
    .add_edge(Edge(hemo, covidpred, EdgeType.DIRECTED)) \
    .add_edge(Edge(lym, covidpred, EdgeType.DIRECTED))

# Convert the BBN to a join tree
join_tree = InferenceController.apply(bbn)

# Set node positions
pos = {0: (-1, 2), 1: (-1, 0.5), 2: (1, 0.5), 3: (0, -1)}

# Set options for graph looks
options = {
    "font_size": 16,
    "node_size": 4000,
    "node_color": "white",
    "edgecolors": "black",
    "edge_color": "red",
    "linewidths": 5,
    "width": 5, }

# Generate graph
n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, **options)

# Update margins and print the graph
ax = plt.gca()
ax.margins(0.10)
plt.axis("off")
plt.show()


# Define a function for printing marginal probabilities
def print_probs():
    for node in join_tree.get_bbn_nodes():
        potential = join_tree.get_bbn_potential(node)
        print("Node:", node)
        print("Values:")
        print(potential)
        print('----------------')


# Use the above function to print marginal probabilities
print_probs()

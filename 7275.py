#!/usr/bin/env python
# coding: utf-8

# test and traning 

# In[61]:


from sklearn.model_selection import train_test_split
import pandas as pd

# Load the processed data
file_path = '/Users/kat/Desktop/7275iris.csv'
df = pd.read_csv(file_path)

# Separate features and target variable
# Assuming the last column is the target variable ('species'), and the rest are features
X = df.drop('species', axis=1)  # Features
y = df['species']  # Target variable

# Split the dataset into training set and test set
# test_size indicates the proportion of the test set, random_state sets the seed for random splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Output the split results
print(f"Training set feature dimensions: {X_train.shape}")
print(f"Test set feature dimensions: {X_test.shape}")
print(f"Training set label dimensions: {y_train.shape}")
print(f"Test set label dimensions: {y_test.shape}")


# In[3]:


import pandas as pd

# Load data
df = pd.read_csv('/Users/kat/Desktop/7275iris.csv')

# Check for missing values in each column
print(df.isnull().sum())

# Remove rows with missing values
df_cleaned = df.dropna()

# Alternatively, fill missing values, for example, using the median of each column
df_filled = df.fillna(df.median())


# # EDA

# 1. Summary Statistics
# 
# Start with basic summary statistics to understand the distribution of your data

# In[5]:


# Summary statistics for numerical features
print(df.describe())

# For categorical data
print(df['species'].value_counts())


# 2. Data Visualization
# 
# Visualize the data to identify patterns, trends, and anomalies.
# 
# Histograms: Show the distribution of each numerical feature.

# In[6]:


import matplotlib.pyplot as plt
df.hist(bins=20, figsize=(14, 10))
plt.show()


# Box Plots: Identify outliers in the data.

# In[7]:


df.boxplot(by='species', figsize=(9, 9))
plt.show()


# Scatter Plots: Explore the relationship between pairs of features.
# 

# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
sns.pairplot(df)
plt.show()


# Correlation Matrix: Understand how features correlate with each other and the target variable.

# In[9]:


import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# Data Cleaning
# 
# Based on the insights gained, clean my data.
# 
# Handle missing values: Drop or impute missing values.
# Remove duplicates.
# Fix structural errors (typos, wrong units, etc.).

# In[10]:


df_no_duplicates = df.drop_duplicates()


# In[11]:


df


# In[12]:


import numpy as np


# In[13]:


from sklearn.preprocessing import StandardScaler

num_features = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])


# SVM
# 

# In[29]:


from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Initialize the SVM model
svm_model = SVC(probability=True)  # Set the probability parameter to True for ROC-AUC calculation

# Train the model
svm_model.fit(X_train, y_train)

# Predict on the test set
svm_predictions = svm_model.predict(X_test)

# ROC-AUC calculation requires probability values, we use the predict_proba method
svm_probs = svm_model.predict_proba(X_test)

# For multi-class problems, it's necessary to binarize y
y_test_bin = label_binarize(y_test, classes=np.unique(y))

# Calculate Precision, Recall, and F1 Score (using 'macro' averaging)
precision = precision_score(y_test, svm_predictions, average='macro')
recall = recall_score(y_test, svm_predictions, average='macro')
f1 = f1_score(y_test, svm_predictions, average='macro')

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test_bin, svm_probs, multi_class='ovr')  # 'ovr' stands for One-vs-Rest approach

# Print the results
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")


# 神经网络

# In[60]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

# Initialize the neural network model
nn_model = MLPClassifier(max_iter=1000)

# Train the model
nn_model.fit(X_train, y_train)

# Predict on the test set
nn_predictions = nn_model.predict(X_test)

# ROC-AUC calculation requires probability values, we use the predict_proba method
nn_probs = nn_model.predict_proba(X_test)

# For multi-class problems, it's necessary to binarize y
y_test_bin = label_binarize(y_test, classes=np.unique(y_train))

# Calculate Precision, Recall, and F1 Score (using 'macro' averaging)
precision = precision_score(y_test, nn_predictions, average='macro')
recall = recall_score(y_test, nn_predictions, average='macro')
f1 = f1_score(y_test, nn_predictions, average='macro')

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test_bin, nn_probs, multi_class='ovr', average='macro')

# Print the results
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")


# 决策树四个指标

# In[27]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer


# Initialize the decision tree model
tree_model = DecisionTreeClassifier()

# Train the model on the training data
tree_model.fit(X_train, y_train)

# Predict the labels for the test set
tree_predictions = tree_model.predict(X_test)

# Evaluate the model's Precision, Recall, and F1 Score on the test set
precision = precision_score(y_test, tree_predictions, average='macro')
recall = recall_score(y_test, tree_predictions, average='macro')
f1 = f1_score(y_test, tree_predictions, average='macro')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# For multi-class classification, calculating ROC-AUC is a bit more complex and requires binarizing the labels
lb = LabelBinarizer()
lb.fit(y_test)
y_test_bin = lb.transform(y_test)
y_pred_bin = lb.transform(tree_predictions)

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test_bin, y_pred_bin, average='macro')

print(f"ROC-AUC: {roc_auc}")


# 使用 GridSearchCV 调优神经网络模型

# In[31]:


from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50,50), (100,100)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

# Initialize grid search
grid_search = GridSearchCV(MLPClassifier(max_iter=1000), param_grid, n_jobs=-1, cv=5)

# Execute the grid search
grid_search.fit(X_train, y_train)

# View the best parameter combination
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)


# 使用 RandomizedSearchCV 调优决策树模型

# In[32]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint

# Define the parameter distribution
param_dist = {
    "max_depth": [3, None],  # Depth of the tree
    "max_features": randint(1, 9),  # The number of features to consider when looking for the best split
    "min_samples_leaf": randint(1, 9),  # The minimum number of samples required to be at a leaf node
    "criterion": ["gini", "entropy"]  # The function to measure the quality of a split
}

# Initialize the random search
random_search = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1)

# Execute the random search
random_search.fit(X_train, y_train)

# View the best parameter combination
print("Best parameters found: ", random_search.best_params_)
print("Best score found: ", random_search.best_score_)


# svm best score

# In[33]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],  
    'gamma': [1, 0.1, 0.01, 0.001],  
    'kernel': ['rbf', 'poly', 'sigmoid']  
}

# Initialize the grid search object
grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=2)

# Execute the grid search on the training set
grid_search.fit(X_train, y_train)

# View the best parameter combination
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)


# In[ ]:


# calculate svm
calculate_evaluation_metrics(y_test, svm_predictions, svm_model.predict_proba(X_test))


# In[48]:


# Initialize the Neural Network model with the optimal hyperparameters
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

# Initialize the model, replacing the optimal hyperparameter values
nn_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=1000)

# Train the model
nn_model.fit(X_train, y_train)

# Predict on the test set
nn_predictions = nn_model.predict(X_test)
nn_probs = nn_model.predict_proba(X_test)  # Probability predictions needed for ROC-AUC

# Binarize y_test for multi-class ROC-AUC calculation
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

# Calculate performance metrics
precision = precision_score(y_test, nn_predictions, average='macro')
recall = recall_score(y_test, nn_predictions, average='macro')
f1 = f1_score(y_test, nn_predictions, average='macro')
roc_auc = roc_auc_score(y_test_binarized, nn_probs, multi_class='ovr', average='macro')

# Print performance metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")


# In[49]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Define the hyperparameter grid
param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ["gini", "entropy"]
}

# Initialize the Decision Tree model
dt = DecisionTreeClassifier()

# Initialize the grid search
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Perform grid search on the training set
grid_search.fit(X_train, y_train)

# View the best parameter combination
print("Best Parameters:", grid_search.best_params_)


# In[50]:


# Initialize the Decision Tree model with the best parameters
best_dt = DecisionTreeClassifier(**grid_search.best_params_)

# Retrain the model
best_dt.fit(X_train, y_train)

# Make predictions
predictions = best_dt.predict(X_test)


# In[51]:


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Binarize y_test for multi-class ROC-AUC calculation
y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))

# Calculate metrics
precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')
roc_auc = roc_auc_score(y_test_binarized, best_dt.predict_proba(X_test), multi_class='ovr', average='macro')

# Print performance metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")


# In[54]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

# Assume the optimal parameters
max_depth = 3
min_samples_split = 2
min_samples_leaf = 1
criterion = 'entropy'

# Initialize the Decision Tree model with the optimal hyperparameters
best_tree_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf, criterion=criterion)

# Retrain the model
best_tree_model.fit(X_train, y_train)

# Make predictions
tree_predictions = best_tree_model.predict(X_test)

# Since Decision Trees support predict_proba, we can directly use it to compute ROC-AUC
tree_probs = best_tree_model.predict_proba(X_test)

# Binarize y_test for multi-class ROC-AUC calculation
y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))

# Calculate precision, recall, and F1 score (using 'macro' averaging)
precision = precision_score(y_test, tree_predictions, average='macro')
recall = recall_score(y_test, tree_predictions, average='macro')
f1 = f1_score(y_test, tree_predictions, average='macro')

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test_binarized, tree_probs, multi_class='ovr', average='macro')

# Print performance metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")


# In[55]:


from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

# Assume the optimal parameters
C = 1.0
gamma = 'scale'
kernel = 'rbf'

# Initialize SVM model with the optimal hyperparameters
# Note: SVC by default does not compute probabilities, so we need to set probability=True
best_svm_model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True)

# Retrain the model
best_svm_model.fit(X_train, y_train)

# Make predictions
svm_predictions = best_svm_model.predict(X_test)

# Calculate probability predictions for ROC-AUC
svm_probs = best_svm_model.predict_proba(X_test)

# Binarize y_test for multi-class ROC-AUC calculation
y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))

# Calculate precision, recall, and F1 score (using 'macro' averaging)
precision = precision_score(y_test, svm_predictions, average='macro')
recall = recall_score(y_test, svm_predictions, average='macro')
f1 = f1_score(y_test, svm_predictions, average='macro')

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test_binarized, svm_probs, multi_class='ovr', average='macro')

# Print performance metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")


# In[56]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

# Assume the optimal parameters
hidden_layer_sizes = (100,)
activation = 'relu'
solver = 'adam'
alpha = 0.0001

# Initialize MLP model with the optimal hyperparameters
best_mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, 
                               solver=solver, alpha=alpha, max_iter=1000)

# Retrain the model
best_mlp_model.fit(X_train, y_train)

# Make predictions
mlp_predictions = best_mlp_model.predict(X_test)

# Calculate probability predictions for ROC-AUC
mlp_probs = best_mlp_model.predict_proba(X_test)

# Binarize y_test for multi-class ROC-AUC calculation
y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))

# Calculate precision, recall, and F1 score (using 'macro' averaging)
precision = precision_score(y_test, mlp_predictions, average='macro')
recall = recall_score(y_test, mlp_predictions, average='macro')
f1 = f1_score(y_test, mlp_predictions, average='macro')

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test_binarized, mlp_probs, multi_class='ovr', average='macro')

# Print performance metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")


# In[ ]:





# In[ ]:





import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from random import sample, seed
from re import sub
from decimal import Decimal
import math
 
 
 
class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data
 
def getEntropy(counts):
    total = sum(counts)
    probabilities = [count / total for count in counts]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)
 
def getWeightedEntropy(subsets):
    total = sum(sum(subset) for subset in subsets)
    weighted_entropy = sum((sum(subset) / total) * getEntropy(subset) for subset in subsets)
    return weighted_entropy
 
def chooseAttribute(xtrain, ytrain, features, num_features):
    n_classes = np.unique(ytrain).size
    best_attribute = None
    lowest_entropy = float('inf')
 
    num_features = min(num_features, len(features))
    selected_features = sample(features, num_features)
 
    for feature in selected_features:
        subsets = []
 
        for class_value in np.unique(ytrain):
            class_counts = []
            for feature_value in [0, 1]:
                count = np.sum((xtrain[:, feature] == feature_value) & (ytrain == class_value))
                class_counts.append(count)
            subsets.append(class_counts)
 
        entropy = getWeightedEntropy(subsets)
        if entropy < lowest_entropy:
            best_attribute = feature
            lowest_entropy = entropy
 
    return best_attribute
 
def MODE(ytrain):
    values, counts = np.unique(ytrain, return_counts=True)
    return values[np.argmax(counts)]
 
def DTL(xtrain, ytrain, features, default, num_features):
    if len(xtrain) == 0:
        return Node(data=default)
    elif np.all(ytrain == ytrain[0]):
        return Node(data=ytrain[0])
    elif len(features) == 0:
        return Node(data=default)
    else:
        best = chooseAttribute(xtrain, ytrain, features, num_features)
        tree = Node(data=best)
        features = [f for f in features if f != best]
        for value in [0, 1]:
            indices = xtrain[:, best] == value
            x_subset = xtrain[indices]
            y_subset = ytrain[indices]
            if len(y_subset) == 0:
                subtree = Node(data=default)
            else:
                subtree_default = MODE(y_subset)
                subtree = DTL(x_subset, y_subset, features, subtree_default, num_features)
            if value == 0:
                tree.left = subtree
            else:
                tree.right = subtree
        return tree
 
def predict(tree, instance, features):
    if tree.left is None and tree.right is None:
        return tree.data
    feature_index = features.index(tree.data)
    if instance[feature_index] == 1:
        return predict(tree.right, instance, features)
    else:
        return predict(tree.left, instance, features)
 
def predictAll(tree, xvalid, features):
    predictions = [predict(tree, instance, features) for instance in xvalid]
    return np.array(predictions)
 
def myDT(xtrain, ytrain, xvalid):
    features = list(range(xtrain.shape[1]))
    default_class = MODE(ytrain)
    tree = DTL(xtrain, ytrain, features, default_class)
    predictions = predictAll(tree, xvalid, features)
    return predictions
 


def RandomForest(XTrain, YTrain, X_val, numberOfTrees=10, numberOfFeatures=None):
    if numberOfFeatures is None:
        numberOfFeatures = int(math.sqrt(len(XTrain[0])))
 
    trees = []
    for x in range(numberOfTrees):
        indices = np.random.randint(len(XTrain), size=len(XTrain))
        XSample = XTrain[indices]
        YSample = YTrain[indices]
 
        tree = DTL(XSample, YSample, list(range(XTrain.shape[1])), MODE(YSample),numberOfFeatures)
        trees.append(tree)
 
    return trees


 
def RandomForestPredict(trees, X_val):

    treePredictionsArray = []
    sumOfPredictions = []

    for tree in trees:
        treePredictions = predictAll(tree, X_val, list(range(X_val.shape[1])))
        treePredictionsArray.append(treePredictions)


    for preds in np.array(treePredictionsArray).T:
        sumOfPredictions.append(np.argmax(np.bincount(preds)))

    sumOfPredictions = np.array(sumOfPredictions)
    return sumOfPredictions


 
df = pd.read_csv("D:/Drexel Quarter Studies/Quarter - 4/CS 613/Final Project/Billionaires Statistics Dataset.csv")
df = df[['finalWorth', 'birthYear',  'gender', 'country', 'industries',
         'cpi_country', 'cpi_change_country', 'gdp_country',
         'gross_tertiary_education_enrollment',
         'gross_primary_education_enrollment_country',
         'population_country', 'life_expectancy_country']]
 
 
label_encoder = LabelEncoder()
df['country'] = label_encoder.fit_transform(df['country']).astype(float)
df['industries'] = label_encoder.fit_transform(df['industries']).astype(float)
df['gender'] = label_encoder.fit_transform(df['gender']).astype(float)
 
df['gdp_country'] = df['gdp_country'].apply(lambda x: np.nan if pd.isnull(x) else Decimal(sub(r'[^\d.]', '', str(x))) if str(x).strip() else np.nan)
df['gdp_country'] = pd.to_numeric(df['gdp_country'])
df['population_country'] = pd.to_numeric(df['population_country'])
df['gdp_per_capita'] = df['gdp_country'] / df['population_country']
 
for col in ['birthYear', 'country', 'industries',
         'cpi_country', 'cpi_change_country', 'gdp_country',
         'gross_tertiary_education_enrollment',
         'gross_primary_education_enrollment_country',
         'population_country', 'life_expectancy_country', 'gdp_per_capita']:
    median_val = df[col].median()
    df[col] = np.where(df[col] > median_val, 1, 0)
 
 
df['finalWorth'] = pd.cut(df['finalWorth'],
                          bins=[float('-inf'), 1500.0, 2300.0, 4200.0, float('inf')],
                          labels=[0, 1, 2, 3])
 
 
df.dropna(inplace=True)
 
X = df.drop('finalWorth', axis=1)
y = df['finalWorth'].astype(int)
 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, shuffle = True, random_state=42)
 
numberOfTrees = 10  
numberOfFeatures = int(math.sqrt(X_train.shape[1]))  
 
trees = RandomForest(X_train.values, y_train.values, X_val.values, numberOfTrees, numberOfFeatures)
predictions = RandomForestPredict(trees, X_val.values)
 
accuracy = accuracy_score(y_val, predictions)
precision = precision_score(y_val, predictions, average='weighted')
recall = recall_score(y_val, predictions, average='weighted')
f1 = f1_score(y_val, predictions, average='weighted')
confusionMatrix = confusion_matrix(y_val, predictions)
 
print(f'Validation Accuracy: {round(accuracy,4) * 100}%')
print(f'Precision: {round(precision,4) * 100}%')
print(f'Recall: {round(recall,4) * 100}%')
print(f'F1 Score: {round(f1,4) * 100}%')
print('Confusion Matrix:')
print(confusionMatrix)
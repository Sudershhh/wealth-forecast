import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from re import sub
from decimal import Decimal

# Calculate class priors
def getClassPrior(ytrain):
  # initialize a dictionary to store all class priors
  class_counts = {}
  for class_value in np.unique(ytrain):
    # we get the total number of appearance for each class
    count = np.sum(ytrain == class_value)
    class_counts[class_value] = count

  # computing class prior
  total = sum(class_counts.values())
  for k in class_counts.keys():
    class_counts[k] = class_counts[k] / total
  return class_counts

# Calculate generative likelihood
def getGenerative(xtrain, ytrain, features):
  # initialize a dictionary to store all generative likelihood
  gen_likelihood = {}
  # iterate over all classes
  for class_value in np.unique(ytrain):
      gen_likelihood[class_value] = {}
      # getting the num of classes
      class_count = np.sum(ytrain == class_value)
      for feature_id in features:
          gen_likelihood[class_value][feature_id] = {}
          # for each feature we get its generative likelihood given the class
          for feature_value in [0, 1]:
              feature_count = np.sum((xtrain[:, feature_id] == feature_value) & (ytrain == class_value))
              feature_prob = feature_count/class_count
              gen_likelihood[class_value][feature_id][feature_value] = feature_prob
  return gen_likelihood

def predict(xvalid, classPrior, genLikelihood):
    res = []
    # iterate through each row of the validation samples
    for r in xvalid:
        probablity = {}
        for cls in classPrior.keys():
            probablity[cls] = classPrior[cls]
            for feature_id in range(len(r)):
                feature_value = r[feature_id]  # getting the feature value
                # multiplying the probabilities for each feature
                probablity[cls] *= genLikelihood[cls][feature_id][feature_value]

        # Normalize the probabilities so they sum to 1
        total_prob = sum(probablity.values())
        for cls in probablity:
            probablity[cls] /= total_prob

        # Get the class with the highest posterior probability
        max_prob_cls = max(probablity, key=probablity.get)
        res.append(max_prob_cls)

    return np.asarray(res)


# Load and preprocess the dataset
df = pd.read_csv("dataset.csv")
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

# Binarizing the continuous features
for col in ['birthYear', 'country', 'industries',
         'cpi_country', 'cpi_change_country', 'gdp_country',
         'gross_tertiary_education_enrollment',
         'gross_primary_education_enrollment_country',
         'population_country', 'life_expectancy_country', 'gdp_per_capita']:
    median_val = df[col].median()
    df[col] = np.where(df[col] > median_val, 1, 0)


# Handling the target variable
df['finalWorth'] = pd.cut(df['finalWorth'],
                          bins=[float('-inf'), 1500.0, 2300.0, 4200.0, float('inf')],
                          labels=[0, 1, 2, 3])


# Splitting the dataset into features and target variable
X = df.drop('finalWorth', axis=1)
y = df['finalWorth'].astype(int)

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, shuffle = True, random_state=42)

# Applying Naive Bayes Classification
genLikelihood = getGenerative(X_train.values, y_train.values, list(range(X_train.shape[1])))
classPrior = getClassPrior(y_train.values)
y_pred = predict(X_val.values, classPrior, genLikelihood)

# Compute various metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')
cf_matrix = confusion_matrix(y_val, y_pred)

# Print the results
print(f'Validation Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('Confusion Matrix:')
print(cf_matrix)


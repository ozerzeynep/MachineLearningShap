import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from boruta import BorutaPy
from scipy import stats
from scipy.special import softmax
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score as R2
from sklearn.datasets import load_diabetes

#Boruta-SHAP Feature Selection Method
def print_feature_importances_random_forest(random_forest_model):
    
    '''
    Prints the feature importances of a Random Forest model in an ordered way.
    random_forest_model -> The sklearn.ensemble.RandomForestRegressor or RandomForestClassifier trained model
    '''
    
    # Fetch the feature importances and feature names
    importances = random_forest_model.feature_importances_
    features = random_forest_model.feature_names_in_
    
    # Organize them in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    
    return feature_importances


def print_feature_importances_shap_values(shap_values, features):
    
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    '''

    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
        
    # Calculates the normalized version
    importances_norm = softmax(importances)

    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}

    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}

    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

def evaluate_regression(y, y_pred):
    
    '''
    Prints the most common evaluation metrics for regression
    '''
    
    mae = MAE(y, y_pred)
    mse = MSE(y, y_pred)
    rmse = mse ** (1/2)
    r2 = R2(y, y_pred)
    
    print('Regression result')
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")



from sklearn.datasets import load_diabetes
# Fetches the data
dataset = load_diabetes(as_frame = True)

# Gets the independent variables
X = dataset['data']
X.head(5)

# Checks the shape of the data
X.shape

# Gets the dependent variable (the target)
y = dataset['target']
y.head(5)

# Splits the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#In order to compare results before and after applying Boruta Shap feature selection, we will fisrt run a simple regression
# Prepares a default instance of the random forest regressor
model = RandomForestRegressor()

# Fits the model on the training data
model.fit(X_train, y_train)

# Evaluates the model
y_pred = model.predict(X_test)
evaluate_regression(y_test, y_pred)

# Prints the feature importances
feature_importances = print_feature_importances_random_forest(model)



# Dönüştürülen sözlüğü ekrana yazdırma
print(feature_importances)

# Feature isimleri ve önem skorları
feature_names = list(feature_importances.keys())
importance_scores = list(feature_importances.values())

# Önem skorlarına göre sıralama
sorted_scores, sorted_names = zip(*sorted(zip(importance_scores, feature_names), reverse=False))


# Grafiğin boyutunu ayarlayın
plt.figure(figsize=(12, 6))

# Sıralanmış bar grafiğini yatay olarak oluşturun
plt.barh(sorted_names, sorted_scores, color='skyblue')

# X ve Y eksen etiketlerini ve başlığını ayarlayın
plt.xlabel('Önem Skorları')
plt.ylabel('Özellikler')
plt.title('Random Forest Feature Importances (Descending)')

# Her çubuğun üzerine önemini yazdırın
for i, v in enumerate(sorted_scores):
    plt.text(v + 0.03, i, f"{v:.3f}", ha='center', va='center', fontsize=12)

# Grafiği gösterin
plt.show()

#SHAP evaluation
# Fits the explainer
explainer = shap.Explainer(model.predict, X)

# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test)

# Prints the SHAP feature importances
print_feature_importances_shap_values(shap_values, X.columns)

# Plots this view
shap.plots.bar(shap_values)

# Plots the beeswarm
shap.plots.beeswarm(shap_values)


#BORUTA---------------
#Select features using Boruta
# Defines the estimator used by the Boruta algorithm
estimator = RandomForestRegressor()

np.int = np.int32
np.float = np.float64
np.bool = np.bool_

# Creates the BorutaPy object
boruta = BorutaPy(estimator = estimator, n_estimators = 'auto', max_iter = 100)

boruta.fit(np.array(X_train), np.array(y_train))
#boruta.fit(X_train.values, y_train.values)
# Important features
important = list(X.columns[boruta.support_])
print(f"Features confirmed as important: {important}")

# Tentative features
tentative = list(X.columns[boruta.support_weak_])
print(f"Unconfirmed features (tentative): {tentative}")

# Unimportant features
unimportant = list(X.columns[~(boruta.support_ | boruta.support_weak_)])
print(f"Features confirmed as unimportant: {unimportant}")


X_train_boruta = boruta.transform(np.array(X_train))
X_train_boruta

#Select features using Boruta-SHAP
# Creates a BorutaShap selector for regression
selector = BorutaShap(importance_measure = 'shap', classification = False)

# Fits the selector
selector.fit(X = X_train, y = y_train, n_trials = 100, sample = False, verbose = True)
# n_trials -> number of iterations for Boruta algorithm
# sample -> samples the data so it goes faster

# Display features to be removed
features_to_remove = selector.features_to_remove
print(features_to_remove)


# Removes them
X_train_boruta_shap = X_train.drop(columns = features_to_remove)
X_test_boruta_shap = X_test.drop(columns = features_to_remove)



X_train_boruta_shap.head()


#Fits a new regression model to the new data
# Prepares a default instance of the random forest regressor
model_new = RandomForestRegressor()

# Fits the model on the data
model_new.fit(X_train_boruta_shap, y_train)

# Evaluates the model
y_pred = model_new.predict(X_test_boruta_shap)
evaluate_regression(y_test, y_pred)

# Prints the feature importances
print_feature_importances_random_forest(model_new)

# Fits the explainer
explainer_new = shap.Explainer(model_new.predict, X_test_boruta_shap)

# Calculates the SHAP values - It takes some time
shap_values = explainer_new(X_test_boruta_shap)

# Prints the SHAP feature importances
print_feature_importances_shap_values(shap_values, X_test_boruta_shap.columns)

# Plots this view
shap.plots.bar(shap_values)

# Plots the beeswarm
shap.plots.beeswarm(shap_values, max_display=14)


#Plots the binomial distribution
#Plots the binomial distribution
n = 20
p = 0.5

pmf = list(stats.binom.pmf(range(n + 1), n, p))


plt.figure(figsize = (12, 5))
sns.scatterplot(x = range(n+1), y = pmf)
sns.lineplot(x = range(n+1), y = pmf)


pmf

# In which "point" will we have a cumulative probability of 0.5%?
stats.binom.ppf(0.005, n, p)

# Confirming:
stats.binom.cdf(4, n, p)


# In which "point" will we have a cumulative probability of 99.5%?
stats.binom.ppf(0.995, n, p)


# Confirming:
stats.binom.cdf(16, n, p)


#Final plot
red_border = int(stats.binom.ppf(0.005, n, p))
red_zone = pmf[:red_border+1]

green_border = int(stats.binom.ppf(0.995, n, p))
green_zone = pmf[green_border:]


blue_zone = pmf[red_border:green_border+1]

plt.figure(figsize = (12, 5))
# Scatters
sns.scatterplot(x = range(0, red_border+1), y = red_zone, color = 'r', s = 120,  linewidth = 0)
sns.scatterplot(x = range(red_border, green_border+1), y = blue_zone, color = 'b', s = 120,  linewidth = 0)
sns.scatterplot(x = range(green_border, n+1), y = green_zone, color = 'g', s = 120,  linewidth = 0)
# Lines
sns.lineplot(x = range(0, red_border+1), y = red_zone, color = 'r', linewidth = 3)
sns.lineplot(x = range(red_border, green_border+1), y = blue_zone, color = 'b', linewidth = 3)
sns.lineplot(x = range(green_border, n+1), y = green_zone, color = 'g', linewidth = 3)
# Vertical lines
for x, y in zip(range(0, red_border+1), red_zone):
    plt.plot([x, x], [0, y], color = 'r')
for x, y in zip(range(red_border, green_border+1), blue_zone):
    plt.plot([x, x], [0, y], color = 'b')
for x, y in zip(range(green_border, n+1), green_zone):
    plt.plot([x, x], [0, y], color = 'g')
# Plot config
plt.xticks(range(n+1))
plt.title("Binomial distribution. p = 0.5; n = 20")


#Table for the article
pd.DataFrame(index = ["hits"], data = {"genre" : 3, "audience_score" : 14, "critic_score"  : 20})

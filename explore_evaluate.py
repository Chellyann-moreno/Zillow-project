# IMPORTS
import pandas as pd 
import env as env
import os
import wrangle as w

# data visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# stats data 
import scipy.stats as stats
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp, ttest_ind,f_oneway
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, LassoLarsCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
alpha=0.05


# function to plot cont target variable
def plot_continuous_target(y):
    "Function to plot target variable"
   
    # Plot the continuous target variable
    plt.figure(figsize=(8, 6))
    plt.hist(y)
    plt.xlabel('Target Variable')
    plt.ylabel('Frequency')
    plt.title('Distribution of Continuous Target Variable')
    plt.show()


#Function plot cont and cat variables
def plot_categorical_and_continuous_vars(df, cat_var, cont_var):
    for var in cont_var:
        # Create a box plot
        plt.figure(figsize=(12, 6))
        sns.violinplot(x=cat_var, y=var, data=df)
        plt.axhline(y=df[var].mean(), color='r', linestyle='--')
        plt.xlabel(cat_var)
        plt.ylabel(var)
        plt.title(f'{cat_var} vs. {var}')
        plt.show()


# Function to plot cat variable

def plot_variable_pairs(df, cols):
    " Function to plot variables"
    sns.pairplot(df[cols], kind='reg')
    plt.show()

# function to scale data:

def robust_scale_data(X_train, X_validate,X_test):
    """this function could be used to scaled your features prior to feature
    """
    # Initialize RobustScaler object
    scaler = RobustScaler()
    
    # Fit scaler object to training data
    scaler.fit(X_train)
    
    # Transform training and validation data
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)
    # Return scaled data
    return X_train_scaled, X_validate_scaled, X_test_scaled

# function to calculate regression model:

def calculate_regression_results(X_train, y_train, X_test, y_test):
    # Calculate baseline (yhat)
    yhat = np.mean(y_train)

    # Initialize dataframe to store results
    results_df = pd.DataFrame(columns=['Model', 'Alpha', 'Degree', 'RMSE', 'R2'])

    # Ordinary Least Squares (OLS) regression
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    ols_y_pred = ols.predict(X_test)
    ols_rmse = np.sqrt(mean_squared_error(y_test, ols_y_pred))
    ols_r2 = r2_score(y_test, ols_y_pred)
    results_df = results_df.append({'Model': 'OLS', 'Alpha': None, 'Degree': None, 'RMSE': ols_rmse, 'R2': ols_r2}, ignore_index=True)

    # LassoLARS regression with different alphas
    alphas = [0.1, 0.01, 0.001]
    for alpha in alphas:
        lasso = LassoLarsCV(cv=5, max_iter=10000, eps=0.1, normalize=True, precompute='auto')
        lasso.fit(X_train, y_train)
        lasso_y_pred = lasso.predict(X_test)
        lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_y_pred))
        lasso_r2 = r2_score(y_test, lasso_y_pred)
        results_df = results_df.append({'Model': 'LassoLARS', 'Alpha': alpha, 'Degree': None, 'RMSE': lasso_rmse, 'R2': lasso_r2}, ignore_index=True)

    # Polynomial regression with different degrees
    degrees = [1, 2, 3, 4]
    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        poly_reg = LinearRegression()
        poly_reg.fit(X_train_poly, y_train)
        poly_y_pred = poly_reg.predict(X_test_poly)
        poly_rmse = np.sqrt(mean_squared_error(y_test, poly_y_pred))
        poly_r2 = r2_score(y_test, poly_y_pred)
        results_df = results_df.append({'Model': 'Polynomial Regression', 'Alpha': None, 'Degree': degree, 'RMSE': poly_rmse, 'R2': poly_r2}, ignore_index=True)

    return results_df


# Stats test two cont variables:
def perform_statistical_tests(variable1, variable2, alpha):
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(variable1, variable2)
    print("T-Test Results:")
    print(f"T-Statistic: {t_statistic}")
    print(f"P-Value: {p_value}")
    if p_value < alpha:
        print('We reject the null hypothesis.')
    else:
        print('We fail to reject the null hypothesis.')

    # Perform Mann-Whitney U test
    u_statistic, p_value = stats.mannwhitneyu(variable1, variable2, alternative='two-sided')
    print("\nMann-Whitney U Test Results:")
    print(f"U-Statistic: {u_statistic}")
    print(f"P-Value: {p_value}")
    if p_value < alpha:
        print('We reject the null hypothesis.')
    else:
        print('We fail to reject the null hypothesis.')

    # Perform Pearson correlation test
    pearson_corr, p_value = stats.pearsonr(variable1, variable2)
    print("\nPearson Correlation Test Results:")
    print(f"Pearson Correlation Coefficient: {pearson_corr}")
    print(f"P-Value: {p_value}")
    if p_value < alpha:
        print('We reject the null hypothesis.')
    else:
        print('We fail to reject the null hypothesis.')

    # Perform Spearman correlation test
    spearman_corr, p_value = stats.spearmanr(variable1, variable2)
    print("\nSpearman Correlation Test Results:")
    print(f"Spearman Correlation Coefficient: {spearman_corr}")
    print(f"P-Value: {p_value}")
    if p_value < alpha:
        print('We reject the null hypothesis.')
    else:
        print('We fail to reject the null hypothesis.')



# function to perform tast test on a cont and cat variable

def statistical_tests_cont_cat(variable_continuous, variable_categorical, alpha=0.05):
    # Perform t-test or Mann-Whitney U test based on the number of categories in the categorical variable
    unique_categories = variable_categorical.unique()
    num_categories = len(unique_categories)

    if num_categories == 2:
        category1 = variable_continuous[variable_categorical == unique_categories[0]]
        category2 = variable_continuous[variable_categorical == unique_categories[1]]

        t_statistic, p_value = stats.ttest_ind(category1, category2)
        test_type = "T-Test"
    else:
        categories = [variable_continuous[variable_categorical == category] for category in unique_categories]
        statistic, p_value = stats.kruskal(*categories)
        t_statistic = statistic
        test_type = "Kruskal-Wallis Test"

    print(f"{test_type} Results:")
    print(f"Test Statistic: {t_statistic}")
    print(f"P-Value: {p_value}")

    if p_value < alpha:
        print("We reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")

    # Perform ANOVA or Kruskal-Wallis test based on the number of categories in the categorical variable
    if num_categories > 2:
        if num_categories <= 10:  # Perform ANOVA test
            groups = [variable_continuous[variable_categorical == category] for category in unique_categories]
            f_statistic, p_value = stats.f_oneway(*groups)
            test_type = "ANOVA"
        else:  # Perform Kruskal-Wallis test
            groups = [variable_continuous[variable_categorical == category] for category in unique_categories]
            statistic, p_value = stats.kruskal(*groups)
            f_statistic = statistic
            test_type = "Kruskal-Wallis Test"

        print(f"\n{test_type} Results:")
        print(f"Test Statistic: {f_statistic}")
        print(f"P-Value: {p_value}")

        if p_value < alpha:
            print("We reject the null hypothesis.")
        else:
            print("We fail to reject the null hypothesis.")


def metrics_reg(y, yhat):
    """
    function calculate RMSE, R2 by Misty
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

    

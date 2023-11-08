import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt


# EDA
def agg_by_count(df: pd.DataFrame, target: str, group_var: str, count: int = None,
                 agg_list: list = None, agg_str: str = "other_agg",
                 return_agg_list: bool = False):
    """Combine categories of a variable to a single value (the agg_str), given the count
    Or, combine categories if provided the agg_list

    Args:
        df (pd.DataFrame): dataset to manipulate
        target (str): variable to count over
        group_var (str): variable with categories to change
        count (int): threshold value where if value <= count, become aggregated category. Defaults to None
        agg_list (list, optional): list of all categories being combined. If None, will determine aggregate value. Defaults to None.
        agg_str (str, optional): string used to represent aggregated categories. Defaults to "other_agg".
        return_agg_list (bool, optional): returns the list of categories to be aggregated
    """
    # Find all values where the count is under a specific value, and aggregate the data together
    data = df.copy()

    # Find categories if not provided
    if agg_list is None:
        agg_list = data.groupby([group_var])[target].count().loc[lambda x: x <= count].index

    data.loc[data[group_var].isin(agg_list), group_var] = agg_str

    if return_agg_list:
        return data, agg_list

    else:
        return data


def impute_cols(x_train: pd.DataFrame, x_test: pd.DataFrame, cols: list, imputation: str):
    """Input missing values for specified columns of both dataframes with a specific value

    Args:
        x_train (pd.DataFrame): Training dataset
        x_test (pd.DataFrame): Test dataset
        cols (list): list of columns that are imputed with a specific value
        imputation (str): Methodology type
    """
    train = x_train.copy()
    test = x_test.copy()

    train[cols].fillna(imputation, inplace=True)
    test[cols].fillna(imputation, inplace=True)

    return train, test


# Plotting
def plot_shap_summary(
    shap_list, data, title_loc: str = None, title: str = None, class_names: list = None
):
    """Plot shaply values, either a stacked barplot for all tiers
    or the summary plot for a specific class

    Args:
        shap_list (list): list of shap values for each feature,
            derived from shap.TreeExplainer(model.best_estimator_).shap_values(data)
        data (pd.DataFrame): test data used to determine shap values
        title_loc (str, optional): string for location of title, either "left", "center", or "right.
            Defaults to None.
        title (str, optional): title for graph. Defaults to None.
        class_names (list, optional): list of names for each class. Defaults to None.
    """
    shap.summary_plot(shap_list, data, show=False, class_names=class_names)
    plt.title(title, loc=title_loc, size=20)
    plt.yticks(rotation=0)
    plt.show()


# Pickle functions
def save_pickle(object, path_name: str):
    """Function to store an object as a pickle

    Args:
        object: object meant to be stored
        path_name (str): string to name the pickle, including the path to the storage
    """
    with open(path_name, "wb") as pickle_out:
        pickle.dump(object, pickle_out)


def load_pickle(path_name: str):
    """Load pickle given filepath to pickle object

    Args:
        path_name (str): path to pickle
    """
    with open(path_name, "rb") as pickle_in:
        return pickle.load(pickle_in)
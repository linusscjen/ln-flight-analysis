import pandas as pd
from plotnine import *


def find_percent(data: pd.DataFrame, group_col: str, target_col: str):
    """Find percent of categories where target_col is True

    Args:
        data (pd.DataFrame): dataframe, must have the ``group_col`` and ``target_col`` columns
        group_col (str): categorical column of data
        target_col (str): target binary column of data

    Returns:
        pd.DataFrame: dataframe
    """
    # Find the total number of target and count for each category
    data_agg = data.groupby(group_col)[target_col].agg(["sum", "count"])

    # Calculate the percentage of target for each category
    data_agg["percent_late"] = round(data_agg["sum"] / data_agg["count"] * 100, 3)

    return data_agg


def plot_cdf(data: pd.DataFrame, num_col: str, group_col: str, title: str = ""):
    """Plot CDF of a numerical column, grouped by ``group_col``

    Args:
        data (pd.DataFrame): dataset with both ``num_col`` and ``group_col`` columns
        num_col (str): column of numerical values to plot in the CDF
        group_col (str): categorical column to group by for CDF
        title (str, optional): Title of plot. Defaults to "".
    """
    p = (
        ggplot(data, aes(x=num_col, group=group_col, color=group_col))
        + stat_ecdf(geom="step")
        + theme_minimal()
        + labs(title=title, y="Cumulative Distribution Function")
    )
    print(p)

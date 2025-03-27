from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error, r2_score,
    root_mean_squared_log_error
)
import matplotlib.pyplot as plt


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    '''This function takes in a pd dataframe, and returns a pd dataframe that describes
    each of the columns within the original input dataframe.'''

    column_descriptions = []
    for column in df.columns:
        column_data = df[column]
        description = {
            'column_name': column,
            'dtype': column_data.dtype,
            'non_null_count': column_data.notna().sum(),
            'null_count': column_data.isna().sum(),
            'unique_count': column_data.nunique(),
            'sample_values': list(column_data.dropna().unique())[:5]
        }
        
        if np.issubdtype(column_data.dtype, np.number):
            description['min'] = column_data.min()
            description['max'] = column_data.max()
            description['mean'] = column_data.mean()
        else:
            description['min'] = "None"
            description['max'] = "None"
            description['mean'] = "None"
        
        column_descriptions.append(description)

    return pd.DataFrame(column_descriptions)

def get_cont_enrolled(init_year, end_year, df):
    '''Inputs a dataframe of CMS inpatient data and date range, and returns a dictionary of how many unique
    id's there are per year, as well as a dictionary of those unique id's per year'''

    select_years = df[df["YR"] < end_year]

    current_ids = select_years[select_years['YR'] == init_year]['BENE_ID'].unique()
    num_ids = [len(current_ids)]
    current_id_dict = {str(init_year): list(current_ids)}
    
    for year in range(init_year + 1, end_year):
        current_ids = select_years[(select_years["BENE_ID"].isin(current_ids)) & 
                                   (select_years['YR'] == year)]["BENE_ID"].unique()
        num_ids.append(len(current_ids))
        current_id_dict[str(year)] = list(current_ids)
    
    result = {
        "id_year_dict": current_id_dict, 
        "nunique_df": pd.DataFrame({"year": range(init_year, end_year), "n_unique": num_ids})
    }
    return result

def train_eval(
    mod: Any, 
    scaler: Any, 
    x_train_features: np.ndarray, 
    y_train_labels: np.ndarray, 
    x_test_features: np.ndarray, 
    y_test_labels: np.ndarray, 
    model_accuracy_compare: Dict[str, float]
) -> float:
    """This function takes in a model type, a scaler type, and training and testing data, and returns the accuracy of the model"""
    
    if not isinstance(model_accuracy_compare, dict):
        raise TypeError("model_accuracy_compare must be a dictionary")
    
    x_train_scaled = scaler.fit_transform(x_train_features)
    x_test_scaled = scaler.transform(x_test_features)

    y_train_labels = np.ravel(y_train_labels)
    mod.fit(x_train_scaled, y_train_labels)
    y_predict = mod.predict(x_test_scaled)

    accuracy = accuracy_score(y_test_labels, y_predict)

    if mod.__class__.__name__ == "SVC":
        model_key = (
            f"{mod.__class__.__name__} ({mod.kernel}) - {scaler.__class__.__name__}"
        )
    else:
        model_key = f"{mod.__class__.__name__} - {scaler.__class__.__name__}"

    model_accuracy_compare[model_key] = accuracy

    return accuracy

def reg_train_eval(
    mod: Any, 
    scaler: Any, 
    x_train_features: np.ndarray, 
    y_train_labels: np.ndarray, 
    x_test_features: np.ndarray, 
    y_test_labels: np.ndarray, 
    model_compare_dict: Dict[str, Any],
    prediction_storage: Dict,
    year: str,
) -> float:
    """This function takes in a REGRESSION model type, a scaler type, and training and testing data, 
    and returns model metrics. This function takes into account the fact that predicted y values
    should not be less than 0 (because length of stay (LOS) cannot be negative)."""
    
    if not isinstance(model_compare_dict, dict):
        raise TypeError("model_compare_dict must be a dictionary")
    
    if year is None:
        raise ValueError("You must input a year for model identification")
    
    if scaler is not None:
        x_train_scaled = scaler.fit_transform(x_train_features)
        x_test_scaled = scaler.transform(x_test_features)
    else:
        x_train_scaled = x_train_features
        x_test_scaled = x_test_features
    try:
        y_train_labels = np.ravel(y_train_labels)
    except Exception as e:
        print(f"Error with np.ravel y_train_labels. Error: {e}")
    
    try:
        mod.fit(x_train_scaled, y_train_labels)
        y_predict_train = mod.predict(x_train_scaled)
        y_predict = mod.predict(x_test_scaled)
    except Exception as e:
        print(f"Error with mod.fit or mod.predict. Error: {e}")

    y_predict[y_predict < 0] = 0
    y_predict_train[y_predict_train < 0] = 0

    rmsle_test = root_mean_squared_log_error(y_test_labels, y_predict)
    r2_test = r2_score(y_test_labels, y_predict)
    mae_test = mean_absolute_error(y_test_labels, y_predict)
    mse_test = mean_squared_error(y_test_labels, y_predict)
    rmsle_train = root_mean_squared_log_error(y_train_labels, y_predict_train)
    r2_train = r2_score(y_train_labels, y_predict_train)
    mae_train = mean_absolute_error(y_train_labels, y_predict_train)
    mse_train = mean_squared_error(y_train_labels, y_predict_train)

    model_key = f"{mod.__class__.__name__} - {scaler.__class__.__name__} - {year}"

    model_compare_dict["Test"][model_key] = {"RMSLE": rmsle_test, 
                                             "R2": r2_test, 
                                             "MAE": mae_test, 
                                             "MSE": mse_test}
    model_compare_dict["Train"][model_key] = {"RMSLE": rmsle_train, 
                                              "R2": r2_train, 
                                              "MAE": mae_train, 
                                              "MSE": mse_train}

    prediction_storage[model_key] = {"y_predict": y_predict, 
                                     "y_predict_train": y_predict_train}

    return r2_test

def piped_traineval(
    mod: Any, 
    scaler: Any, 
    x_train_features: np.ndarray, 
    y_train_labels: np.ndarray, 
    x_test_features: np.ndarray, 
    y_test_labels: np.ndarray, 
    model_accuracy_compare: Dict[str, float]
) -> float:
    """This function takes in a model type, a scaler type, and training and testing data, and returns the accuracy of the model using a pipeline"""
    
    if not isinstance(model_accuracy_compare, dict):
        raise TypeError("model_accuracy_compare must be a dictionary")
    
    pipe = make_pipeline(scaler, mod)
    pipe.fit(x_train_features, y_train_labels)
    y_pred = pipe.predict(x_test_features)
    accuracy = accuracy_score(y_test_labels, y_pred)

    if mod.__class__.__name__ == "SVC":
        model_key = (
            f"{mod.__class__.__name__} ({mod.kernel}) - {scaler.__class__.__name__}"
        )
    else:
        model_key = f"{mod.__class__.__name__} - {scaler.__class__.__name__}"

    model_accuracy_compare[model_key] = accuracy

    return accuracy

def df_train_test(df, target_col, test_size=0.2, random_state=42):
    '''This function takes in a dataframe, a target column, and a test size, and returns the training and testing data'''
    from sklearn.model_selection import train_test_split

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

def graph_results(
    y_train_pred: np.ndarray, 
    y_test_pred: np.ndarray, 
    y_train_labels: np.ndarray, 
    y_test_labels: np.ndarray
) -> None:
    """This function graphs the predicted results of a regression model."""
    
    x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
    x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

    ax1.scatter(
        y_test_pred,
        y_test_pred - y_test_labels,
        c="limegreen",
        marker="s",
        edgecolor="white",
        label="Test data",
    )
    ax2.scatter(
        y_train_pred,
        y_train_pred - y_train_labels,
        c="steelblue",
        marker="o",
        edgecolor="white",
        label="Training data",
    )
    ax1.set_ylabel("Residuals")

    for ax in (ax1, ax2):
        ax.set_xlabel("Predicted values")
        ax.legend(loc="upper left")
        ax.hlines(y=0, xmin=x_min - 100, xmax=x_max + 100, color="black", lw=2)

    plt.tight_layout()

    plt.show()
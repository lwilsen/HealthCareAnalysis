from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

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


def piped_traineval(
    mod: Any, 
    scaler: Any, 
    x_train_features: np.ndarray, 
    y_train_labels: np.ndarray, 
    x_test_features: np.ndarray, 
    y_test_labels: np.ndarray, 
    model_accuracy_titanic_compare_piped: Dict[str, float]
) -> float:
    """This function takes in a model type, a scaler type, and training and testing data, and returns the accuracy of the model using a pipeline"""
    
    if not isinstance(model_accuracy_titanic_compare_piped, dict):
        raise TypeError("model_accuracy_titanic_compare_piped must be a dictionary")
    
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

    model_accuracy_titanic_compare_piped[model_key] = accuracy

    return accuracy

def df_train_test(df, target_col, test_size=0.2, random_state=42):
    '''This function takes in a dataframe, a target column, and a test size, and returns the training and testing data'''
    from sklearn.model_selection import train_test_split

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test
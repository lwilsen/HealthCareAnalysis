from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

def describe_dataframe(df):
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
        'sample_values': list(column_data.dropna().unique())[:5],
    }
    column_descriptions.append(description)

  return pd.DataFrame(column_descriptions)

def get_cont_enrolled(init_year, end_year, df):
    '''Takes a dataframe of CMS inpatient data, and gives you the number of unique id's that were present
    in every year from the init year to the end year'''


    select_years = df[df["YR"] < end_year]

    current_ids = select_years[select_years['YR'] == init_year]['BENE_ID'].unique()
    num_ids = [len(current_ids)]
    
    for year in range(init_year + 1, end_year):
        
        current_ids = select_years[(select_years["BENE_ID"].isin(current_ids)) & 
                                   (select_years['YR'] == year)]["BENE_ID"].unique()
        num_ids.append(len(current_ids))
    
    return(pd.DataFrame({"year" : range(init_year,end_year),
                         "n_unique" : num_ids}))

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

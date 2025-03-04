import pandas as pd

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

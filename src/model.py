# Train and testing model

import pandas as pd

def load_ndjson_to_df(file_name):
    """
    Load an ndjson file into a pandas DataFrame.
    
    Args:
        file_name (str): The name of the ndjson file.
        
    Returns:
        pandas.DataFrame: A DataFrame containing the data from the ndjson file.
    """
    try:
        df = pd.read_json(file_name, lines=True, orient='records')
        return df
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

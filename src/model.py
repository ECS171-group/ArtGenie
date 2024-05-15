# Data loading

import pandas as pd

def load_ndjson_to_df(file_names, print_progress=False):
    """
    Load one or more ndjson files into a pandas DataFrame.

    Args:
        file_names (list or str): A list of ndjson file names or a single file name.
        print_progress: if True, prints debug messages that shows the progress

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the ndjson file(s).
    """
    try:
        # Create an empty DataFrame to store the combined data
        combined_df = pd.DataFrame()

        # Check if file_names is a list or a single string
        if isinstance(file_names, list):
            files = file_names
        else:
            files = [file_names]

        # Iterate over the list of file names
        for file_name in files:
            # Read the ndjson file into a DataFrame
            if (print_progress):
                print(f"Reading {file_name}...", end=" ")
            df = pd.read_json(file_name, lines=True, orient='records')

            # Concatenate the DataFrame with the combined DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            if (print_progress):
                print("Completed")

        return combined_df

    except FileNotFoundError as e:
        print(f"File '{e.filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


data_files = ["./data/full_simplified_apple.ndjson",
                "./data/full_simplified_banana.ndjson",
                "./data/full_simplified_blueberry.ndjson",
                "./data/full_simplified_watermelon.ndjson",]
print("Loading data...")
df = load_ndjson_to_df(data_files, print_progress=True)
print(df);

# EDA
print("Preprocessing data...")

def clean_ndjson_data(df):
    """
    Clean up the ndjson data by removing metadata columns and filtering out unrecognized
    data
    
    Metadata columns removed: "countrycode", "timestamp", and "key_id" columns.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the ndjson data.
        
    Returns:
        pandas.DataFrame: A new DataFrame with the specified columns removed.
    """
    # Drop the specified columns
    cleaned_df = df[df['recognized'] == True]
    cleaned_df = cleaned_df.drop(columns=["countrycode", "timestamp", "key_id", "recognized])

    return cleaned_df

cleaned_df = clean_ndjson_data(df)
# One-hot encode "word"
cleaned_df = pd.get_dummies(cleaned_df, columns=["word"], prefix=["fruit"])
print(cleaned_df);
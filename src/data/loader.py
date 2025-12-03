def load_raw_data(file_path):
    import pandas as pd
    """
    Load raw data from the specified file path.

    Parameters:
    file_path (str): The path to the raw data file.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def load_processed_data(file_path):
    import pandas as pd
    """
    Load processed data from the specified file path.

    Parameters:
    file_path (str): The path to the processed data file.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
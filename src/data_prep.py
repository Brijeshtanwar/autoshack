import os
import pandas as pd
import numpy as np

dataframes = []
files_with_na = []
files_with_duplicates = []


def get_dataframes():
    folder_path = 'autoshackdata/all_combined_data/'
    print("Loading the files from:", folder_path)
    print("Total number of files found:", len(os.listdir(folder_path)))
    # print("Name list of files found:", os.listdir(folder_path))

    # Iterate over the files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is a CSV or Excel file
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_name.endswith('.xls') or file_name.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            continue  # Skip files that are not CSV or Excel
            # Handle different column names for "email" and "issue" files
        if 'email' in df.columns and 'issue' in df.columns:
            df.rename(columns={'email': 'text', 'issue': 'label'}, inplace=True)
        elif 'email' in df.columns:
            df.rename(columns={'email': 'text'}, inplace=True)
        elif 'issue' in df.columns:
            df.rename(columns={'issue': 'label'}, inplace=True)

        df = df[['text', 'label']]
        # Append the dataframe to the list
        dataframes.append(df)

        # Check if the dataframe contains NA records
        if df.isna().any().any():
            files_with_na.append(file_name)

        # Check if the dataframe contains duplicate records
        if df.duplicated().any():
            files_with_duplicates.append(file_name)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def filtered_data(dataframes):
    print("Shape before dropping NA and duplicates:", dataframes.shape)
    # Drop duplicate and NA records
    dataframes.drop_duplicates(inplace=True)
    dataframes.dropna(inplace=True)
    print("Shape after dropping NA and duplicates:", dataframes.shape)

    print("Final Combined File Shape used for model:", dataframes.shape)
    print("Files with NA records:", files_with_na)
    print("Files with duplicate records:", files_with_duplicates)

    print("* From the ", dataframes.shape[0],
          "records, we have", len(dataframes.label.value_counts()), "unique categories.")

    category_counts = dataframes['label'].value_counts()
    categories_above_threshold = category_counts[category_counts >= 100]
    print("* Count of Categories with at least 100 records:", len(categories_above_threshold))
    print("* Categories with at least 100 records:")
    print(categories_above_threshold)

    # Filtering the lables which have atleast 100 records
    label_counts = dataframes['label'].value_counts()
    labels_below_threshold = label_counts[label_counts < 100].index
    # Removing Please Specify category
    dataframes = dataframes[dataframes['label'] != 'Please Specify']

    combined_df_filtered = dataframes[~dataframes['label'].isin(labels_below_threshold)]
    return combined_df_filtered


def down_sampling(df):
    """
    this function will down sample the data for the given category  and return the balanced dataframe
    :param category: category for which data needs to be down sampled
    :param df: dataframe
    :return: balanced dataframe
    """
    exp_df = df.copy()
    x = exp_df.label.value_counts()
    condition = x > 20000
    filtered_label = x[condition].index

    # Set the desired number of records for the class "Return Request - Part Returned"
    desired_records = 20000
    for category in filtered_label:
        # Get the indices of the records for the class "Return Request - Part Returned"
        indices = exp_df[exp_df['label'] == category].index

        # Randomly sample the indices to get the desired number of records
        np.random.seed(42)
        sampled_indices = np.random.choice(indices, size=desired_records, replace=False)

        # Create a new dataframe with the sampled records
        exp_df = pd.concat([exp_df[exp_df['label'] != category], exp_df.loc[sampled_indices]])

        # Shuffle the dataframe to ensure randomness
        exp_df = exp_df.sample(frac=1).reset_index(drop=True)

    print(exp_df.label.value_counts())
    print("Shape of the balanced dataframe:", exp_df.shape)
    return exp_df


# data = get_dataframes()
# combined_df = check_header_names(data)
# balanced_df = down_sampling(combined_df)
# # Down sample the data for the class "Claim - Amazon"
# balanced_df = down_sampling(balanced_df)
# print(balanced_df.label.value_counts())
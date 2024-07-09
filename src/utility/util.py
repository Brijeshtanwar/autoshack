import configparser
import math
import os

import boto3
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import fasttext
import warnings
from nltk.corpus import wordnet
from bs4 import BeautifulSoup

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def _pos_tag(word):
    t = wordnet.synsets(word)
    try:
        tag = t[0].pos()
    except:
        tag = 'n'
    return tag

def _encode_labels(labels):
    """
    encode the labels
    :param labels: classes in the output column
    :return: encoded classes and list of all unique classes
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y_encoded = label_encoder.transform(labels)
    class_labels = label_encoder.classes_
    return y_encoded, class_labels.tolist()


def preprocess_data(dataframe):
    df = dataframe.copy()
 
    # Apply text preprocessing
    df['text'] = df['text'].apply(preprocess_text)

     # Encode labels
    y_processed, class_labels = _encode_labels(df.get('label'))

    # Create label dictionary
    label_dict = {"__label__" + str(idx): label for idx, label in enumerate(class_labels)}

    # Create reverse label dictionary
    reverse_label_dict = {label: key for key, label in label_dict.items()}

    # Replace labels in DataFrame
    df['label'] = df['label'].replace(reverse_label_dict)    

    return df, label_dict, reverse_label_dict


def preprocess_text(text):
    if isinstance(text, str):
        # Tokenize
        words = nltk.word_tokenize(text)
        # Remove Numbers
        words = [word for word in words if not re.search(r'\d', word)]
 
        # Lemmatization
        lemmatized_words = [lemmatizer.lemmatize(word.lower(), pos=_pos_tag(word)) for word in words]
        # Stemming not required now
 
        # Remove stopwords
        filtered_words = [word for word in lemmatized_words if word.lower() not in stop_words]
 
        # Lowercasing
        filtered_text = ' '.join(filtered_words).lower()
        # Special character removal
        filtered_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', filtered_text)
 
        # Punctuation removal
        filtered_text = re.sub(r'[^\w\s]', '', filtered_text)
        
        # Remove HTML/CSS tags from text
        soup = BeautifulSoup(filtered_text, 'html.parser')
        filtered_text = soup.get_text(separator=' ', strip=True)
        # Remove HTML entities (e.g., &amp;, &lt;)
        filtered_text = re.sub(r'&\w+;', '', filtered_text)
        # Remove residual tags and styles
        filtered_text = re.sub(r'<.*?>', '', filtered_text)
        filtered_text = re.sub(r'\{.*?\}', '', filtered_text)  # Remove CSS styles within {}
 
        # Remove extra whitespaces
        filtered_text = ' '.join(filtered_text.split())
 
        return filtered_text
    return text


def split_and_save_data(df:pd.DataFrame, model_name,  test_size=0.2, random_state=42, output_folder="data/", s3_folder_name=None):
    # Split data into train, validation, and test sets
    train_data, test_data = train_test_split(df,
                                            test_size=test_size, 
                                            random_state=random_state,
                                            stratify=df['label'])
                                            
    train_data, validation_data = train_test_split(train_data, 
                                                   test_size=test_size, 
                                                   random_state=random_state,
                                                   stratify=train_data['label'])
                                                   


    # Define file names
    train_file_name = f"{output_folder}/data.train"
    validation_file_name = f"{output_folder}/data.val"
    test_file_name = f"{output_folder}/data.test"

    # Save DataFrames to CSV files
    train_data.to_csv(train_file_name, index=False, sep='\t')
    validation_data.to_csv(validation_file_name, index=False, sep='\t')
    test_data.to_csv(test_file_name, index=False, sep='\t')

    upload_file_to_s3(train_file_name, f"model_data/{model_name}", s3_folder_name)
    upload_file_to_s3(validation_file_name, f"model_data/{model_name}", s3_folder_name)
    upload_file_to_s3(test_file_name, f"model_data/{model_name}", s3_folder_name)

    return train_file_name, validation_file_name, test_file_name


def attach_samplecount (target_df:pd.DataFrame, master_record:pd.DataFrame):
    # Combine training sample size with the metrics. 
    # It changes the original dataframe
    look_up_labels = target_df.index.to_list()
    target_df["sample_count"] = master_record[master_record['label'].isin(look_up_labels)]['label'].value_counts().sort_index()
    target_df = target_df.dropna()


# def get_score_matrix_on_labels(prediction_df, threshold=0.75):
#     from sklearn.metrics import classification_report
#
#     predicted_df = prediction_df.copy()
#     # Calculate precision, recall, and F1-score for each class
#     classification_metrics = classification_report(predicted_df['True_Label'], predicted_df['Prediction_Class'], output_dict=True)
#
#     # Convert classification metrics to a dataframe
#     metrics_all_labels = pd.DataFrame(classification_metrics).transpose()[['precision', 'recall', 'f1-score']]
#
#     # metrics_df['predicted_count'] = predicted_df['Prediction_Class'].value_counts().sort_index()
#     # metrics_df['true_count'] = predicted_df['True_Label'].value_counts().sort_index()
#
#     # Select only the records which have an F1 score above 0.75
#     metrics_selected_labels = metrics_all_labels[metrics_all_labels['f1-score'] > threshold]
#     return metrics_all_labels, metrics_selected_labels





################## Training

def _calculate_f1_score(precision, recall):
    f1_score = (2 * precision * recall) / (precision + recall + .001)
    if math.isnan(f1_score):
        return 0
    return f1_score

def _get_best_model(train_file, validation_file):
    lr = 0.5
    dim = 100
    epoch = 200
    min_count = 1
    bucket = 200000
    wordNgrams = 10
    loss = 'ova'
    # Train the final model with the best hyperparameters
    final_model = fasttext.train_supervised(train_file, lr=lr, dim=dim, bucket=bucket,
                                            epoch=epoch, minCount=min_count,
                                            wordNgrams=wordNgrams, loss=loss,
                                            verbose=2)
    return final_model

def fasttext_train(train_file, validation_file, test_file):
    # finding the best model using hyperparmeter tuning
    model = _get_best_model(train_file, validation_file)
    (testSamples, precision, recall) = model.test(test_file)
    f_score = _calculate_f1_score(precision, recall)
    return model, f_score

def upload_file_to_s3(file_path, folder_name, model_version):
    """
    Uploads a file to an S3 bucket with a specific folder structure.
    :param file_path: The local path of the file to upload.
    :param folder_name: The name of the folder to create inside the MODELVERSION folder.
    :param model_version: The version of the model.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    # Get the AWS credentials from the configuration
    access_key = config.get('AWS', 'access_key')
    secret_key = config.get('AWS', 'secret_key')
    region = config.get('AWS', 'region')
    bucket_name = config.get('AWS', 'bucket_name')

    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)

    s3_key = f"{model_version}/{folder_name}/{os.path.basename(file_path)}"  # Construct the S3 key with the folder structure

    s3.upload_file(file_path, bucket_name, s3_key)



def get_path(folder_name, model_name):
    os.makedirs(os.path.join(folder_name, 'results_metrics', model_name), exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'prediction', model_name), exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'exported_model', model_name), exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'model_data', model_name), exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'input_data', model_name), exist_ok=True)

    predictions_file_path = os.path.join(folder_name, 'prediction', model_name, 'predictions.csv')
    all_metrics_file_path = os.path.join(folder_name, 'results_metrics', model_name, 'all_metrics.csv')
    selected_metrics_file_path = os.path.join(folder_name, 'results_metrics', model_name, 'selected_metrics.csv')
    confusion_matrix_path = os.path.join(folder_name, 'results_metrics', model_name, 'confusion_matrix.png')
    exported_model_file_path = os.path.join(folder_name, 'exported_model', model_name, 'trained_model_train.bin')
    label_dict_file_path = os.path.join(folder_name, 'exported_model', model_name, 'label_dict.joblib')
    reverse_label_dict_file_path = os.path.join(folder_name, 'exported_model', model_name, 'reverse_label_dict.joblib')
    model_data_path = os.path.join(folder_name, 'model_data', model_name)
    input_data_path = os.path.join(folder_name, 'input_data', model_name, 'input_data.csv')
    confident_labels_model_path = os.path.join(folder_name, 'exported_model', model_name,
                                               'confident_labels_model.joblib')
    return all_metrics_file_path, confident_labels_model_path, exported_model_file_path, label_dict_file_path, model_data_path, predictions_file_path, reverse_label_dict_file_path, selected_metrics_file_path, input_data_path, confusion_matrix_path

"""
this is the main file for training the model  where input is the text classification data in csv format and output
will be the artifacts of the model which will be used for prediction.
Artifacts will be :-
1. exported model(model.pkl, id2label.pkl, label2id.pkl)
2. prediction ( prediction.csv)
3. matrices (confusion_matrix.csv, classification_report.csv)
4. input data (input_data.csv)
5. model train data (train, test, validation)
"""
import configparser
# train.py  - only use for handling training and retraining
# predict.py     - byom bussiness logic
# preprocess.py - use for preprocessing
# util.py    -
# dataprep.py -  use for data pull  - done
# evaluation.py - use for evaluation and conguision matrix input = model.pkl, id2label.pkl, label2id.pkl, test_data.csv


## in result matrices directory
import os
import sys

import boto3
import pandas as pd
import nltk
import fasttext
import warnings
import joblib

from data_prep import get_dataframes, filtered_data, down_sampling
from evaluation import evalute_model, get_score_matrix_on_labels

from utility.util import preprocess_data, split_and_save_data, fasttext_train, preprocess_text, upload_file_to_s3, \
    get_path

warnings.filterwarnings('ignore')

# nltk.download()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# test the check_header_names function
# data = get_dataframes()


# combined_df = check_header_names(data)
# balanced_df = down_sampling(combined_df)
# # Down sample the data for the class "Claim - Amazon"
# balanced_df = down_sampling(balanced_df)
# print(balanced_df.label.value_counts())

# ******************************************************************************
# folder_path = 'autoshackdata/all_combined_data/'


def get_artifacts():
    """
    this function will return the artifacts of the model
    :return: artifacts of the model
    """
    s3_file_path = sys.argv[1]
    folder_name = sys.argv[2]
    file_path = sys.argv[3]
    # folder_name = "model_deepu"
    # file_path = "AUTOSHACK_COMBINED.csv"
    # s3_file_path = "s3://autoshackdata/all_combined_data/"
    data = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')

    # folder_name = "artifacts"
    combined_df = filtered_data(data)
    # Down sample the data which label count is greater than 3000
    balanced_df = down_sampling(combined_df)
    # get all artifacts path
    all_metrics_file_path_1, confident_labels_model_path_1, exported_model_file_path_1, label_dict_file_path_1, model_data_path_1, predictions_file_path_1, reverse_label_dict_file_path_1, selected_metrics_file_path_1, input_data_path_1, confusion_matrix_path_1 = get_path(
        folder_name, "model-1")
    # save the input data
    balanced_df.to_csv(input_data_path_1, index=False, sep="\t")
    upload_file_to_s3(input_data_path_1, "input_data/model-1", folder_name)
    # Train model-1 on the entire dataset
    model_1_supported_labels = training_model(balanced_df, label_dict_file_path_1, reverse_label_dict_file_path_1,
                                              model_data_path_1, exported_model_file_path_1, predictions_file_path_1,
                                              all_metrics_file_path_1, selected_metrics_file_path_1,
                                              confident_labels_model_path_1, confusion_matrix_path_1,
                                              folder_name, model_name="model-1")

    all_metrics_file_path_2, confident_labels_model_path_2, exported_model_file_path_2, label_dict_file_path_2, model_data_path_2, predictions_file_path_2, reverse_label_dict_file_path_2, selected_metrics_file_path_2, input_data_path_2, confusion_matrix_path_2 = get_path(
        folder_name, "model-2")

    # Train another model only on the labels on which the model-1 performed poorly
    # Drop all the labels that were identified by
    model_2_balanced_df = balanced_df[~balanced_df['label'].isin(model_1_supported_labels)]
    model_2_balanced_df.to_csv(input_data_path_2, index=False, sep="\t")
    upload_file_to_s3(input_data_path_2, "input_data/model-2", folder_name)
    # Train model-2 on the entire dataset of model-2
    model_2_supported_labels = training_model(balanced_df, label_dict_file_path_2, reverse_label_dict_file_path_2,
                                              model_data_path_2, exported_model_file_path_2, predictions_file_path_2,
                                              all_metrics_file_path_2, selected_metrics_file_path_2,
                                              confident_labels_model_path_2, confusion_matrix_path_2, folder_name, model_name = "model-2")
    print("done")


def training_model(balanced_df, label_dict_path, reverse_label_dict_path, split_data_path, model_data_path,
                   prediction_report_path, model_performance_report_on_all_labels_path,
                   model_performance_report_on_selected_labels_path, confident_labels_model_path, confusion_matrix_path,  folder_name, model_name):
    # preprocessing the data
    # from src.utility.util import preprocess_data
    processed_df, label_dict, reverse_label_dict = preprocess_data(balanced_df)
    print("Shape of the processed dataframe:", processed_df.shape)
    print("-----------------------------------------")
    print(processed_df.head())
    dump_label_dict(label_dict, label_dict_path,
                    reverse_label_dict, reverse_label_dict_path, folder_name, model_name)

    # split the data into train, test and validation and save the data
    model_train_file_name, model_validation_file_name, model_test_file_name = split_and_save_data(
        processed_df, model_name, output_folder=split_data_path, s3_folder_name=folder_name)
    model, model_f1_score = train_model(model_data_path, model_test_file_name, model_train_file_name,
                                        model_validation_file_name)
    model.save_model(model_data_path)
    upload_file_to_s3(model_data_path, f"exported_model/{model_name}", folder_name)

    # Predictions
    predicted_df_model = prediction(model, model_test_file_name, label_dict)
    print("predicted_df_model_1", predicted_df_model.shape)
    print(predicted_df_model.head())

    # save the prediction
    predicted_df_model.to_csv(prediction_report_path, index=False, sep='\t')
    upload_file_to_s3(prediction_report_path, f"prediction/{model_name}", folder_name)

    # Evaluation of the model on the test data set and save the metrics in the artifacts folder
    evalute_model(model_data_path, label_dict, predicted_df_model['True_Label'],
                  predicted_df_model["Prediction_Class"], confusion_matrix_path, folder_name, model_name)

    model_metrics_all_labels, model_metrics_selected_labels = get_score_matrix_on_labels(predicted_df_model)
    # Combine training sample size with the metrics
    model_metrics_all_labels["training_sample_count"] = \
        balanced_df[balanced_df['label'].isin(model_metrics_all_labels.index.to_list())][
            'label'].value_counts().sort_index()
    model_metrics_selected_labels["training_sample_count"] = \
        balanced_df[balanced_df['label'].isin(model_metrics_selected_labels.index.to_list())][
            'label'].value_counts().sort_index()
    # Write to reports
    model_metrics_all_labels.to_csv(model_performance_report_on_all_labels_path, sep="\t")
    upload_file_to_s3(model_performance_report_on_all_labels_path, f"results_metrics/{model_name}", folder_name)

    model_metrics_selected_labels.to_csv(model_performance_report_on_selected_labels_path, sep="\t")
    upload_file_to_s3(model_performance_report_on_selected_labels_path, f"results_metrics/{model_name}", folder_name)

    model_1_supported_labels = model_metrics_selected_labels.index.to_list()
    # Store the labels on which we can trust model-1
    joblib.dump(model_metrics_selected_labels.index.to_list(), confident_labels_model_path)
    upload_file_to_s3(confident_labels_model_path, f"exported_model/{model_name}", folder_name)
    return model_1_supported_labels


def train_model(model_1_path, model_1_test_file_name, model_1_train_file_name, model_1_validation_file_name):
    # Model and f1
    model_1, model_1_f1_score = fasttext_train(model_1_train_file_name, model_1_validation_file_name,
                                               model_1_test_file_name)
    return model_1, model_1_f1_score


def dump_label_dict(label_dict_model, label_dict_path, reverse_label_dict_model, reverse_label_dict_path, folder_name, model_name):
    # dump the label dict and reverse label dict
    joblib.dump(label_dict_model, label_dict_path)
    joblib.dump(reverse_label_dict_model, reverse_label_dict_path)

    # upload the label dict and reverse label dict to s3
    upload_file_to_s3(label_dict_path, f"exported_model/{model_name}", folder_name)
    upload_file_to_s3(reverse_label_dict_path, f"exported_model/{model_name}", folder_name)


def prediction(model, data_path, label_dict):
    data = pd.read_csv(data_path, sep='\t')
    input_text = data['text'].copy()  # Create a copy of the 'email' column
    data['text'] = data['text'].apply(preprocess_text)
    # input_text = data['email'].copy()  # Create a copy of the 'email' column
    # data['email'] = data['email'].apply(preprocess_text)
    predicted_class = []
    prediction_confidence = []
    true_label = []
    total_datapoints = data.shape[0]
    for i in range(total_datapoints):
        s = str(data.iloc[i][0])
        prob = model.predict(s)
        # print("LABEL",label_dict[prob[0][0]])
        predicted_class.append(label_dict[prob[0][0]])

        true_label.append(label_dict[data.iloc[i][1]])  # Convert true label to text
        prediction_confidence.append(round(prob[1][0], 2))

    df = pd.DataFrame({
        'Input_Text': input_text,  # Use the original 'email' column
        'Prediction_Class': predicted_class,
        'Prediction_Confidence': prediction_confidence,
        'True_Label': true_label
    })

    return df


get_artifacts()

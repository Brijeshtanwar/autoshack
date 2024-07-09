import fasttext
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from utility.util import upload_file_to_s3


def evalute_model(model_path, label_dict, true_labels, predicted_labels, confusion_matrix_path,  folder_name, model_name):
    # Calculate and display evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    # print(conf_matrix)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print('Confusion Matrix:')
    # print(conf_matrix)
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))

    model = fasttext.load_model(model_path)
    labels = [label_dict[label] for label in model.labels]
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(confusion_matrix_path)
    try:
        upload_file_to_s3(confusion_matrix_path,  f"results_metrics/{model_name}", folder_name)
        print('Confusion matrix uploaded to S3 successfully')
    except Exception as e:
        print(f'Error uploading confusion matrix to S3: {str(e)}')


def get_score_matrix_on_labels(prediction_df, threshold=0.75):

    predicted_df = prediction_df.copy()
    # Calculate precision, recall, and F1-score for each class
    classification_metrics = classification_report(predicted_df['True_Label'], predicted_df['Prediction_Class'], output_dict=True)

    # Convert classification metrics to a dataframe
    metrics_all_labels = pd.DataFrame(classification_metrics).transpose()[['precision', 'recall', 'f1-score']]

    # metrics_df['predicted_count'] = predicted_df['Prediction_Class'].value_counts().sort_index()
    # metrics_df['true_count'] = predicted_df['True_Label'].value_counts().sort_index()

    # Select only the records which have an F1 score above 0.75
    metrics_selected_labels = metrics_all_labels[metrics_all_labels['f1-score'] > threshold]
    return metrics_all_labels, metrics_selected_labels
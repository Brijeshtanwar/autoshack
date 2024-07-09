# Sample business logic script with Flask
import asyncio
import os
import time
import datetime
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, current_app
from functools import wraps
import joblib
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import threading
import fasttext
import random
import configparser
import boto3
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from bs4 import BeautifulSoup
import re
from subprocess import Popen, PIPE

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ------------------------- App Name ---------------------------------------------------
app = Flask(__name__)


# ------------------------- Core Function Definitions ----------------------------------
def preprocess_text(texts):
    preprocessed_texts = []
    for text in texts:
        if isinstance(text, str):
            # Tokenize
            words = nltk.word_tokenize(text)

            # Remove Numbers
            words = [word for word in words if not re.search(r'\d', word)]

            # Lemmatization
            lemmatized_words = [WordNetLemmatizer().lemmatize(word) for word in words]

            # Remove stopwords
            filtered_words = [word for word in lemmatized_words if word.lower() not in set(stopwords.words('english'))]

            # Lowercasing
            filtered_text = ' '.join(filtered_words).lower()

            # Special character removal
            filtered_text = re.sub(r'[^a-zA-Z0-9\s]', '', filtered_text)

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

            preprocessed_texts.append(filtered_text)
    return preprocessed_texts


def authenticate(func):
    """
    Decorator function to authenticate API requests.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check for API key in request headers
        api_key = request.headers.get('X-API-Key')
        if api_key != VALID_API_KEY:
            return jsonify({'error': 'Invalid API key'}), 401
        return func(*args, **kwargs)

    return wrapper


# Route for the OpenAPI specs
@app.route('/<modelname>/openapi')
@authenticate
def openapi(modelname):
    """
    This endpoint returns the OpenAPI Specification (OAS) API specifications of the model.
    """
    with open('open-api-spec.json', 'r') as handle:
        parsed = handle.read()
    return parsed


# Route for healthcheck
@app.route('/<modelname>/health')
def health_check(modelname):
    return 'OK', 200


# ------------------------- AWS Credentials Setup --------------------------------------

@app.route("/<modelname>/aws_credentials", methods=["POST"])
@authenticate
def aws_credentials(modelname):
    # Get the access key, secret key, and region from the request
    aws_credentials = request.get_json()

    # Extract the access key, secret key, and region from the JSON data
    access_key = aws_credentials.get('access_key')
    secret_key = aws_credentials.get('secret_key')
    region = aws_credentials.get('region')
    bucket_name = aws_credentials.get('bucket_name')

    # Store the AWS credentials in a configuration file
    config = f"""
    [AWS]
    access_key = {access_key}
    secret_key = {secret_key}
    region = {region}
    bucket_name = {bucket_name}
    """

    # Write the configuration to a file
    with open('config.ini', 'w') as f:
        f.write(config)

    # Read the creds from file
    config = configparser.ConfigParser()
    config.read('config.ini')
    # Get the AWS credentials from the configuration
    access_key = config.get('AWS', 'access_key')
    secret_key = config.get('AWS', 'secret_key')
    region = config.get('AWS', 'region')
    bucket_name = config.get('AWS', 'bucket_name')

    response = {
        "message": "AWS credentials stored successfully.",
        "access_key": access_key,
        "secret_key": secret_key,
        "region": region,
        "bucket_name": bucket_name
    }
    return jsonify(response)


def get_aws_credentials():
    # Read the credentials from the configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')
    print("CONFIG OBJECT", config)

    # Get the AWS credentials from the configuration
    access_key = config.get('AWS', 'access_key')
    secret_key = config.get('AWS', 'secret_key')
    region = config.get('AWS', 'region')
    bucket_name = config.get('AWS', 'bucket_name')

    # Create a dictionary with the AWS credentials
    aws_credentials = {
        "access_key": access_key,
        "secret_key": secret_key,
        "region": region,
        "bucket_name": bucket_name
    }

    return aws_credentials


# ------------------------- GLobal Variable definition ---------------------------------

retrain_lock = threading.Lock()
retraining_done = False
label_dict_model_1 = None
label_dict_model_2 = None
model_1_supported_labels = None
model_2_supported_labels = None
# extended_support_labels = joblib.load('model_metadata/confident_labels_extended.joblib')

# Preload Model
fasttext_model_1 = None
fasttext_model_2 = None
current_folder = None
# Get the API key from the environment variable
VALID_API_KEY = os.environ.get('API_KEY')


def retrain_async(file):
    global retraining_done, retrain_lock
    # Check if the config.ini file exists
    if not os.path.exists('config.ini'):
        return jsonify({"error": "AWS credentials not found. Please store the credentials first."}), 400

    # Acquire the lock
    retrain_lock.acquire()
    try:
        print("INSIDE RETRAIN")

        if file.filename.endswith('.csv'):
            # Read the AWS credentials from the configuration file
            config = configparser.ConfigParser()
            config.read('config.ini')
            access_key = config.get('AWS', 'access_key')
            secret_key = config.get('AWS', 'secret_key')
            bucket_name = config.get('AWS', 'bucket_name')
            region = config.get('AWS', 'region')

            # Create a unique folder name based on the current timestamp
            timestamp = str(int(time.time()))
            folder_name = f"model_{timestamp}"
            os.makedirs(folder_name)
            os.makedirs(os.path.join(folder_name, 'input_data'))
            file_path = os.path.join(folder_name, 'input_data', file.filename)
            file.save(file_path)

            # Convert the timestamp to a human-readable format
            timestamp_str = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
            print("TIME", timestamp_str)
            print("FOLDER", folder_name, f"s3://{bucket_name}/{file_path}")
            print("bucket name ===", bucket_name)

            # Upload the file to S3
            s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)
            s3.upload_file(file_path, bucket_name, file_path)

            retrain_thread = threading.Thread(target=start_subprocess, args=(bucket_name, file_path, folder_name,))
            retrain_thread.start()


        else:
            return "Invalid file format. Only CSV files are supported.", 400
    finally:
        # Release the lock
        retrain_lock.release()


def start_subprocess(bucket_name, file_path, folder_name):
    print("MODEL LOCATION", file_path)
    global retraining_done
    # Run the retrain pipeline from train.py using subprocess asynchronously
    process = Popen(["python", "src/train.py", f"s3://{bucket_name}/{file_path}", folder_name, file_path],
                    stdout=PIPE, stderr=PIPE)
    # loop = asyncio.get_event_loop()
    stdout, stderr = process.communicate()
    # Check if the retrain pipeline was successful
    if process.returncode == 0:
        retraining_done = True
        print("Retrain pipeline completed successfully", stdout.decode('utf-8'))
        return "Retrain pipeline completed successfully"
    else:
        print("Retrain pipeline failed with error:", stderr.decode('utf-8'))
        return f"Retrain pipeline failed with error: {stderr.decode('utf-8')}"


# Example route in a Flask application
@app.route("/<modelname>/retrain", methods=["POST"])
@authenticate
def retrain(modelname):
    # Check if a file was uploaded
    print("REQUEST", request)
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    print("FILE----", file)
    retrain_async(file)
    print("retraining are running")
    return "Retrain process started in the background"


def get_selected_folders(s3_client, bucket_name):
    # Create an S3 client
    # s3 = boto3.client('s3')

    # List all objects in the bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name, Delimiter='/')

    # Extract the folder names from the response
    folders = [common_prefix['Prefix'] for common_prefix in response.get('CommonPrefixes', [])]

    # Filter the folders to select only the ones with the specified structure
    selected_folders = [folder for folder in folders if re.match(r'^model_\d+/$', folder)]

    # Sort the selected folders based on the timestamp in the name
    sorted_folders = sorted(selected_folders, key=lambda x: int(re.search(r'\d+', x).group()))

    # Get the latest folder name
    latest_folder = sorted_folders[-1] if sorted_folders else None

    return selected_folders, latest_folder


def check_folder_artifacts(s3_client, bucket_name, folder_path):
    # Check if the required artifacts are present in the folder
    required_artifacts = ['exported_model']  # Add any other required artifacts here
    for artifact in required_artifacts:
        artifact_path = os.path.join(folder_path, artifact)
        if not os.path.exists(artifact_path):
            return False

        # Check if the files in the local folder match the files in the S3 bucket
        # s3 = boto3.client('s3')
        s3 = s3_client

        bucket_name = bucket_name
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=artifact_path)
        s3_files = set(obj['Key'] for obj in response['Contents'])
        local_files = set(os.path.join(artifact_path, file) for file in os.listdir(artifact_path))
        if s3_files != local_files:
            return False
    return True


def download_folder_from_s3(s3_client, bucket_name, folder_name):
    # Create an S3 client
    # s3 = boto3.client('s3')
    s3 = s3_client

    # List all objects in the folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)

    # Debug statement to print the response object
    # print("S3 Response:", response)

    # Check if the folder is already present locally and has all the artifacts
    local_folder_path = folder_name.split('/')[-2]  # Get the local folder path
    if check_folder_artifacts(s3_client, bucket_name, local_folder_path):
        print("Latest folder is already present locally with all the artifacts.")
        return

    # print("Response Contents:", response.get('Contents'))
    if 'Contents' in response:
        # Download each file in the folder
        for obj in response['Contents']:
            key = obj['Key']
            filename = key.split('/')[-1]  # Extract the filename from the key
            local_path = os.path.join(local_folder_path,
                                      key[len(folder_name):])  # Create the local path with the same folder structure
            os.makedirs(os.path.dirname(local_path), exist_ok=True)  # Create the necessary directories
            s3.download_file(bucket_name, key, local_path)  # Download the file and save it in the local path
    else:
        print("Response does not contain the expected key 'Contents'.")


# ---------------------------------------------------------------------
# Function to check for model updates and reload the model if necessary
def check_model_updates():
    # Specify the bucket name
    aws_credentials = get_aws_credentials()
    bucket_name = aws_credentials["bucket_name"]
    access_key = aws_credentials["access_key"]
    secret_key = aws_credentials["secret_key"]
    region = aws_credentials["region"]

    print("BUCKET NAME", bucket_name)
    s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)

    print("Check Model Updates running")
    global fasttext_model_1, fasttext_model_2, label_dict_model_1, label_dict_model_2, retraining_done, current_folder, retrain_lock

    # Get the selected folders and the latest folder in the bucket
    selected_folder_list, latest_folder = get_selected_folders(s3_client, bucket_name)

    # Check if retraining has been done and the latest folder is different from the current one
    print("Sleeping for 15 seconds...", retraining_done)
    if retraining_done and latest_folder != current_folder:
        print("RETRAINING FLAG TRUE")
        # Acquire the lock
        retrain_lock.acquire()
        try:
            # Download the latest folder from S3 if not already present locally with all the artifacts
            if latest_folder:
                download_folder_from_s3(s3_client, bucket_name, latest_folder)

            # Check if the model file exists
            model_file_path = os.path.join(latest_folder, "exported_model/model-1", "trained_model_train.bin")
            fasttext_model_1_file_path = os.path.join(latest_folder, "exported_model/model-1",
                                                      "trained_model_train.bin")
            fasttext_model_2_file_path = os.path.join(latest_folder, "exported_model/model-2",
                                                      "trained_model_train.bin")
            if os.path.exists(fasttext_model_1_file_path) and os.path.exists(fasttext_model_2_file_path):
                # Load the new model for predictions
                fasttext_model_1 = fasttext.load_model(fasttext_model_1_file_path)
                fasttext_model_2 = fasttext.load_model(fasttext_model_2_file_path)
                label_dict_model_1 = joblib.load(
                    os.path.join(latest_folder, "exported_model/model-1", "label_dict.joblib"))
                label_dict_model_2 = joblib.load(
                    os.path.join(latest_folder, "exported_model/model-2", "label_dict.joblib"))

                # Update the current folder
                current_folder = latest_folder

                # Print the folder name
                print("New version found. Folder name:", latest_folder)

                # Reset the retraining_done flag
                retraining_done = False
            else:
                print("Model file does not exist:", model_file_path)
        finally:
            # Release the lock
            retrain_lock.release()
    else:
        print("No new version found. Using current version:", current_folder)


# Function to run the check_model_updates() function every 15 seconds
def run_check_model_updates():
    while True:
        check_model_updates()
        time.sleep(15)  # Adjust the interval as needed


# Specify the bucket name
aws_credentials = get_aws_credentials()
bucket_name = aws_credentials["bucket_name"]
access_key = aws_credentials["access_key"]
secret_key = aws_credentials["secret_key"]
region = aws_credentials["region"]

print("BUCKET NAME", bucket_name)
s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)

# bucket_name = 'byoe-sandbox'

# ---------------------------------------------------------------------
# Get the selected folders and the latest folder in the bucket
selected_folder_list, latest_folder = get_selected_folders(s3_client, bucket_name)

if retraining_done:
    print("Getting the model from retrainined folder")

    # Get the selected folders and the latest folder in the bucket
    selected_folder_list, latest_folder = get_selected_folders(s3_client, bucket_name)

    # Download the latest folder from S3 if not already present locally with all the artifacts
    if latest_folder:
        download_folder_from_s3(s3_client, bucket_name, latest_folder)

    print("Latest Folder:", latest_folder)
    # Load the new model for predictions
    model1_path = os.path.join(latest_folder, "exported_model/model-1", "trained_model_train.bin")
    model2_path = os.path.join(latest_folder, "exported_model/model-2", "trained_model_train.bin")

    print("Model 1 Path:", model1_path)
    print("Model 2 Path:", model2_path)

    fasttext_model_1 = fasttext.load_model(model1_path)
    fasttext_model_2 = fasttext.load_model(model2_path)
    # fasttext_model = fasttext.load_model(os.path.join(latest_folder, "exported_model", "trained_model_train.bin"))
    label_dict_model_1 = joblib.load(os.path.join(latest_folder, "exported_model/model-1", "label_dict.joblib"))
    label_dict_model_2 = joblib.load(os.path.join(latest_folder, "exported_model/model-2", "label_dict.joblib"))

    model_1_supported_labels = joblib.load(
        os.path.join(latest_folder, "exported_model/model-2", "confident_labels_model.joblib"))
    model_2_supported_labels = joblib.load(
        os.path.join(latest_folder, "exported_model/model-2", "confident_labels_model.joblib"))
else:
    print("Getting the model from DEFAULT folder")
    # Import label dict
    # Import label dict
    label_dict_model_1 = joblib.load('model_metadata/label_dict_model_1.joblib')
    label_dict_model_2 = joblib.load('model_metadata/label_dict_model_2.joblib')
    model_1_supported_labels = joblib.load('model_metadata/confident_labels_model_1.joblib')
    model_2_supported_labels = joblib.load('model_metadata/confident_labels_model_2.joblib')
    # extended_support_labels = joblib.load('model_metadata/confident_labels_extended.joblib')

    # Preload Model
    fasttext_model_1 = fasttext.load_model('trained_model/model-1.bin')
    fasttext_model_2 = fasttext.load_model('trained_model/model-2.bin')


    # ------------------------- Retraining Done ---------------------------
    # ------------------------- Predict Pipeline Starts -------------------
    # Route for the prediction
    # Route for the prediction
    @app.route('/<modelname>/predict', methods=['POST'])
    @authenticate
    def predict(modelname):
        """
        This endpoint accepts POST requests with input data to make predictions using the model.
        The input data should be a JSON object.
        The endpoint returns a JSON response with the predicted label and the confidence score.
        """
        input_data = request.get_json()
        input_data = input_data.get('text')
        # Remove multiple double quotes

        if len(input_data.split()) < 9:
            not_an_email_confidence_score = round(random.uniform(83, 87), 2)
            result = {
                "predicted_class": "Not an Email",
                "confidence_score": not_an_email_confidence_score,
            }
            return jsonify(result)

        result = actully_predict(input_data)

        return jsonify(result)


    def actully_predict(input_data):
        # Perform any required pre-processing on the input data
        preprocessed_data = preprocess_text([input_data])[0]
        # FastText Model
        model_1_predicted_class, model_1_confidence_score = calculate_prediction(fasttext_model_1, label_dict_model_1,
                                                                                 preprocessed_data)
        final_prediction_class = ""
        if model_1_predicted_class in model_1_supported_labels:
            # If the prediction from model-1 is amongst the supported labels of model-1, then honour that
            if model_1_confidence_score > 0.95:
                normalized_conf_score = model_1_confidence_score * 0.95
            else:
                normalized_conf_score = model_1_confidence_score
            final_prediction_class = model_1_predicted_class
        else:
            # Check if 2nd model can do better
            model_2_predicted_class, model_2_confidence_score = calculate_prediction(fasttext_model_2,
                                                                                     label_dict_model_2,
                                                                                     preprocessed_data)
            if model_2_predicted_class in model_2_supported_labels:
                # If the prediction from model-2 is amongst the supported labels of model-2, then honour that
                if model_2_confidence_score > 0.95:
                    normalized_conf_score = model_2_confidence_score * 0.95
                else:
                    normalized_conf_score = model_2_confidence_score
                final_prediction_class = model_2_predicted_class
            # elif model_2_predicted_class in extended_support_labels:
            #     # Check if the prediction is in extended list of labels
            #     final_prediction_class = model_2_predicted_class
            #     normalized_conf_score = model_2_confidence_score * 0.80
            else:
                normalized_conf_score = model_2_confidence_score * 0.6
                print("Krista - Low Confidence Flag:", model_2_predicted_class)
                final_prediction_class = "Krista - Low Confidence Flag"

        result = {
            "predicted_class": final_prediction_class,
            "confidence_score": round(normalized_conf_score, 2)
        }
        return result


    def calculate_prediction(model, label_dict, preprocessed_data):
        # Fast text Predictions
        model_predictions = model.predict(preprocessed_data)
        predicted_class = label_dict[model_predictions[0][0]]
        prediction_confidence_score = model_predictions[1][0]
        # print("FastText model Predicted Class:", predicted_class)
        # print("FastText model Confidence Score:", prediction_confidence_score)
        return predicted_class, prediction_confidence_score

# ------------------------- Thread to check for model Updates ------------
# Start the thread for model update checking
update_thread = threading.Thread(target=run_check_model_updates)
update_thread.daemon = True
update_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9595)

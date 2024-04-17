import boto3
import sagemaker
import pandas as pd
from sklearn.model_selection import train_test_split

# %%

sm_boto3 = boto3.client('sagemaker')
sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = 'diabetessagemakerbucket'
print(region)
print('using this bucket: ', bucket)
# %% md
# Load the data
# %%
df = pd.read_csv('diabetes.csv')
# %%
features = list(df.columns)
print(features)
labels = features.pop(-1)
print(labels)
# %%
x = df[features]
y = df[labels]
# %%
print(x.shape)
print(y.shape)
# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# %%
trainx = pd.DataFrame(x_train)
trainx[labels] = y_train

testx = pd.DataFrame(x_test)
testx[labels] = y_test
# %%
print(trainx.shape)
print(testx.shape)
# %%
trainx.to_csv('train.csv', index=False)
testx.to_csv('test.csv', index=False)
# %%
sk_prefix = 'sagemaker/diabetes/sklearn'
trainpath = sess.upload_data(path='train.csv', bucket=bucket, key_prefix=sk_prefix)

testpath = sess.upload_data(path='test.csv', bucket=bucket, key_prefix=sk_prefix)

print(trainpath)
print(testpath)
# %%
%%writefile
script.py

import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sklearn
import argparse

print('Importing Libraries')


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


print('Model Function has been defined')
if __name__ == '__main__':
    print('[INFO] Extracting Arguments')
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=0)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='train.csv')
    parser.add_argument('--test-file', type=str, default='test.csv')

    print('Arguments have been defined')

    args, _ = parser.parse_known_args()
    print("sklearn version: ", sklearn.__version__)
    print('joblib version: ', joblib.__version__)

    print('[INFO] Reading Data')
    print()

    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features.pop(-1)

    print("Building training and testing datasets")
    print()

    x_train = train_df[features]
    x_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

    print('Data has been splitted')

    print('column order: ')
    print(features)
    print()

    print('label column is : ', label)
    print(label)
    print()

    print('data shape: ')

    print("------train data shape: ")
    print('x_train shape: ', x_train.shape)
    print('y_train shape: ', y_train.shape)

    print("------test data shape: ")
    print('x_test shape: ', x_test.shape)
    print('y_test shape: ', y_test.shape)
    print()

    print('Training Model.......')
    print()
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, verbose=1)
    model.fit(x_train, y_train)
    print('Model has been trained')

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print('Model has been saved')
    print()

    print('Evaluating Model')
    y_pred_test = model.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_rep = classification_report(y_test, y_pred_test)

    print()
    print('Model has been evaluated')
    print()
    print('Test Accuracy: ', test_acc)
    print('Classification Report: ')
    print(test_rep)
    print()
# %%
from sagemaker.sklearn.estimator import SKLearn

FRAMEWORK_VERSION = "0.23-1"

sklearn_estimator = SKLearn(
    entry_point='script.py',
    role='',  # fill in the role(copy the arn from the role you created in the IAM console)
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version=FRAMEWORK_VERSION,
    base_job_name='diabetes-sklearn',
    hyperparameters={
        'n_estimators': 100,
        'random_state': 0
    },
    use_spot_instances=True,
    max_wait=7200,
    max_run=3600
)
# %%
# launch training job with asynchronous call
sklearn_estimator.fit({'train': trainpath, 'test': testpath}, wait=True)
# %%
# deploy the model
sklearn_estimator.latest_training_job.wait(logs='None')
artifact = sm_boto3.describe_training_job(
    TrainingJobName=sklearn_estimator.latest_training_job.name
)['ModelArtifacts']['S3ModelArtifacts']

print('Model artifact saved at: ', artifact)
# %%
from sagemaker.sklearn.model import SKLearnModel
from time import gmtime, strftime

model_name = 'diabetes-sklearn-model' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
sklearn_model = SKLearnModel(
    name=model_name,
    model_data=artifact,
    role='',  # fill in the role(copy the arn from the role you created in the IAM console)
    entry_point='script.py',
    framework_version=FRAMEWORK_VERSION
)
# %%
print(sklearn_model)
model_name
# %%
endpoint_name = 'diabetes-sklearn-endpoint' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
predictor = sklearn_model.deploy(
    instance_type='ml.t2.medium',
    initial_instance_count=1,
    endpoint_name=endpoint_name
)
# %%
print(endpoint_name)
predictor
# %%
a = testx[features][152:154].values.tolist()
print(a)

# %%
predictor.predict(a)
# %%
sm_boto3.delete_endpoint(EndpointName=endpoint_name)
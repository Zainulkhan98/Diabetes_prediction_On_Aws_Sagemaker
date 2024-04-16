
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pathlib
from io import StringIO
import boto3
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
    
    #Hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=0)
    
    
    
    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='train.csv')
    parser.add_argument('--test-file',type=str, default='test.csv')
    
    print('Arguments have been defined')
    
    args,_ = parser.parse_known_args()
    print("sklearn version: ",  sklearn.__version__)
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
    x_test  = test_df[features]
    y_train = train_df[label]
    y_test  = test_df[label]
    
    print('Data has been splitted')
    
    print('column order: ')
    print(features)
    print()
    
    
    print('label column is : ',label)
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
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state,verbose=1)
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

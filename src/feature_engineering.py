import os
import yaml
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_enginnering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter(
    fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)->dict:
    try: 
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as E:
        logger.error("Exception occurred: %s",E)

def load_data(data_url:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data successfuly loaded")
        return df
    except Exception as E:
        logger.error("Error occured: %s",E)

def apply_tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int)->pd.DataFrame:
    try:
        train_data['text'] = train_data['text'].fillna('')
        test_data['text'] = test_data['text'].fillna('')
        logger.debug("Missing values in 'text' column replaced with empty strings.")

        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train = train_data['text'].values
        X_test = test_data['text'].values
        y_train = train_data['target'].values
        y_test = test_data['target'].values
        logger.debug("TFIDF Vectorizer initialized with max_features=%d", max_features)
        # print(X_test[:5])
        # print(X_train[:5])
        # print(y_test[:5])
        # print(y_train[:5])
        x_train_bow = vectorizer.fit_transform(X_train)
        x_test_bow = vectorizer.transform(X_test)
        logger.debug("TFIDF transformation applied on training and testing data")

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['label'] = y_test
        logger.debug("Transformed data converted to DataFrame")
        return train_df, test_df
    except Exception as E:
        logger.error("Error occured: %s",E)

def main():
    try:
        params = load_params(params_path = "params.yaml")
        max_features = params['feature_engineering']['max_features']

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data,test_data, max_features)

        file_path = "./data/preprocessed"
        os.makedirs(file_path,exist_ok=True)
        train_df.to_csv(os.path.join(file_path,'train_tdidf.csv'),index=False)
        test_df.to_csv(os.path.join(file_path,'test_tdidf.csv'),index=False)
        logger.debug("Transformed data saved to %s", file_path)
        logger.info("Feature engineering Pipeline completed successfully")
    except Exception as E:
        logger.error("Error occured: %s",E)

if __name__=="__main__":
    main()
import os 
import yaml
import logging 
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')
nltk.download('punkt')

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(file_path, mode='a')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter(
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text) # the row has been converted to list from string
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]
    return " ".join(text)

def preprocess_df(df, text_column = "text",target_column = "target"):
    try:
        # Remove Duplicates
        df = df.drop_duplicates()
        logger.debug('Duplicates Removed')

        # Label encode the target
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column in encoded.')

        # Label encode the text
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('New preprocessed transformed text.')
        return df

    except Exception as E:
        logger.error("error occurred: %s",E)

def main(text_column = "text", target_column = "target"):
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data Loaded Successfully')

        preprocessed_train = preprocess_df(train_data,text_column,target_column)
        preprocessed_test = preprocess_df(test_data,text_column,target_column)

        data_path = os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)
        print("Preprocessed Test:")
        print(preprocessed_test)
        print("Preprocessed Train:")
        print(preprocessed_train)
        preprocessed_train.to_csv(os.path.join(data_path,'train_processed.csv'),index=False)
        preprocessed_test.to_csv(os.path.join(data_path,'test_processed.csv'),index=False)
        logger.debug('Processed data saved to %s', data_path)
        logger.info('Data Preprocessing Pipeline Completed Successfully')
    except Exception as E:
        logger.error("error occurred: %s",E)

if __name__=="__main__":
    main()





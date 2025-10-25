import pandas as pd
import os
import yaml
import logging
from sklearn.model_selection import train_test_split

log_dir = "logs"
os.makedirs(log_dir,exist_ok = True)
log_file_path = os.path.join(log_dir,'data_ingestion.log')

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
file_handler= logging.FileHandler(log_file_path,mode='a')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
            logger.info('Params loaded successfully from %s', params_path)
            return params
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except yaml.YAMLError as ye:
        logger.error('YAML Error :%s', ye)
        raise
    except Exception as E:
        logger.error('Unexpected Error :%s', E)
        raise

def load_data(data_path:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        if df.empty:
            logger.warning('Dataframe is empty')
            raise pd.errors.EmptyDataError('Dataframe is empty')
        logger.debug('Data successfully loaded from %s',data_path)
        return df
    except FileNotFoundError:
        logger.error("File Not found at %s",data_path)
        raise 
    except pd.errors.ParseError as e:
        logger.error("Failed to parse csv file: %s",e)
        raise
    except IOError as E:
        logger.error("Failed to fetch data from URL or file: %s",E)
        raise
    except Exception as E:
        logger.error("Error occured: %s",E)

def preprocess_dataset(df:pd.DataFrame)->pd.DataFrame:
    try:
        df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)
        df.rename(columns = {'v1':'target','v2':'text'},inplace=True)
        logger.debug('Data Preprocessing Complete')
        return df
    except KeyError as ke:
        logger.error("Missing Column in dataframe: %s",ke)
        raise
    except Exception as E:
        logger.error("Unexpected error occured: %s",E)

def save_data(train:pd.DataFrame, test:pd.DataFrame, data_path:str)->None:
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug('Train and test data saved at path: %s',raw_data_path)
    except Exception as E:
        logger.error("Error occured: %s",E)
        raise
def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']
        data_path = "/Users/shriyansh/Documents/Data Science/mlops/aws/experiments/spam.csv"
        df = load_data(data_path=data_path)
        print("Initial Data:")
        print(df.head())
        df = preprocess_dataset(df)
        print("Data after Preprocessing:")
        print(df.head())
        train,test = train_test_split(df,test_size=test_size,random_state=random_state)
        save_data(train,test,data_path = "./data")
        logger.info("Data Ingestion Pipeline completed successfully.")
    except Exception as e:
        logger.error("Failed to complete data ingestion pipeline: %s", e)
        return

if __name__=='__main__':
    main()
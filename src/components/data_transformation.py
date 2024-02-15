import os
from src import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from src.utils.common import save_pickle
from pathlib import Path
from src.config.configuration import (DataTransformationConfig)

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def train_test_spliting(self):

        cols2read = ['alpha','delta','u','g','r','i','z','redshift','class']
        data = pd.read_csv(self.config.data_path, usecols=cols2read)

        logger.info('Data loaded')

        mapping = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
        data[self.config.TARGET_COLUMN] = data[self.config.TARGET_COLUMN].replace(mapping)
        save_pickle(path=Path(self.config.target_encoding_path), data=mapping)

        logger.info('Target encoding done')

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data, random_state=42)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# from src.components.data_transformation import (
#     DataTransformation,
#     # DataTransformationConfig,
# )
# from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.exception import CustomError
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = str(Path("artifacts") / "train.csv")
    test_data_path: str = str(Path("artifacts") / "test.csv")
    raw_data_path: str = str(Path("artifacts") / "data.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str]:
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebooks/data/stud.csv")
            logging.info("Read the dataset as dataframe")

            Path(self.ingestion_config.train_data_path).parent.mkdir(
                parents=True, exist_ok=True,
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True,
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True,
            )

            logging.info("Ingestion of the data is completed")

        except Exception as e:
            raise CustomError(e, sys)
        else:
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    # data_transformation = DataTransformation()
    # train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
    #     train_data,
    #     test_data,
    # )

    # modeltrainer = ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr, test_arr))

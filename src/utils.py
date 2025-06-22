import pickle
import sys
from pathlib import Path

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomError


def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = Path(file_path).parent

        dir_path.mkdir(parents=True, exist_ok=True)

        with Path(file_path).open("wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomError(e, sys)


def evaluate_models(X_train: object, y_train: object, X_test: object, y_test: object, models: dict, param: dict) -> dict:
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

    except Exception as e:
        raise CustomError(e, sys)
    else:
        return report

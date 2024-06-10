import seaborn as sns
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, classification_report
import joblib
import numpy as np
from las_file_manager import PointCloudManager
import matplotlib.pyplot as plt
from settings import COLUMNS
from typing import Optional, Tuple
import time


def show_classification_report(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Show the balanced accuracy score and classification report.

    Parameters:
    y_test (np.ndarray): The true labels.
    y_pred (np.ndarray): The predicted labels.
    """
    acuracy = balanced_accuracy_score(y_test, y_pred)
    print(f'Balanced accuracy: {acuracy:.2f}')
    print(classification_report(y_test, y_pred))


def save_features(features: pd.DataFrame, y_train: np.ndarray, file_name: str) -> None:
    """
    Saves features and corresponding labels to a parquet file.

    Parameters:
    features (pd.DataFrame): The features dataframe.
    y_train (np.ndarray): The labels array.
    file_name (str): The name of the file to save.
    """
    y_train_series = pd.Series(y_train, name='y_train')
    combined_df = pd.concat([features, y_train_series], axis=1)
    combined_df.to_parquet(file_name)


def load_features(file_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Loads features and corresponding labels from a parquet file.

    Parameters:
    file_name (str): The name of the file to load.

    Returns:
    Tuple[pd.DataFrame, np.ndarray]: A tuple containing features dataframe and labels array.
    """
    df = pd.read_parquet(file_name)
    y_train = df['y_train']
    features = df.drop(columns=['y_train'])

    return features, y_train


def shuffle_dataset(features: pd.DataFrame, y_train: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Shuffles the dataset.

    Parameters:
    features (pd.DataFrame): The features dataframe.
    y_train (np.ndarray): The labels array.

    Returns:
    Tuple[pd.DataFrame, np.ndarray]: A tuple containing shuffled features and labels.
    """
    df = features
    df = df.assign(y_train=y_train)
    df = df.sample(frac=1, replace=False)
    y_train = df['y_train']
    features = df.drop(columns=['y_train'])

    return features, y_train


class GeoportalClassifier:
    def __init__(self):
        """
        Initializes the GeoportalClassifier with an XGBClassifier model and parameter grid.
        """
        self.model = XGBClassifier(n_estimators=100, max_depth=8, tree_method="hist", device="cuda")
        self.param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 8, 16],
        }

    def show_feature_importances(self) -> None:
        """
        Displays a bar plot of feature importances.
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        columns = np.array(COLUMNS)

        plt.figure(figsize=(12, 8))
        plt.title("Feature importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), columns[indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.xlim([-1, len(importances)])
        plt.show()

    def save(self, filename: str) -> None:
        """
        Saves the model to a file using joblib.

        Parameters:
        filename (str): The name of the file to save.
        """
        joblib.dump(self.model, filename)

    def load(self, filename: str) -> None:
        """
        Loads the model from a file using joblib.

        Parameters:
        filename (str): The name of the file to load.
        """
        self.model = joblib.load(filename)

    @staticmethod
    def grid_search_results_to_csv(grid_search: GridSearchCV) -> None:
        """
        Converts grid search results to CSV format and saves to a file.

        Parameters:
        grid_search (GridSearchCV): The grid search object.
        """
        results = pd.DataFrame(grid_search.cv_results_)
        sorted_results = results.sort_values(by='rank_test_score')
        sorted_results.to_csv('grid_search_results.csv', index=False)

    @staticmethod
    def plot_grid_search_results(grid_search: GridSearchCV) -> None:
        """
        Plots grid search results as a heatmap.

        Parameters:
        grid_search (GridSearchCV): The grid search object.
        """
        results = pd.DataFrame(grid_search.cv_results_)

        scores = results.pivot(index='param_max_depth', columns='param_n_estimators', values='mean_test_score')

        plt.figure(figsize=(8, 6))
        sns.heatmap(scores, annot=True, cmap="viridis", fmt=".3f")

        plt.title('Grid Search CV Results')
        plt.xlabel('n_estimators')
        plt.ylabel('max_depth')
        plt.savefig('grid_search_results.png')

    def classify(self, las_file_manager: PointCloudManager) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classifies points using the model.

        Parameters:
        las_file_manager (PointCloudManager): The point cloud manager object.

        Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing true labels and predicted labels.
        """
        ind_filtered = las_file_manager.filter_outliers()
        ind_overlapped = np.where(~np.isin(las_file_manager.original_classifications, [12]))[0]
        ind = np.intersect1d(ind_filtered, ind_overlapped)

        values_dict = las_file_manager.get_model_values(ind)

        ind_test = np.where(np.isin(las_file_manager.original_classifications, [2, 3, 7, 9, 12]))[0]

        las_file_manager.original_classifications[ind_test] = 0
        y_test = las_file_manager.original_classifications[las_file_manager.ind]
        unique_classifications = np.unique(y_test)
        for i, unique_classification in enumerate(unique_classifications):
            y_test[y_test == unique_classification] = i

        features = pd.DataFrame(values_dict)
        features = features.apply(pd.to_numeric, errors='coerce')

        y_pred = self.model.predict(features)
        las_file_manager.classifications = y_pred

        return y_test, y_pred

    def grid_search(self, features: pd.DataFrame, y_train: np.ndarray) -> None:
        """
        Performs grid search to find the best hyperparameters.

        Parameters:
        features (pd.DataFrame): The features dataframe.
        y_train (np.ndarray): The labels array.
        """
        # features, y_train = self.ShuffleDataset(features, y_train)
        # n_jobs = multiprocessing.npu_count() - 1
        # grid_search = GridSearchCV(self.model, self.param_grid, cv=5, verbose=2, n_jobs=n_jobs)

        X_train = np.array(features)
        grid_search = GridSearchCV(self.model, self.param_grid, cv=2, verbose=2)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_

        self.grid_search_results_to_csv(grid_search)
        self.plot_grid_search_results(grid_search)
        self.model.set_params(**best_params)

    def train_model(self, las_file_manager: PointCloudManager, file_name: Optional[str] = None,
                    read: Optional[bool] = None) -> None:
        """
        Trains the model using the provided data.

        Parameters:
        las_file_manager (PointCloudManager): The point cloud manager object.
        file_name (Optional[str]): The name of the file to load or save features. Cannot be None if read is not None.
        read (Optional[bool]): Determines the operation mode if :
            - If True, reads features from the specified file.
            - If False, saves features to the specified file.
            - If None, trains the model without saving features to a file.
        """
        if file_name is not None and read is True:
            features, y_train = load_features(file_name)
        else:
            ind_filtered = las_file_manager.filter_outliers()

            ind_overlapped = np.where(~np.isin(las_file_manager.classifications, [2, 3, 7, 9, 12]))[0]
            ind = np.intersect1d(ind_filtered, ind_overlapped)

            values_dict = las_file_manager.get_model_values(ind, ground_classifications=[2])
            features = pd.DataFrame(values_dict)

            features = features.apply(pd.to_numeric, errors='coerce')

            y_train = las_file_manager.classifications
            y_train = np.array(y_train, order='C')

            if file_name is not None and read is False:
                save_features(features, y_train, file_name)

        start_time = time.time()
        self.grid_search(features, y_train)
        print("Grid search time:", time.time() - start_time)

        X_train = np.array(features, order='C')
        start_time = time.time()
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train)])
        print("training time:", time.time() - start_time)

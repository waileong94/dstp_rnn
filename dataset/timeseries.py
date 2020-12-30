
from torch.utils.data import Dataset

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import collections
from torch.utils.data._utils.collate import default_collate
Feature = collections.namedtuple('Feature', 'X y_prev y_target')
Feature.__new__.__defaults__ = (None,) * 3


class TimeSeriesDataset(object):
    def __init__(self, data, categorical_cols, target_col, seq_length, prediction_window=1,drop_col = False):
        '''
        :param data: dataset of type pandas.DataFrame
        :param categorical_cols: name of the categorical columns, if None pass empty list
        :param target_col: name of the targeted column
        :param seq_length: window length to use
        :param prediction_window: window length to predict
        '''
        self.data = data
        self.categorical_cols = categorical_cols
        self.numerical_cols = list(set(data.columns) - set(categorical_cols) - set(target_col))
        self.target_col = target_col
        self.seq_length = seq_length
        self.prediction_window = prediction_window
        self.preprocessor = None
        self.drop_target = drop_col
        
        if self.drop_target:
            self.input_size = self.data.shape[1] - len(self.target_col)
        else:
            self.input_size = self.data.shape[1]

    def preprocess_data(self):
        '''Preprocessing function'''
        if self.drop_target:
            X = self.data.drop(self.target_col, axis=1)
            y = self.data[self.target_col]
        else:
            X = self.data
            y = self.data[self.target_col]

        self.preprocess = ColumnTransformer(
            [("scaler", StandardScaler(), self.numerical_cols),
             ("encoder", OneHotEncoder(), self.categorical_cols)],
            remainder="passthrough"
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
        if self.preprocessor is not None:
            X_train = self.preprocessor.fit_transform(X_train)
            X_test = self.preprocessor.transform(X_test)

        if self.target_col:
            return X_train.values, X_test.values, y_train.values, y_test.values
        return X_train, X_test

    def frame_series(self, X, y=None):
        '''
        Function used to prepare the data for time series prediction
        :param X: set of features
        :param y: targeted value to predict
        :return: TensorDataset
        
            nb_obs = 100
            seq_length = 10
            prediction_window = 1
            i loop (1,100-10-1 = 89)
            
            i = 1
            x = x[1:1+10] > x[1:11]
            y = y[1+10:1+10+1] > y[11:12]
            y_list = y[1 + 10 - 1 : 1+ 10 +1-1] > y[10:11]
        '''
        nb_obs, nb_features = X.shape
        features, target, y_prev = [], [], []
        batch = []
        # for i in range(0, nb_obs - self.seq_length - self.prediction_window):
        #     features.append(torch.FloatTensor(X[i:i + self.seq_length, :]).unsqueeze(0))

        # features_var = torch.cat(features)

        # if y is not None:
        #     for i in range(0, nb_obs - self.seq_length - self.prediction_window):

        #         target.append(
        #             torch.tensor(y[(i + self.seq_length):(i + self.seq_length + self.prediction_window)]).unsqueeze(0))
        #         y_prev.append(
        #             torch.tensor(y[i:i + self.seq_length]).unsqueeze(0))
        #     target_var, y_prev_var = torch.cat(target), torch.cat(y_prev)
        #     return TensorDataset(features_var, target_var, y_prev_var)
        # return TensorDataset(features_var)
        
        for i in range(0, nb_obs - self.seq_length - self.prediction_window):
            batch_feature = torch.FloatTensor(X[i:i + self.seq_length-1, :])
            if y is not None:
                batch_target = torch.tensor(y[(i + self.seq_length-1):(i + self.seq_length + self.prediction_window-1)])
                batch_yhist = torch.tensor(y[i:i + self.seq_length-1])
         
            batch.append(Feature(X = batch_feature,y_prev = batch_yhist,y_target = batch_target))
            
            features.append(batch_feature)
            target.append(batch_target)
            y_prev.append(batch_yhist)   
                    
        # results = Feature(*(default_collate(samples) for samples in zip(*batch)))
        return batch
    
    def get_loaders(self, batch_size: int):
        '''
        Preprocess and frame the dataset
        :param batch_size: batch size
        :return: DataLoaders associated to training and testing data
        '''
        X_train, X_test, y_train, y_test = self.preprocess_data()

        train_dataset = self.frame_series(X_train, y_train)
        test_dataset = self.frame_series(X_test, y_test)

        train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        return train_iter, test_iter




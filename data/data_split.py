from sklearn.model_selection import train_test_split
import pandas as pd
from collections import namedtuple

train_test_data = namedtuple(typename='train_test_data',
                             field_names=['X_train','X_test',
                                          'y_train','y_test'
                                          ]
                             )

def split_data(data: pd.DataFrame, 
               features: list, 
               target: str, 
               test_size: int = 0.3, 
               random_state=2022):
    if isinstance(features, str):
        features = [features]
    
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state
                                                        )
    return train_test_data(X_train=X_train, 
                           X_test=X_test,
                           y_train=y_train,
                           y_test=y_test
                           )

    

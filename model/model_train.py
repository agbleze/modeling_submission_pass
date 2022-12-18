
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score,cross_validate
from preprocess_pipeline import PipelineBuilder
import pandas as pd 
from joblib import dump, load
from sklearn.metrics import classification_report
from ..arguments import args



pipeline = PipelineBuilder()

model_pipeline = pipeline.build_model_pipeline()

class Model(object):
    def __init__(self, training_features: pd.DataFrame, training_target_variable: pd.DataFrame,
                 test_features: pd.DataFrame = None, test_target_variable: pd.DataFrame = None,
                 ):
        self.training_features = training_features
        self.training_target_variable = training_target_variable
        self.test_features = test_features
        self.test_target_variable = test_target_variable
    
    def model_fit(self, model_pipeline = model_pipeline):
        self.model_pipeline = model_pipeline
        
        self.model_fitted = self.model_pipeline.fit(self.training_features, self.training_target_variable)
        return self.model_fitted
      
    
    def cross_validate(self, metric: str = 'accuracy', cv = 10) -> list:
        self.model = self.model_fitted
        self.scores = cross_val_score(estimator=self.model, X=self.training_features, 
                                        y=self.training_target_variable,
                                        scoring=metric, cv=cv
                                    )
        return self.scores
             

    @property
    def compute_mean_score(self):
        return self.scores.mean()
        
    
    @property
    def predict(self, test_features = None):
        if test_features is None:
            predictions = self.model.predict(self.test_features)
        else:
            self.test_features = test_features
            predictions = self.model.predict(self.test_features)
        return predictions
    
    @property
    def model_evaluate(self):
        y_pred = self.predict()
        
        model_report = classification_report(y_true=self.test_features, y_pred=y_pred)
        return model_report
    
    
    def save_model(self):
        dump(value=self.model_pipeline, filename='model.model')
        print('model successfully saved')




# # fit model on train data

# def model_fit(training_features: pd.DataFrame, target_variable: pd.DataFrame,
#               model_pipeline):
#     model_pipeline.fit(training_features, target_variable)
    
    
# logit_model_pipeline = make_pipeline(preprocess_pipeline,
#                                     LogisticRegression(class_weight='balanced')
#                                     )

# logit_model_pipeline.fit(X_train, y_train)


# # using model for prediction 

# y_pred= logit_model_pipeline.predict(X_test)







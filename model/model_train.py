#%%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score,cross_validate
from .preprocess_pipeline import PipelineBuilder
import pandas as pd 
from joblib import dump, load
from sklearn.metrics import classification_report
from arguments import args
from .candidate_models import candidate_classifiers
from typing import List, Tuple
import plotly.express as px



pipeline = PipelineBuilder()

model_pipeline = pipeline.build_model_pipeline()

class Model(object):
    def __init__(self, training_features: pd.DataFrame, 
                 training_target_variable: pd.DataFrame,
                 test_features: pd.DataFrame = None, test_target_variable: pd.DataFrame = None,
                 metric: str = 'accuracy', cv: int = 10
                 ):
        self.training_features = training_features
        self.training_target_variable = training_target_variable
        self.test_features = test_features
        self.test_target_variable = test_target_variable
        self.metric = metric
        self.cv = cv
    
    #@classmethod
    def model_fit(self, model_pipeline = model_pipeline):
        self.model_pipeline = model_pipeline
        
        self.model_fitted = self.model_pipeline.fit(self.training_features, self.training_target_variable)
        return self.model_fitted
      
    
    def cross_validate_score(self) -> list:
        self.model = self.model_fitted
        self.scores = cross_val_score(estimator=self.model, X=self.training_features, 
                                        y=self.training_target_variable,
                                        scoring=self.metric, cv=self.cv
                                    )
        return self.scores#['test_score']
             

    @property
    def compute_mean_score(self):
        return self.scores.mean()
        
    
    #@classmethod
    def predict_values(self, test_features = None):
        if test_features is not None:
            self.test_features = test_features
        self.predictions = self.model_fitted.predict(self.test_features)
        return self.predictions
    
    #@property
    def evaluate_model(self, y_true = None, y_pred = None):
        if y_true is None:
            y_true = self.test_target_variable
        else:
            y_true = y_true
            
        if y_pred is None:
            y_pred = self.predictions
        else:
            y_pred = y_pred
        
        model_report = classification_report(y_true=y_true, 
                                             y_pred=y_pred
                                             )
        return model_report
    
    
    def save_model(self, model = None, filename = args.model_store_path):
        if model is not None:
            model_to_save = model
        model_to_save = self.model_fitted
        dump(value=model_to_save, filename=filename)
        print('model successfully saved')
        
        
    def run_classifiers(self, estimators: dict = candidate_classifiers
                        ):
        df_test_score_list = []
        df_fit_time_list = []
        df_score_time_list = []
        
        for model_name, mod_pipeline in estimators.items():              
            score = cross_validate(estimator=mod_pipeline,
                                     X=self.training_features,
                                     y=self.training_target_variable, 
                                     cv=self.cv,
                                    scoring=self.metric,
                                    return_train_score=False
                                    )
            test_score_dict = {'model': model_name, 'test_score': score['test_score']}
            fit_time_dict = {'model': model_name, 'fit_time': score['fit_time']}
            score_time_dict = {'model': model_name, 'score_time': score['score_time']}
            
            df_test_score = pd.DataFrame(data=test_score_dict)
            df_fit_time = pd.DataFrame(data=fit_time_dict)
            df_score_time = pd.DataFrame(data=score_time_dict)
            
            df_test_score_list.append(df_test_score)
            df_fit_time_list.append(df_fit_time)
            df_score_time_list.append(df_score_time)
            
            
        self.cv_test_score_df = pd.concat(df_test_score_list)
        self.cv_fit_time_df = pd.concat(df_fit_time_list)
        self.cv_score_time_df = pd.concat(df_score_time_list)
        
        return {'cv_test_score': self.cv_test_score_df, 
                'cv_fit_time': self.cv_fit_time_df, 
                'cv_score_time': self.cv_score_time_df
                }
        
        
    def plot_classifers_cv_results(self):
        test_score_fig = px.box(data_frame=self.cv_test_score_df, x='model', y='test_score', 
                                color='model',
                                title=f'Test score ({self.metric}) {self.cv} fold cross validation on Classifiers',
                                template='plotly_dark', height=700,
                                )
        
        fit_time_fig = px.box(data_frame=self.cv_fit_time_df, x='model', y='fit_time', 
                                color='model',
                                title=f'Fit time of various Classifiers',
                                template='plotly_dark', height=700,
                                )
        
        score_time_fig = px.box(data_frame=self.cv_score_time_df, x='model', y='score_time', 
                                color='model',
                                title=f'score time of various Classifiers',
                                template='plotly_dark', height=700,
                                )
        
        return {'boxplot_classifiers_test_score': test_score_fig, 
                'boxplot_classifiers_fit_time': fit_time_fig,
                'boxplot_classifiers_score_time': score_time_fig
                }
        
    def get_best_model_name(self, models_fit_df = None, colname_for_models: str = 'model', 
                   colname_for_score: str = 'test_score') -> str:
            """Accepts data for models and test score and returns the name of best model detected after cross validation made from running 
                from running all candidate models

            Args:
                models_fit_df (pd.DataFrame): Dataframe with columns for model name and test score
                colname_for_models (str): Name of column with values as model names
                colname_for_score (str): Name of column with values as test score (accuracy)
                
            Returns: 
                The name of the best algorithm (model)
            """
            if models_fit_df is not None:
                self.model_cv_acc_df = models_fit_df
            self.model_cv_acc_df = self.cv_test_score_df
            self.model_mean_cv_acc_df = (self.model_cv_acc_df.groupby(colname_for_models)
                                         [colname_for_score]
                                         .mean().reset_index()
                                         )
            max_acc = self.model_mean_cv_acc_df[colname_for_score].max()
            best_model_name = (self.model_mean_cv_acc_df[self.model_mean_cv_acc_df[colname_for_score]==max_acc]
                               [colname_for_models].item()
                               )
            return best_model_name

    @property
    def best_model_fitted(self, candidate_models = candidate_classifiers):
        """Retrieves best candidate model pipeline and fit on data
        
        Returns:
            Model pipeline

        """
        best_model_name = self.get_best_model_name()
        
        best_model_pipeline = candidate_models[best_model_name]
        print("best candidate model being fit on data")
        _best_model_fitted = self.model_fit(model_pipeline=best_model_pipeline)
        print("model fitting completed")
        return _best_model_fitted
    
    def save_best_model(self):
        self.save_model(model=self.best_model_fitted, 
                        filename=args.best_model_store_path
                        )


            
                
        


  

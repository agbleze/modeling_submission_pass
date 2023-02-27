from sklearn.metrics import roc_curve, auc, precision_recall_curve
from model.candidate_models import candidate_classifiers
import pandas as pd
from model.model_utils import model_fit, classification_report_as_df
from plots.plot_graph import plot_test_data_precision_recall, plot_test_data_roc_curve, plot_table


class ClassifiersEvaluation(object):
    def __init__(self, test_features: pd.DataFrame, 
               test_target_variable, training_features: pd.DataFrame,
               training_target_variable: pd.DataFrame,
               estimators: dict = candidate_classifiers,
                ):
        """This class evaluates a list of classifiers provided to it and returns 
            their ROC curves and precision recall curves for test set. It act as a 'turn knob' 
            for running all classifiers to explore results further. Note that it should be used 
            in that automated manner and not for model development 

        Args:
            test_features (pd.DataFrame): Predictors in the test dataset
            test_target_variable (_type_): Target variable in the test dataset
            training_features (pd.DataFrame): Predictors in the training dataset
            training_target_variable (pd.DataFrame): Target variable in the training dataset
            estimators (dict, optional): _description_. Defaults to candidate_classifiers. contains models pipelines 
                                        as values and and their names as keys
        """
        self.estimators = estimators
        self.test_features = test_features
        self.test_target_variable = test_target_variable
        self.training_features = training_features
        self.training_target_variable = training_target_variable
    
    def evaluate_classifiers(self):
        self.classifiers_roc_curves = {}
        self.classifiers_precision_recall_curves = {}
        self.all_models_fitted = {}
        for model_name, mod_pipeline in self.estimators.items():
            
            self.model_fitted = model_fit(training_features=self.training_features,
                                          training_target_variable=self.training_target_variable,
                                          model_pipeline = mod_pipeline
                                        )
            self.all_models_fitted[model_name] = self.model_fitted
            
            test_proba_score = self.model_fitted.predict_proba(self.test_features)[:,1]
            print(f'Successfully fit {model_name} on training data') 
            
            fpr, tpr, roc_thresholds = roc_curve(self.test_target_variable, test_proba_score)
            print(f'Successfully computed ROC Curve on test data')
            
            precision, recall, prec_recall_thresholds = precision_recall_curve(y_true=self.test_target_variable,
                                                                                probas_pred=test_proba_score
                                                                            )
            print(f'Successfully computed precision recall curve')
            
            
            test_roc_curve = plot_test_data_roc_curve(false_positive_rate=fpr, true_positive_rate=tpr,
                                                      title=f'{model_name} ROC Curve'
                                                        )
            print('plot roc curve for test data')
            
            test_precision_recall_curve = plot_test_data_precision_recall(recall=recall, precision=precision,
                                                                          title=f'{model_name} Precision-Recall curve'
                                                                          )
            print('Successfully plot precision recall curve for test data')
            self.classifiers_roc_curves[model_name] = test_roc_curve
            self.classifiers_precision_recall_curves[model_name] = test_precision_recall_curve
        return {'classifiers_roc_curves': self.classifiers_roc_curves, 
                'classifiers_precision_recall_curves': self.classifiers_precision_recall_curves
                }
    
    
    def get_classifiers_test_classification_report_as_df(self):
        self.models_report = {}
        for model_name in self.all_models_fitted.keys():
            model_fitted = self.all_models_fitted[model_name]
            y_pred = model_fitted.predict(X=self.test_features)
            report_df = classification_report_as_df(y_true=self.test_target_variable, 
                                                    y_pred=y_pred,
                                                    class_names={ '0': 'not_pass','1': 'pass' }
                                                    )
            
            report_formated = report_df.reset_index().rename(columns={'index': 'class'})

            self.models_report[model_name] = report_formated
        return self.models_report
    
    def plot_classification_reports(self):
        classification_report_plot = {}
        for model_name in self.models_report.keys():
            model_report_df = self.models_report[model_name]
            table = plot_table(data=model_report_df)
            classification_report_plot[model_name] = table
        return classification_report_plot
            
    
    
    
    
    
    
    
    
    
            
            
            
            
      
    
    

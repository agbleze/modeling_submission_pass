from .preprocess_pipeline import PipelineBuilder
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
import pandas as pd

pipeline = PipelineBuilder()

model_pipeline = pipeline.build_model_pipeline()


def model_fit(training_features, 
             training_target_variable,
              model_pipeline = model_pipeline,
              ):
        
        model_fitted = model_pipeline.fit(training_features, 
                                        training_target_variable
                                        )
        return model_fitted
    
    
def classification_report_as_df(y_true, y_pred,
                                #output_dict=True, 
                                class_names: dict = {'0': 'not_pass', '1': 'pass'}
                                ):
    """Returns classification report in the form of a dataframe

    Args:
        y_true (_type_, optional): Actual outcome Defaults to None.
        y_pred (_type_, optional): Predicted outcome. Defaults to None.
        class_names (_type_, optional): Dictionary of how the outcome classes are labelled in the dataset
                                        and their actual labels. Example, 0 and used to depict classes of 
                                        'No' and 'Yes' will have this parameter specified as {0: 'No', 1:'Yes'}.
                                        Defaults to {'0': 'not_pass', '1': 'pass'}.
    """
    class_report = classification_report(y_true=y_true, y_pred=y_pred,
                        output_dict=True
                        )
    report_df = pd.DataFrame(class_report).transpose().iloc[:2,:]
    report_df_rename = report_df.rename(index={'0': 'not_pass', '1': 'pass'})
    return report_df_rename


# make cross validation y_pred
# use cross validation y_pred to make classification report
# group and find mean of each classification metric  


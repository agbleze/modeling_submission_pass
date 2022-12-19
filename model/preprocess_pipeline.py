#%% import all models required for the analysis
from arguments import args
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
#from argparse import Namespace
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns




one = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

# preprocess_pipeline =  make_column_transformer((scaler, args.selected_numeric_features),
#                                                 (one, args.categorical_features)
#                                                 )

# logit_model_pipeline = make_pipeline(preprocess_pipeline,
#                                     LogisticRegression(class_weight='balanced')
#                                     )


class PipelineBuilder(object):
    def __init__(self, num_features: list = args.selected_numeric_features,
                 categorical_features: list = args.categorical_features):
        self.num_features = num_features
        self.categorical_features = categorical_features
   
    
    @classmethod
    def build_data_preprocess_pipeline(cls):
        cls.preprocess_pipeline =  make_column_transformer((scaler, args.selected_numeric_features),
                                                        (one, args.categorical_features)
                                                      )
        
        return cls.preprocess_pipeline
        
    
    @classmethod
    def build_model_pipeline(cls, model = None, preprocess_pipeline = None,
                             class_weight='balanced'):
        if (model == None):
            cls.model = LogisticRegression(class_weight=class_weight)
        else:
            cls.model = model
            
        if (preprocess_pipeline == None):
            cls.preprocess_pipeline = cls.preprocess_pipeline
        else:
            cls.preprocess_pipeline = preprocess_pipeline
            
        logit_model_pipeline = make_pipeline(cls.preprocess_pipeline,
                                             cls.model
                                            )         
            
        return logit_model_pipeline
        



# %%

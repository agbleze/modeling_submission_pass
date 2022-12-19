from xgboost import XGBRFClassifier, XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, BaggingClassifier, 
                              HistGradientBoostingClassifier,ExtraTreesClassifier,
                              VotingClassifier,
                              )
#from sklearn.neighbors import KNeighborsClassifier
from .preprocess_pipeline import PipelineBuilder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from typing import List, Tuple
from sklearn.model_selection import cross_val_score,cross_validate
import pandas as pd


pipeline = PipelineBuilder()

preprocess_pipeline = pipeline.build_data_preprocess_pipeline()

svc_rbf = SVC(kernel='rbf', class_weight='balanced')
svc_linear = SVC(kernel='linear', class_weight='balanced')
svc_poly = SVC(kernel='poly', class_weight='balanced')
rfc = RandomForestClassifier(class_weight='balanced')

decision_tree = DecisionTreeClassifier(class_weight='balanced')
extra_decision_tree = ExtraTreeClassifier(class_weight='balanced')
svc_rbf_pipeline = pipeline.build_model_pipeline(model=svc_rbf)
svc_linear_pipeline = pipeline.build_model_pipeline(model=svc_linear)
svc_poly_pipeline = pipeline.build_model_pipeline(model=svc_poly)
rfc_pipeline = pipeline.build_model_pipeline(model=rfc)
decision_tree_pipeline = pipeline.build_model_pipeline(model=decision_tree)
extra_decision_tree_pipeline = pipeline.build_model_pipeline(model=extra_decision_tree)



candidate_classifiers = [("Extra decision tree", extra_decision_tree_pipeline),
                        ("Decision tree", decision_tree_pipeline),
                        ("Radom forest classifier", rfc_pipeline),
                        ("SVC poly", svc_poly_pipeline),
                        ("SVC linear", svc_linear_pipeline),
                        ("SVC rbf", svc_rbf_pipeline)
                        ]




def run_classifiers(cv: int = 10, scoring: str ='accuracy',
                    estimators: List[Tuple] = candidate_classifiers
                    ):
    df_test_score_list = []
    df_fit_time_list = []
    df_score_time_list = []
    for model_name, mod_pipeline in estimators:              
        score = cross_validate(estimator=mod_pipeline,
                               X=X, y=y, cv=cv,
                                scoring=scoring,
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
        
        
    cv_test_score_df = pd.concat(df_test_score_list)
    cv_fit_time_df = pd.concat(df_fit_time_list)
    cv_score_time_df = pd.concat(df_score_time_list)
    
    return cv_test_score_df, cv_fit_time_df, cv_score_time_df
    




# >>> cv_results = cross_validate(lasso, X, y, cv=3)
# >>> sorted(cv_results.keys())
# ['fit_time', 'score_time', 'test_score']
# >>> cv_results['test_score']
# array([0.3315057 , 0.08022103, 0.03531816])




# ridge = RidgeCV()
# all_models = [("Random Forest", rf_pipeline),
#               ("Lasso", lasso_pipeline),
#               ("Hist Gradient Boosting", hgb_pipeline),
#               ("Extreme Gradient Boosting Random Forest", xgb_pipeline),
#               ("KNN k=5", knn_pipeline),
#               #("Ridge", rd_pipeline),
#               ("SVR rbf", svr_rbf_pipeline)
#               ]
# stack_regressors = StackingRegressor(estimators=all_models, final_estimator=ridge)

# candidate_models = all_models.copy()

# candidate_models.extend([("Ridge", rd_pipeline), ('stacked_models', stack_regressors)])

# def plot_models_cv_test_error(cv: int = 10, scoring: str ='neg_mean_squared_error',
#                              estimators: List[Tuple] = candidate_models
#                              ):
#     test_score_list = []
#     for model_name, mod_pipeline in estimators:              
#         score = cross_validate(estimator=mod_pipeline,
#                                X=X, y=y, cv=cv,
#                                 scoring=scoring,
#                                 return_train_score=False
#                                 )
#         test_score_dict = {'model': model_name, 'test_score': -(score['test_score'])}
#         df = pd.DataFrame(data=test_score_dict)
#         df['test_RMSE'] = df['test_score'].apply(lambda x: math.sqrt(x))
#         df.drop(columns='test_score', inplace=True)
#         test_score_list.append(df)
#     cv_score_test_df = pd.concat(test_score_list)
#     #print(cv_score_test_df)
#     mean_model_cv_rmse = cv_score_test_df.groupby('model')['test_RMSE'].mean().reset_index()
#     #print(mean_model_cv_rmse)
#     fig = px.box(data_frame=cv_score_test_df, x='model', y='test_RMSE', 
#                  color='model',# notched=True, 
#                  title=f'Test error of 10 fold cross validation on Models',
#                  template='plotly_dark', height=700,
#                  )
#     #fig.show()
    
#     fig1 = px.scatter(data_frame=mean_model_cv_rmse, x='model', 
#                       y='test_RMSE', color='model', symbol='model',
#             labels={'test_RMSE': 'Average of 10 CV RMSE'},
#             title='Average of 10 CV test RMSE for various models',
#             template='plotly_dark', height=700
#             )
#     fig1.update_traces(marker_size=15)
#     #fig1.show()
#     return {'test_rmse': fig, 'avg_test_rmse': fig1}


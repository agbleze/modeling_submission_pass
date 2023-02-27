#%% import
from utils.get_path import get_data_path
from arguments import args
from transform.data_transform import DataTransformer
from model.model_train import Model
from data.data_split import split_data
import pandas as pd


#%% get data path
datapath = get_data_path(folder_name='DS_Practical', file_name='all_merge_df.csv')

data = pd.read_csv(datapath)

#%% transform features in dataset
data_transformer = DataTransformer(data = data)

data_with_avg_answertime_min = data_transformer.convert_hours_to_minute()

data_workrate_added = data_transformer.define_work_rate(data_to_transform=data_with_avg_answertime_min)

data_with_extra_time = data_transformer.calculate_extra_time_min(data_to_transform=data_workrate_added)

data_cat_in_num = data_transformer.convert_categorical_to_numeric(data_with_extra_time)

data_target_variable_transformed = data_transformer.transform_target_variable(data=data_cat_in_num)


#%% split data

X_train, X_test, y_train, y_test = split_data(data=data_target_variable_transformed, 
                                              features=args.selected_predictors,
                                              target=args.target_variable_transformed
                                            )

#%% model 

model = Model(training_features=X_train, training_target_variable=y_train,)

#%%
model.model_fit()

#%%
cross_val_accuracy = model.cross_validate_score()

#%%
model.compute_mean_score

#%%
model.predict_values(test_features=X_test)

#%% evaluate model
print(model.evaluate_model(y_true=y_test))


#%%
class_report = model.model_report


#%%
report_df = pd.DataFrame(class_report).transpose().iloc[:2,:]#.rename(index={0: 'not_pass', 1: 'pass'})

#%%
report_df.rename(index={'0': 'not_pass', '1': 'pass'})



#%% save model 

model.save_model(filename=args.model_store_path)


#%%# Automate evaluation of various classifiers

classifiers_result = model.run_classifiers()

#%%

classifiers_result.keys()

#%%
classifiers_test_score = classifiers_result['cv_test_score']

classifier_fit_time = classifiers_result['cv_fit_time']

classifiers_score_time = classifiers_result['cv_score_time']




#%%
classifiers_plot = model.plot_classifers_cv_results()

#%%
classifiers_plot.keys()

#%%
classifiers_plot['boxplot_classifiers_test_score']


# %%
classifiers_plot['boxplot_classifiers_fit_time']

#%%
classifiers_plot['boxplot_classifiers_score_time']

#%%
model.get_best_model_name()

#%%
#model.fit_best_candidate_model()

model.best_model_fitted

#%%

model.save_best_model()


#%%

data_dict = {'progress_percent': 90, 'extra_time_min': 40, 'work_rate': 'normal'}

dummy_data = pd.DataFrame(data=data_dict, index=[0])

#%%

model.predict_values(dummy_data)

# %%

import joblib

#%%

best_model_loaded = joblib.load('model_store/best_model.model')

#%%

best_model_loaded.predict(dummy_data)

#%% using the ClassifiersEvaluation class
   
from model.classifiers_evaluations import ClassifiersEvaluation    

classifier_eval = ClassifiersEvaluation(test_features=X_test, test_target_variable=y_test, training_features=X_train,
                                        training_target_variable=y_train
                                        )

# %%
a = classifier_eval.evaluate_classifiers()

#%%

classifier_roc_curves = classifier_eval.classifiers_roc_curves

#%%
for model_name in classifier_roc_curves.keys():
  print(model_name)
  classifier_roc_curves[model_name].show() 
  
  
#%%
classifier_precision_recall_curves = classifier_eval.classifiers_precision_recall_curves

for model_name in classifier_precision_recall_curves.keys():
  print(model_name)
  classifier_precision_recall_curves[model_name].show()


#%%

classification_reports = classifier_eval.get_classifiers_test_classification_report_as_df()

#%% get classification reports for various models

for model_name in classification_reports.keys():
  print(model_name)
  classification_reports[model_name]

#%%

classification_plots = classifier_eval.plot_classification_reports()

#%%
for model_name in classification_plots.keys():
  print(model_name)
  classification_plots[model_name].show()
  
#classification_plots['Extra decision tree']

#%%
#report_format = classification_reports['Extra decision tree'].reset_index().rename(columns={'index': 'class'})

#%%

#from plots.plot_graph import plot_table

#plot_table(report_format)


#%%

  
#%%  testing cross_val_predict
# from sklearn.model_selection import cross_val_predict, cross_val_score,
# from model.candidate_models import candidate_classifiers
# from sklearn.metrics import accuracy_score

# #%%%
# randomforest = candidate_classifiers["Radom forest classifier"]


# #%%
# crossp_a = cross_val_predict(estimator=randomforest, X=X_train, y=y_train, cv=10)


# #%% 
# accuracy_score(y_true=y_train, y_pred=crossp_a)


# # %%
# cross_val_acc = cross_val_score(estimator=randomforest,X=X_train, y=y_train, cv=10)


# # %%
# cross_val_acc.mean()

# #%%
# from sklearn.model_selection import cross_validate

# #%%
# crossval_score = cross_validate(estimator=randomforest, 
#                                 X=X_train, y=y_train, cv=10,
#                                 return_estimator=True,
#                                 return_train_score=False, scoring='accuracy'
#                               )


# #%%
# len(crossval_score['estimator'])


# %%


  





# %%

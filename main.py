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




# %%

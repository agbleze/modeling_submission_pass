
#%%
from numbers_parser import Document
doc = Document("student_-_exercise_progress.numbers")
sheets = doc.sheets
tables = sheets[0].tables
rows = tables[0].rows()


#%%

import pandas as pd
from numbers_parser import Document
doc = Document("student_-_exercise_progress.numbers")
sheets = doc.sheets
tables = sheets[0].tables
data = tables[0].rows(values_only=True)
df = pd.DataFrame(data[1:], columns=data[0])
df.to_csv("tab1.csv")


#%%
table1 = sheets[1].tables
data1 = table1[0].rows(values_only=True)
df1 = pd.DataFrame(data1[1:], columns=data1[0])

df1.to_csv('tab2.csv')


#%%

table2 = sheets[2].tables

#%%
data2 = table2[0].rows(values_only=True)

#%%
df2 = pd.DataFrame(data2[1:], columns=data2[0])

df2.to_csv('tab3.csv')






"""
2. Submissions- We are interested in discovering about how student,
exercise and submission features contribute to the likelihood of a 
submission being passed or not. Create (or describe how you would create) an 
ML model which attempts this. What challenges do you face? What additional 
information would you like to have? Is your solution production-ready?


TODO:
1. Determin target variable
--- state
i. create binary feature for state
---- merge all values not 'approved' to be 'not approved
---- visualize state variable
---- state how 

2. Determin features (feature engineering)
--- features to use as it is
------ average_answer_time_in_hours
------  progress_percent
------  test_variable

3. engineered features
----- 
     Extratime_in_min  =  average_answer_time_in_hours  - max_task_time_in_min
     if overtime < 0 then = 0
     //// average_answer_time_in_hours should be converted to minutes
     
----- Rate_of_work = if the average_answer_time_in_hours(mins) < min_task_time_in_min then 'fast'
                    if min_task_time_in_min <= average_answer_time_in_hours(mins) <= max_task_time_in_min then 'normal'
                    if average_answer_time_in_hours(mins) > max_task_time_in_min then 'slow'


numeric_features = ['average_answer_time_in_hours', 'progress_percent', 
'min_task_time_in_min', 'max_task_time_in_min' 
]

4. Handling unbalanced data
--- oversampling of underrepresented class



TODO: MODELLING
1. MODEL 1 -- develop a model with only features already in dataset without 
handling unbalanced data


2. MODEL 2 -- Handle unbalanced data for model 1


3. Undertake feature engineering and use it to improve MODEL 2 to get MODEL 3


"""






"""
1. Students- Here we provide you with 2 datasets associated with 2 differently 
modeled progress. Note: the data is example data and doesn’t show the 
students’ actual performance rates. With the different progress models we are aiming 
to improve our performance indicators. One performance indicator is consistency in time 
between submissions for the students. We are interested in evaluating if either of these
two models is performing better in terms of time between submissions. How would you evaluate this?


TODO:
1. Identify the models that are being used

--- test_variable

2. Identify variable for time used in submission
--- average_answer_time_in_hours

3. merge data from the different tabs
--- merge tab 2 and tab3 on student_id to have student_id, test_variable,
    submission_id, average_answer_time_in_hours,

4. Do hypothesis test to find out if there is significant difference in average_answer_time_in_hours
between the models (A and B) in test_variable

5. If there is, then the model with the least average_answer_time_in_hours is doing better












""" 




"""


 
"""


# %%
from imblearn import over_sampling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             r2_score, precision_score, recall_score, 
                             roc_auc_score, f1_score, roc_curve, auc, 
                             confusion_matrix, classification_report
                             )
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np


#%%

df2_df_merge = df2.merge(right=df, on='exercise_id', how='left')


#%%
all_merge_df =  df2_df_merge.merge(right=df1, how='left', on='student_id')

#%%
all_merge_df.to_csv("all_merge_df.csv")


#%%
target_variable = 'state'
predictor_variables =['average_answer_time_in_hours', 'progress_percent', 'test_variable']


#y = all_merge_df[target_variable]
X = all_merge_df[predictor_variables]

"""merge 'waiting_for_review', 'almost_there', 'not_yet', 'a_little_more'  """

not_yet_passed = ['waiting_for_review', 'almost_there', 'not_yet', 'a_little_more']

categorize_state = lambda x: 0 if x in not_yet_passed else 1

#%%
transform_test_variable = lambda x: 0 if x == 'a' else 1


#%% apply transform data
all_merge_df['state_category'] = all_merge_df[target_variable].apply(categorize_state)


#%%
all_merge_df['test_variable_transform'] = all_merge_df[predictor_variables[2]].apply(transform_test_variable)


#%%
from argparse import Namespace

args = Namespace(
    target_variable = ['state_category'],
    predictor_variables = ['average_answer_time_in_hours', 'progress_percent', 'test_variable_transform'],
    numeric_features = ['progress_percent', 'average_answer_time_in_min', 'extra_time_min',
                        #'min_task_time_in_min', 'max_task_time_in_min'
                        ],
    categorical_features = ['work_rate'],
    binary_feature = ['test_variable_transform'],
    predictors = ['progress_percent', 'average_answer_time_in_min', 'extra_time_min',
                  'work_rate', 'test_variable_transform'],
    selected_predictors = ['progress_percent', 'extra_time_min', 'work_rate',
                        ],
    selected_numeric_features = ['progress_percent', 'extra_time_min']

)

#%%

#print(args.target_variable.extend((args.categorical_features, args.binary_feature, args.numeric_features)))






# %%
all_merge_df['state_category'].value_counts().plot(kind='bar')

#%% 
y = all_merge_df['state_category']
X = all_merge_df[args.predictor_variables]

#%% split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022)


# %%  ################ MODEL 1: available features with unbalanced data
logit = LogisticRegression(class_weight='balanced')

#%%

logit.fit(X_train, y_train)


#%%
y_pred = logit.predict(X_test)

#%%
accuracy_score(y_true=y_test, y_pred=y_pred)


#%%
r2_score(y_true=y_test, y_pred=y_pred)

#%% TP / TP + FP 
precision_score(y_true=y_test, y_pred=y_pred)


# %% TP / TP + FN
recall_score(y_true=y_test, y_pred=y_pred)


#%% f1_score = 2*(precision * recall) / (precision + recall)

f1_score(y_true=y_test, y_pred=y_pred)


#%%
confusion_matrix(y_true=y_test, y_pred=y_pred)


#%%
print(classification_report(y_true=y_test, y_pred=y_pred))



#%%
logit.coef_

#%%

logit.feature_names_in_




# %% ############## MODEL 2: Available features with balanced data -- class_weight = 'balanced'

logit_balanaced = LogisticRegression(class_weight='balanced')


#%%
logit_balanaced.fit(X=X_train, y=y_train)


#%%
y_pred_b = logit_balanaced.predict(X_test)

#%%
print(classification_report(y_true=y_test, y_pred=y_pred_b))

#%%
print(f"Balanced accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred_b)}")


#%%
print(f"Balanced data R2: {r2_score(y_true=y_test, y_pred=y_pred_b)}")

#%%
print(f"Balanced data precision score: {precision_score(y_test, y_pred_b)}")
print(f"Balance data recall score: {recall_score(y_test, y_pred_b)}")
print(f"Balance data features: {logit_balanaced.feature_names_in_}")
print(f"Balanced data coef: {logit_balanaced.coef_}")

# %%
roc_auc_score(y_test, y_pred_b)





#%%  ###############  MODEL 3: create features ###########################

## Extratime_in_min  =  average_answer_time_in_hours  - max_task_time_in_min


#1. convert time in hours to mins

all_merge_df['average_answer_time_in_min'] = all_merge_df['average_answer_time_in_hours'] * 60


#%%

#all_merge_df.drop(columns='average_answer_time_in_min', inplace=True)

#%%
convert_hours_to_minute = lambda x: x * 60

all_merge_df['average_answer_time_in_min'] = all_merge_df['average_answer_time_in_hours'].apply(convert_hours_to_minute)



#%%
rescale_from_zero =  lambda x: 0 if x < 0 else x


# %%
# 2. compute extra time
all_merge_df['average_answer_time_in_min'] - all_merge_df['max_task_time_in_min']

def calculate_extra_time_min(data: pd.DataFrame = None, max_time_min="max_task_time_in_min", 
                             actual_time_min="average_answer_time_in_min",
                             make_negative_time_zero_minutes: callable = rescale_from_zero
                             ):
    data['time_diff'] = data['average_answer_time_in_min'] - data['max_task_time_in_min']
    
    data['extra_time_min'] = data['time_diff'].apply(make_negative_time_zero_minutes)
    
    return data
    
    
#%% 

calculate_extra_time_min(data=all_merge_df)

    


#%%##

"""
work_rate = if the average_answer_time_in_hours(mins) < min_task_time_in_min then 'fast'
                    if min_task_time_in_min <= average_answer_time_in_hours(mins) <= max_task_time_in_min then 'normal'
                    if average_answer_time_in_hours(mins) > max_task_time_in_min then 'slow'

"""


def define_work_rate(data: pd.DataFrame, 
                     min_time_alloted: str = 'min_task_time_in_min', 
                     max_time_alloted: str = 'max_task_time_in_min',
                     actual_time_used: str = 'average_answer_time_in_min'
                     ):
    #for i in len(data[actual_time_used]):
    
    data['work_rate'] = np.where((data[actual_time_used] < data[min_time_alloted]), 'fast', np.nan)
    data['work_rate'] = np.where((data[actual_time_used] >= data[min_time_alloted]) & (data[actual_time_used] <= data[max_time_alloted]), 'normal', data['work_rate'])
    data['work_rate'] = np.where((data[actual_time_used] > data[max_time_alloted]), 'slow', data['work_rate'])
    return data
    
    # if data[actual_time_used] < data[min_time_alloted]:
    #     data['work_rate'] = 'fast'
    # elif (data[actual_time_used] >= data[min_time_alloted]) & (data.loc[data[actual_time_used] <= data[max_time_alloted]]):
    #     data['work_rate'] = 'normal'
    # else:
    #     data['work_rate'] = 'slow'
        
    # return data

    
#%%

df_feature_trans = define_work_rate(data=all_merge_df)


lambda x: 0 if x == 'slow' else(1 if x == 'nornal' else 'fast')


#%%

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

#%%

one = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

#%%
X_cat_encode = one.fit_transform(df_feature_trans[['work_rate']])

#%%
X_numeric_scaled = scaler.fit_transform(df_feature_trans[args.numeric_features])



#%%

preprocess_pipeline =  make_column_transformer((scaler, args.numeric_features),
                        (one, args.categorical_features))


logit_balanced_pipeline = make_pipeline(preprocess_pipeline,
                               LogisticRegression(class_weight='balanced')
                               )




#%%

X = df_feature_trans[args.predictors]


#%%

y = df_feature_trans[args.target_variable]


#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022)


#%%

logit_balanced_pipeline.fit(X_train, y_train)


#%%
y_pred_feature_created = logit_balanced_pipeline.predict(X_test)

#%%

print(classification_report(y_test, y_pred_feature_created))


#%%
print(f"Balanced with features created accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred_feature_created)}")


#%%
print(f"Balanced with features created R2: {r2_score(y_true=y_test, y_pred=y_pred_feature_created)}")

#%%
print(f"Balanced with features created precision score: {precision_score(y_test, y_pred_feature_created)}")
print(f"Balance with features created recall score: {recall_score(y_test, y_pred_feature_created)}")
print(f"Balance with features created features: {logit_balanced_pipeline.feature_names_in_}")


#%%
print(f"Balanced with features created coef: {logit_balanced_pipeline}")

# %%
roc_auc_score(y_test, y_pred_feature_created)


#%%

logit_balanced_pipeline.named_steps['logisticregression'].coef_

#%%
logit_balanced_pipeline.named_steps['logisticregression'].feature_names_in_


# %%
logit_balanced_pipeline[:-1].get_feature_names_out()


#%%


feature_names = logit_balanced_pipeline[:-1].get_feature_names_out()

coefs = pd.DataFrame(
    logit_balanced_pipeline.named_steps['logisticregression'].coef_.reshape(6,1),
    columns=["Coefficients"],
    index=feature_names,
)

coefs


#%%
coefs.plot.barh(figsize=(9, 7))


#%%
num_df = df_feature_trans[args.numeric_features]

#%%
num_df.corr()


# average_answer_time_in_min and extra_time_min are strongly positively correlated which 
# means that including both variables provide very similar information orsignals 
# to the model hence redundant when both are included in the model for the model.
# Hence, one of the them is selected and the other dropped to reduce overfitting. 
# In deciding which of them to include, ease of interoretatbility and relevance for decision
# making is considered. Generally, the relevance of average_answer_time_in_min will depend
# on the min and max time allocated for the task. Extra time is an indicator that can easily
# be communicated to students in a manner that is easily to be optimized. For example,
# if students are told that the probability of an assignment being passed reduces by 10% 
# for every hour extra time used then, they can easily minimize using extra time and focus 
# on working within time. For this reason, extra_time will be used for the analysis
#%%
from scipy.stats import chi2_contingency

#%%
workrate_state_crosstab_df = pd.crosstab(df_feature_trans['work_rate'], df_feature_trans['state_category'], margins = False)

workrate_state_crosstab_df.columns = ['not-passed', 'passed']
workrate_state_crosstab_df.index = ['fast', 'normal', 'slow']


#%%

chi2_contingency(observed=workrate_state_crosstab_df)

# With p-value less than 0.05, the null hypothesis that there is not reationship b/t state (pass/not_pass)
# and work_rate is rejected. This variable is expected to influence state hence selected to be in
# included in the model

#%%

workrate_state_crosstab_df.plot(kind="bar")



# %% should variable test_variable be included

test_state_crosstab_df = pd.crosstab(df_feature_trans['test_variable_transform'], df_feature_trans['state_category'], margins = False)



#%%
test_state_crosstab_df.columns = ['not-passed', 'passed']
test_state_crosstab_df.index = ['0', '1']


#%%
chi2_contingency(observed=test_state_crosstab_df)

# With a p-value greater than 0.05, the null hypothesis is not rejected which suggested
# there is no relation b/t state and test_variable

#%%  ##### check distrbution of progress_percent

sns.boxplot(df_feature_trans['progress_percent'])

# it is normally distributed with no outliers

#%%
df_feature_trans['progress_percent'].plot(kind='hist')




# %%

############# REFIT MODEL WITHOUT average_answer_time_in_min, and test_variable_transform


#%%

X_selected = df_feature_trans[args.selected_predictors]


#%%

y = df_feature_trans[args.target_variable]


#%%
X_sel_train, X_sel_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=2022)




#%%

feat_selected_preprocess_pipeline =  make_column_transformer((scaler, args.selected_numeric_features),
                                                            (one, args.categorical_features)
                                                            )


logit_sel_balanced_pipeline = make_pipeline(feat_selected_preprocess_pipeline,
                               LogisticRegression(class_weight='balanced')
                               )

#%%

logit_sel_balanced_pipeline.fit(X_sel_train, y_train)


#%%
y_pred_sel_feature_created = logit_sel_balanced_pipeline.predict(X_test)



#%%

print(classification_report(y_test, y_pred_sel_feature_created))



#%%
print(f"Balanced with selected features created accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred_sel_feature_created)}")


#%%
print(f"Balanced with selected features created R2: {r2_score(y_true=y_test, y_pred=y_pred_sel_feature_created)}")

#%%
print(f"Balanced with selected features created precision score: {precision_score(y_test, y_pred_sel_feature_created)}")
print(f"Balance with selected features created recall score: {recall_score(y_test, y_pred_sel_feature_created)}")
print(f"Balance with selected features created features: {logit_sel_balanced_pipeline.feature_names_in_}")


#%%
print(f"Balanced with selected features created coef: {logit_sel_balanced_pipeline}")



#%%
logit_sel_balanced_pipeline.named_steps['logisticregression'].coef_



#%%

selected_model_features = logit_sel_balanced_pipeline[:-1].get_feature_names_out()



#%%

model_coefs = pd.DataFrame(
    logit_sel_balanced_pipeline.named_steps['logisticregression'].coef_.reshape(5,1),
    columns=["Coefficients"],
    index=selected_model_features,
)

model_coefs


#%%
model_coefs.plot.barh(figsize=(9, 7))





############################################################################################################################



# %%  ######### QUESTION 1:   We are interested in evaluating if either of these
##  two models is performing better in terms of time between submissions. How would you evaluate this?


#pd.crosstab(df_feature_trans['test_variable_transform'], df_feature_trans['average_answer_time_in_hours'], margins = False)

df_feature_trans.groupby('test_variable')['average_answer_time_in_hours'].agg('mean')


#%%
#df_feature_trans.groupby('test_variable')['average_answer_time_in_hours'].plot(kind='box')

test_a = df_feature_trans[df_feature_trans['test_variable']=='a']

test_a_avg_ans_time = test_a.copy()['average_answer_time_in_hours']

#%%
test_b = df_feature_trans[df_feature_trans['test_variable']=='b']

test_b_avg_ans_time = test_b.copy()['average_answer_time_in_hours']
#%%
test_a['average_answer_time_in_hours'].plot(kind='hist')



#%%
test_a_variance = np.var(test_a['average_answer_time_in_hours'])

#%%
test_b_variance = np.var(test_b['average_answer_time_in_hours'])

#%%

test_a_variance / test_b_variance


#%%
import statistics
import scipy.stats as stats
import pingouin as pg



# %%
statistics.variance(test_a['average_answer_time_in_hours'])


#%%
statistics.variance(df_feature_trans[df_feature_trans['test_variable']=='b']['average_answer_time_in_hours'])

#%%  ############## .  testing normality of distribution

stats.shapiro(test_a['average_answer_time_in_hours'])

## p-value is 0.0 hence the null hypothesis that the sample is not different from a normal distrobution is 
# rejected. Therefore the data is not normaliy distributed
#%%
stats.shapiro(test_b['average_answer_time_in_hours'])

## p-value is 0.0 hence the null hypothesis that the sample is not different from a normal distrobution is 
# rejected. Therefore the data is not normaliy distributed

#%%
stats.levene(test_a_avg_ans_time, test_b_avg_ans_time)

# result indicates a p-value of 0.3188 hence fail to reject the hypothesis that variance is constant 
# hence there is homogeneity of variance between groups




# %%

stats.ttest_ind(a=test_a['average_answer_time_in_hours'], b=test_b['average_answer_time_in_hours'])


#%%
pg.ttest(test_a['average_answer_time_in_hours'], test_b['average_answer_time_in_hours'], correction=True)

## The result shows a p-value of 0.27 hence fail to reject the null hypothesis that there is no 
# statistical difference in mean answer time between test group A and B









# %%

df_feature_trans['student_id'].nunique()

#%%
df_feature_trans.describe()

df_feature_trans.info()

#%%

df_feature_trans.groupby('submission_id')['student_id'].count()

#%%

df_feature_trans.groupby('student_id')['submission_id'].count()


## groupby student, submission
# take approved date and sort in decending order
# find the difference in time between each submission




#%%

df_feature_trans['approval_date'].diff()

#%%

grp = df_feature_trans.groupby(['student_id'])#['approval_date']#.reset_index() #.sort('approval_date') #['submission']


#%%

df_feature_trans.groupby(['student_id'])



#%%

grp_df = grp.apply(lambda x: x)

#%%

#grp['approval_date'].sort_values()

#df_feature_trans.groupby('student_id')['approval_date'].diff()

############################################

df_feature_trans_narm = df_feature_trans.copy().dropna(subset='approval_date')

#%%

df_feature_trans_narm_grp = df_feature_trans_narm.groupby(by=['student_id', 'name_x']).apply(lambda x: x).sort_values(by='approval_date')




#%%

#df_feature_trans_narm_grp.sort_values(by='approval_date')['approval_date'].diff().max()#.dt.days.max()


df_feature_trans_narm_grp.sort_values(by='approval_date')['approval_date'].diff().max()

#%%

#df_feature_trans_narm_grp[df_feature_trans_narm['student_id']==39960]['approval_date'].diff().max()



#%%

try_df = df_feature_trans.copy()
try_df_narm = try_df.dropna(subset='approval_date').sort_values(by=['student_id', 'approval_date'])

try_df_narm['submission_time_interval'] = try_df_narm.groupby('student_id')['approval_date'].diff()

try_df_narm[try_df_narm['student_id']==39960]


try_df_narm['submission_time_interval_days'] = try_df_narm['submission_time_interval'].dt.days


#%%

# try_df_narm[['student_id', 'submission_id', 'approval_date', 'submission_time_interval_days', 'test_variable']]


test_a_try_df = try_df_narm[try_df_narm['test_variable']== 'a']['submission_time_interval_days']

test_b_try_df = try_df_narm[try_df_narm['test_variable']== 'b']['submission_time_interval_days']

#%%

test_a_try_df.describe()


#%%

statistics.variance(test_a_try_df.dropna())

statistics.variance(test_b_try_df.dropna())


#%%
stats.shapiro(test_a_try_df.dropna())


#%%
stats.shapiro(test_b_try_df.dropna())


#%%

stats.levene(test_a_try_df.dropna(), test_b_try_df.dropna())


#%%
pg.ttest(test_a_try_df.dropna(), test_b_try_df.dropna(), correction=False)






#%%
df_sorted_approvedate = df_feature_trans.dropna(subset='approval_date').sort_values(by='approval_date')


#%%

df_sorted_grouped =  df_sorted_approvedate.groupby(by=['student_id'])#['approval_date'].diff()#.reset_index()


df_sorted_grouped_df = df_sorted_grouped.apply(lambda x: x)


#%%

#df_sorted_grouped_df['submission_time_interval'] =


df_sorted_grouped_df['approval_date'].transform(lambda x: x.diff()).max()


#%%

df_sorted_grouped_df.info()


#%%

df_sorted_grouped_df[df_sorted_grouped_df['student_id']==39960]['approval_date'].diff().dt.days.tolist()







#%%

df_test_data_subtime_interval = df_sorted_grouped_df[['test_variable', 'submission_time_interval']]


#%%

df_test_data_subtime_interval.submission_time_interval.mean()


#%%

df_test_data_subtime_interval['submission_time_interval'].dt.total_seconds() / 60


#%%


df_test_data_subtime_interval['submission_time_interval'].max()

#%%

test_a_submission_time_interval = df_test_data_subtime_interval[df_test_data_subtime_interval['test_variable']=='a']

test_a_submission_time_interval = test_a_submission_time_interval.copy()['submission_time_interval']

test_b_submission_time_interval = df_test_data_subtime_interval[df_test_data_subtime_interval['test_variable']=='b']
test_b_submission_time_interval = test_b_submission_time_interval.copy()['submission_time_interval']



#%%

test_a_submission_time_interval.submission_time_interval.dt.days

#%%

statistics.variance(test_b_submission_time_interval)







# WSG means Weekly Submission Goals.
# %%

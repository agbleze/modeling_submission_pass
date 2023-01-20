



#%% create correlation
corr = X.corr()
corr


fig = px.box(data_frame=cv_score_test_df, x='model', y='test_RMSE', 
                 color='model',# notched=True, 
                 title=f'Test error of 10 fold cross validation on Models',
                 template='plotly_dark', height=700,
                 )
from xgboost import XGBRFClassifier, XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, BaggingClassifier, 
                              HistGradientBoostingClassifier,ExtraTreesClassifier,
                              VotingClassifier,
                              )
#from sklearn.neighbors import KNeighborsClassifier
from preprocess_pipeline import PipelineBuilder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


pipeline = PipelineBuilder()

preprocess_pipeline = pipeline.build_data_preprocess_pipeline()



#knn = KNeighborsRegressor(n_neighbors=10)

#knn = KNeighborsClassifier(n_neighbors=10)

svc_rbf = SVC(kernel='rbf', class_weight='balanced')
svc_linear = SVC(kernel='linear', class_weight='balanced')
svc_poly = SVC(kernel='poly', class_weight='balanced')
rfc = RandomForestClassifier(class_weight='balanced')

decision_tree = DecisionTreeClassifier(class_weight='balanced')
extra_decision_tree = ExtraTreeClassifier(class_weight='balanced')












#rf = RandomForestRegressor(random_state=0)
#rf_pipeline = make_pipeline(decision_tree_data_preprocess, rf)


svc_rbf_pipeline = pipeline.build_model_pipeline(model=svc_rbf)
svc_linear_pipeline = pipeline.build_model_pipeline(model=svc_linear)
svc_poly_pipeline = pipeline.build_model_pipeline(model=svc_poly)
rfc_pipeline = pipeline.build_model_pipeline(model=rfc)
decision_tree_pipeline = pipeline.build_model_pipeline(model=decision_tree)
extra_decision_tree_pipeline = pipeline.build_model_pipeline(model=extra_decision_tree)



all_models = [("Extra decision tree", extra_decision_tree_pipeline),
              ("Decision tree", decision_tree_pipeline),
              ("Radom forest classifier", rfc_pipeline),
              ("SVC poly", svc_poly_pipeline),
              ("SVC linear", svc_linear_pipeline),
              ("SVC rbf", svc_rbf_pipeline)
              ]








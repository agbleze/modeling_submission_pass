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












rf = RandomForestRegressor(random_state=0)
rf_pipeline = make_pipeline(decision_tree_data_preprocess, rf)


svc_rbf_pipeline = pipeline.build_model_pipeline(model=svc_rbf)




all_models = [("Extra decision tree", extra_decision_tree),
              ("Decision tree", decision_tree),
              ("Radom forest classifier", rfc),
              ("SVC poly", svc_poly),
              ("SVC linear", svc_linear),
              ("SVC rbf", svc_rbf)
              ]


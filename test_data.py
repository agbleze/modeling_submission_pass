import pytest
from utils.get_path import get_data_path
import pandas as pd
from data.data_read import get_data_from_numbers_file
from data.data_split import split_data
from arguments import args
from transform.data_transform import DataTransformer


datapath = get_data_path(folder_name='DS_Practical', file_name='all_merge_df.csv')

@pytest.fixture()
def dataset():
    datapath = get_data_path(folder_name='DS_Practical', 
                             file_name='all_merge_df.csv'
                             )
    df = pd.read_csv(datapath)
    return df


@pytest.fixture()
def data_transformer():
    data_transformer_obj = DataTransformer(data=dataset)
    return data_transformer_obj


@pytest.fixture()
def transformed_hours_to_minute_data(data_transformer):
    data_transformed = data_transformer.convert_hours_to_minute()
    return data_transformed
    

@pytest.fixture()
def train_test_data(dataset):
    data_split = split_data(data=dataset,
                            features=args.selected_predictors,
                            target=args.target_variable_transformed
                            )
    return data_split
     
     
     
@pytest.fixture()
def model():
    pass


##

#TODO:
    # TEST define work rate for work rate column
    # Test define work rate column for values slow, normal, fast
    
    
def test_work_rate_column_available(data_transformer,
                                    transformed_hours_to_minute_data
                                    ):
    work_rate_df = data_transformer.define_work_rate(data_to_transform=transformed_hours_to_minute_data
                                      )
    assert('work_rate' in work_rate_df.columns) 


def test_work_rate_values():
    pass



def test_get_data_from_numbers_file():
    data_path = get_data_path(folder_name='DS_Practical',
                              file_name='student_-_exercise_progress.numbers'
                              )
    df = get_data_from_numbers_file(data_filepath=data_path,
                               tab=0
                               )
    
    assert(pd.DataFrame == type(df))
    

def test_split_data():
    pass



def test_target_variable_available():
    pass




def test_target_variable_is_binary():
    pass
    
    
    
def test_predictor_variables_available():
    pass


def test_predict_pass_status():
    pass


def test_load_model():
    pass


def test_candidate_model():
    pass


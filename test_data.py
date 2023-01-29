import pytest
from utils.get_path import get_data_path
import pandas as pd
from data.data_read import get_data_from_numbers_file
from data.data_split import train_test_split


datapath = get_data_path(folder_name='DS_Practical', file_name='all_merge_df.csv')

@pytest.fixture()
def dataset():
    datapath = get_data_path(folder_name='DS_Practical', 
                             file_name='all_merge_df.csv'
                             )
    df = pd.read_csv(datapath)
    return df

@pytest.fixture()
def model():
    pass


##

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


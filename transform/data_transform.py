import pandas as pd
import numpy as np

rescale_negative_to_zero =  lambda x: 0 if x < 0 else x
convert_hours_to_minute = lambda x: x * 60
transform_test_variable = lambda x: 0 if x == 'a' else 1


class DataTransformer(object):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
        
    def convert_hours_to_minute(self, column_to_convert_minutes: str = 'average_answer_time_in_hours'):
        self.data_copied = self.data.copy()
        new_columnname = column_to_convert_minutes[:-3] + 'min'
        self.data_copied[new_columnname] = self.data_copied[column_to_convert_minutes].apply(convert_hours_to_minute)
        return self.data_copied
            
        

    # function to create work_rate variable
    @classmethod
    def define_work_rate(self, data_to_transform: pd.DataFrame = None, 
                        min_time_alloted: str = 'min_task_time_in_min', 
                        max_time_alloted: str = 'max_task_time_in_min',
                        actual_time_used: str = 'average_answer_time_in_min'
                        ):
        if data_to_transform is None:           
            self.data_to_transform = self.data_copied
        else:
            self.data_to_transform = data_to_transform
        
        self.data_to_transform['work_rate'] = (np.where((self.data_to_transform[actual_time_used] < 
                                                         self.data_to_transform[min_time_alloted]
                                                         ), 
                                                        'fast', np.nan
                                                        )
                                               )
        self.data_to_transform['work_rate'] = (np.where((self.data_to_transform[actual_time_used] >= 
                                                         self.data_to_transform[min_time_alloted]
                                                         ) & 
                                                        (self.data_to_transform[actual_time_used] <= 
                                                         self.data_to_transform[max_time_alloted]
                                                         ), 
                                                        'normal', self.data_to_transform['work_rate']
                                                        )
                                               )
        self.data_to_transform['work_rate'] = (np.where((self.data_to_transform[actual_time_used] > 
                                                         self.data_to_transform[max_time_alloted]
                                                         ), 
                                                        'slow', self.data_to_transform['work_rate']
                                                        )
                                               )
        return self.data_to_transform


# compute extra time
    @classmethod
    def calculate_extra_time_min(self, data_to_transform: pd.DataFrame = None, max_time_min="max_task_time_in_min", 
                                actual_time_min="average_answer_time_in_min",
                                make_negative_time_zero_minutes: callable = rescale_negative_to_zero
                                ):
        if data_to_transform is None:
            self.data_to_transform = self.data_copied
        else:
            self.data_to_transform = data_to_transform
            
        self.data_to_transform['time_diff'] = self.data_to_transform[actual_time_min] - self.data_to_transform[max_time_min]
        
        self.data_to_transform['extra_time_min'] = self.data_to_transform['time_diff'].apply(make_negative_time_zero_minutes)
        
        return self.data_to_transform
    
    
    @classmethod
    def convert_categorical_to_numeric(self, data: pd.DataFrame = None,
                                       column_to_convert: str = 'test_variable', 
                                       string_value_pair: dict = {'a': 0, 'b': 1}
                                       ):
        if data is None:
            self.data_to_transform = self.data_copied
        self.data_to_transform = data
        new_columnname = column_to_convert + '_transform'
        self.data_to_transform[new_columnname] = self.data_to_transform[column_to_convert].apply(transform_test_variable)
        return self.data_to_transform
        
        









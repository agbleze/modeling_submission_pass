from argparse import Namespace


args = Namespace(
    target_variable = 'state',
    target_variable_transformed = 'state_category',
    predictor_variables = ['average_answer_time_in_hours', 'progress_percent', 'test_variable_transform'],
    numeric_features = ['progress_percent', 'average_answer_time_in_min', 'extra_time_min',
                        ],
    categorical_features = ['work_rate'],
    binary_feature = ['test_variable_transform'],
    predictors = ['progress_percent', 'average_answer_time_in_min', 'extra_time_min',
                  'work_rate', 'test_variable_transform'],
    selected_predictors = ['progress_percent', 'extra_time_min', 'work_rate',
                        ],
    selected_numeric_features = ['progress_percent', 'extra_time_min'],
    not_passed = ['waiting_for_review', 'almost_there', 
                  'not_yet', 'a_little_more'],


)
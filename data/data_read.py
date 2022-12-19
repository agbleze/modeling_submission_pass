import pandas as pd
from numbers_parser import Document
import os
import scipy.stats as stats



def get_data_from_numbers_file(data_filepath: str, tab: int):
    doc = Document(data_filepath)
    sheets = doc.sheets
    tables = sheets[tab].tables
    data = tables[0].rows(values_only=True)
    df = pd.DataFrame(data[1:], columns=data[0])
    return df
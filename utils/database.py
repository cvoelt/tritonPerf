from json.tool import main
import sqlite3
import pathlib
import logging
import pandavro
import typing

from numpy import insert
from . import config
from . import utilities
import pandas

def csv_to_dataframe(csv_path: pathlib.Path, main_config: config.MainConfig,  analyzer_config: config.MeasurementConfig) -> pandas.DataFrame:

    data_frame = pandas.read_csv(csv_path.as_posix())
    data_frame.columns = data_frame.columns.str.replace(" ","")
    data_frame.columns = data_frame.columns.str.replace("/","_")
    data_frame.columns = data_frame.columns.str.replace("+","and")
    data_frame = data_frame.drop(labels='Concurrency', axis=1)
    data_frame = add_main_settings_to_dataframe(data_frame, main_config)
    data_frame = add_measurement_settings_to_dataframe(data_frame, analyzer_config)

    return data_frame

def add_measurement_settings_to_dataframe(data_frame: pandas.DataFrame, measure_config: config.MeasurementConfig) -> pandas.DataFrame:
    
    measurement_attributes = measure_config.get_values()
    insert_at = 0
    for attribute, value in measurement_attributes:
        data_frame.drop(attribute, inplace= True, axis=1, errors='ignore')
        data_frame.insert(insert_at, attribute, value, True)
        insert_at += 1

    return data_frame

def add_main_settings_to_dataframe(data_frame: pandas.DataFrame, main_config: config.MainConfig) -> pandas.DataFrame:
    
    attributes = ['trt_workspace_size', 'triton_ip', 'triton_http_port', 'warmup']
    insert_at = 0
    for insert_at, attribute in enumerate(attributes):
        value = getattr(main_config, attribute)
        data_frame.drop(attribute, inplace= True, axis=1, errors='ignore')
        data_frame.insert(insert_at, attribute, value, True)
        insert_at += 1
    return data_frame

def add_empty_row_to_dataframe(data_frame: pandas.DataFrame, execution_count: int, measure_config: config.MeasurementConfig, main_config: config.MainConfig) -> pandas.DataFrame:
    #cant handle empty dataframes
    empty_row = data_frame.tail(1).copy(deep=True)#copy the last row from the original dataframe
    for col in empty_row.columns:
        empty_row[col].values[:] = -1   #set all values to -1
    add_main_settings_to_dataframe(empty_row,main_config)    
    add_measurement_settings_to_dataframe(empty_row, measure_config)
    empty_row.iloc[-1, empty_row.columns.get_loc('execution_count')] = execution_count
    return concat_dataframes(data_frame, empty_row)

def concat_dataframes(frame1: pandas.DataFrame, frame2: pandas.DataFrame):
    if frame1 is not None:
        #frames = [frame1, frame2]
        #return pandas.concat(frames)
        return frame1.append(frame2)
    else:
        return frame2

def save_dataframe_to_db(df: pandas.DataFrame, table_name: str, connection: sqlite3.Connection) -> None:
    df = df.round(decimals=2)
    df.to_sql(table_name ,connection, if_exists='replace')

def save_dataframe_to_avro(df: pandas.DataFrame, main_config: config.MainConfig) -> None:
    output_dir = main_config.get_results_dir().as_posix() + '/result_frame.avro'
    pandavro.to_avro(output_dir, df)


def save_dataframe_to_csv(df:pandas.DataFrame, main_config: config.MainConfig) -> None:
    df.to_csv(main_config.get_results_dir().as_posix() + '/result_frame.csv')

def save_results(df:pandas.DataFrame, main_config: config.MainConfig) -> None:
    connection = create_connection(utilities.results_folder_db_filename(main_config)) 
    df[['triton_ip']] = df[['triton_ip']].astype(str)
    save_dataframe_to_db(df, 'results', connection)
    save_dataframe_to_avro(df, main_config)
    save_dataframe_to_csv(df, main_config)


def create_connection(db_path:pathlib.Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(db_path.as_posix())
    except sqlite3.Error as e:
        logging.error('Error opening database: ' + str(e))
    return connection

def less_inferences(result_frame: pandas.DataFrame, current_measurement: pandas.DataFrame):
    second_last_elem = result_frame['Inferences_Second'].iloc[-2].copy()
    last_element = current_measurement['Inferences_Second'].iloc[-1].copy()
    stop_measuring = last_element < second_last_elem
    return stop_measuring

#---currently not needed, maybe interesting?------
'''
def labels_to_table(table_name: str, labels:list, connection:sqlite3.Connection) -> bool:
    cursor = connection.cursor()
    create_table_command = 'CREATE TABLE IF NOT EXISTS ' + table_name + ' ('
    for label in labels:#add all labels to create table
        create_table_command += label + ' FLOAT, '
    create_table_command = create_table_command[:len(create_table_command)-2] #remove last comma and whitespace
    create_table_command += ')'
    cursor.execute(create_table_command)
    connection.commit()
    return True

def csv_to_labels(csv_path:pathlib.Path) -> list:
    with open(csv_path.as_posix(),'r',newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        return next(reader)


def dataframe_columns(df:pandas.DataFrame) -> list:
    return df.columns.values.tolist()

'''
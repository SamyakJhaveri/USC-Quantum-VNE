"""
BQM Approach
(Did not need to use this separate script as I integrated all the functionality of this into the 
conductor script to make things simpler and smoother)

Semi-Final Script. Works perfectly well but not used in the Final version of the Experiment.
Final Script is in 'Working Code to Use' Folder with the other scripts and the conductor script. 


Script to Log data about every experiment

References:
- https://stackoverflow.com/questions/35992556/storing-and-analysing-experimental-data-in-efficient-way-sql-python
- https://codeburst.io/copy-pastable-logging-scheme-for-python-c17efcf9e6dc
- https://www.geeksforgeeks.org/exporting-pandas-dataframe-to-json-file/
- https://www.projectpro.io/recipes/save-pandas-dataframe-as-csv-file

"""
# Importing Stuff
import time
import os
import datetime
import pandas as pd
"""
columns = [
        'experiment_label', 
        'key', 
        'number of nodes in G1', 
        'number of edges in G1', 
        'number of nodes in G2', 
        'number of edges in G2',
        'isomorphic_pair?', 
        'dwave_leap_problem_id', 
        'mapping from G2 to G1', 
        'runtime', 
        'charge time', 
        'qpu access time', 
        'number of annealing cycles',
        'duration per anneal cycle'
        ]
"""
def store_experiment_log_data():
    df_for_csv = pd.DataFrame(result_data_for_csv, columns = columns)
    df_for_csv.to_csv(experiment_label + '.csv', index = True)


def retrieve_experiment_log_data():
    df_from_csv = pd.read_csv(experiment_label + '.csv')
    
    print(df_for_csv.head())

# CSV
result_data_for_csv = {
    'experiment_label': [experiment_label], 
    'key': [key], 
    'number of nodes in G1': [n1],
    'number of edges in G1': [e1], 
    'number of nodes in G2': [n2], 
    'number of edges in G2': [e2],
    'isomorphic_pair?': [isomorphic_pair], 
    'dwave_leap_problem_id': [dwave_leap_problem_id],
    'mapping from G2 to G1': [mapping], 
    'runtime': [runtime],
    'charge time': [charge_time],
    'qpu access time': [qpu_access_time], 
    'number of annealing cycles': [no_annealing_cycles],
    'duration per anneal cycle': [duration_per_anneal_cycle]
}  

"""
# JSON
result_data_for_json = [[
    experiment_label, key, n1, e1, n2, e2, isomorphic_pair, 
    dwave_leap_problem_id, mapping, runtime, charge_time, 
    qpu_access_time, no_annealing_cycles, duration_per_anneal_cycle]]

df_for_json = pd.DataFrame(result_data_for_json, columns = columns)
df_for_json.to_json(experiment_label + '.json', orient = 'split', compression = 'infer', index = 'true')

df_from_json = pd.read_json(experiment_label + '.json', orient ='split', compression = 'infer')
# print(df_from_json)
"""

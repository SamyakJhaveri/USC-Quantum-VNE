"""
DQM Approach
Final Working Script to be used for Experiments. 

Script that Conducts the experiments and coordinated the activities of all the scripts 

References:
- https://towardsdatascience.com/simple-trick-to-work-with-relative-paths-in-python-c072cdc9acb9
- Using Pickle - https://ianlondon.github.io/blog/pickling-basics/
- Appending a new row of information to an existing csv file -https://www.youtube.com/watch?v=sHf0CJU8y7U
- https://blog.finxter.com/how-to-append-a-new-row-to-a-csv-file-in-python/
- Using Pickle - https://ianlondon.github.io/blog/pickling-basics/
- Appending a new row of information to an existing csv file -https://www.youtube.com/watch?v=sHf0CJU8y7U
- https://blog.finxter.com/how-to-append-a-new-row-to-a-csv-file-in-python/
"""

# Importing the right libraries and frameworks
import networkx as nx
import random 
import pickle
import sys
import itertools
import numpy as np  
import dimod
from dwave.system import LeapHybridDQMSampler 
import time 
import argparse
import os
import pandas as pd


# Ignore errors importing matpotlib.pyplot
try:
    import matplotlib.pyplot as plt  
    import matplotlib.colors as mcolors
except ImportError:
    pass

# Importing the scripts and their functions

# importing `graph_subgraph_pair_generator_script.py` script
import graph_subgraph_pair_generator_script_for_experiment


# importing `sgi_qvne_H1.py`
# H1 is the baseline Hamiltonian, 
# H2 is the compact Hamiltonian (tweaked and fine tuned for the the Virtual Network Embedding problem)
import sgi_qvne_H1_for_experiment

# importing `testing_and_evaluation.py` script
import testing_and_evaluation_for_experiment


if __name__ == "__main__":
    
    dataset_filepath = "/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/DQM Approach/Dataset(DQM)/G1_G2_PAIR_DICTIONARY_DATASET.pickle"
    
    # ---------- x ---------- x ---------- x ---------- 

    """ For Generating a Dataset of Graph - SubGraph Pair in Dictionary Form """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n1")
    args = parser.parse_args()
    # `n1` = Maximum number of Nodes in Graph G1 you want in the Dataset

    # The Dataset wil be in dictionary form which will have the folowing format:
    # G1_G2_pair_dict: {key: [G1, G2]}
    # where `key` is the key name given to the Graph - SubGraph Pair
    # `[G1, G2]` is the Graph - SubGraph Pair (G1 is Graph, G2 is SubGraph)
    print("The Default Probability of an Edge being present betweeeen two nodes (p) is set to 0.5 i.e. 50%")
    p = 0.5 # chosen as per Zick's paper
    n1 = int(args.n1)
    n2 = n1 - 1 # if n1 = 20, then n2 = 20 - 1 = 19, n2 is number of nodes possible in SubGraph
    G1_G2_pair_dict = {} # dictionary to keep all the graph-subgraph pairs in one place. 
    # Will store only this one dictinoary instead of storing every graph-subgraph pair in 
    # a separate file that has to be stored and retreived everytime a new pair is 
    # generated which makes the whole process computationally heavy

    key = ""
    for i in range(5, n1 + 1):
                # This loop iterates over the number of nodes G can have.
                # range() is written this way as it only meaningful to have a G with at least 5 nodes
                # and the upper limit of the range is chosen as `n1+1` because range() with upper limit 
                # as `n1` does not include the last number (i.e. `n1` number of nodes)
                G1 = graph_subgraph_pair_generator_script_for_experiment.generate_random_parent_graph(i, p)
                for j in range(3, n2 + 1):
                        # This loop iterates over the number of nodes G2 can have. 
                        # range() is written this way because it is only meaningful to start having subgraphs 
                        # that at least have 3 nodes and make a triangle.
                        # the reason behind chosing `n2 + 1` as the upper limit of the range() is the same as 
                        # that for the above loop for G1. 
                        G2 = graph_subgraph_pair_generator_script_for_experiment.generate_child_subgraph(G1, j)
                        key = "G1(" + str(i) + ")_G2(" + str(j) + ")_iso" # G1(20)_G2(19)
                        G1_G2_pair_dict[key] = [G1, G2]
    
    # Saving the Graph-SubGraph Pair Dataset in the designated directory
    graph_subgraph_pair_generator_script_for_experiment.save_G1_G2_pair_dict(dataset_filepath, G1_G2_pair_dict)
    print("SAVED!")
    """

    # ---------- x ---------- x ---------- x ----------   
    
    """ For Picking a Single Pair of Graph-SubGraph from the Dataset and 
    getting a mapping from G2 to G1 """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n1")
    parser.add_argument("--n2")
    args = parser.parse_args()
    # `n1` = Pick Graph G1 from the dataset that has `n1` Number of Nodes
    # `n2` = Pick Graph G2 from the dataset that has `n2` Number of Nodes

    # Retreiving the G-SG Pair Dataset from the Directory 
    
    G1_G2_pair_dict_loaded = graph_subgraph_pair_generator_script_for_experiment.load_G1_G2_pair_dictionary(dataset_filepath)
    print("The Dataset has {} Graph-SubGraph Pairs.".format(len(G1_G2_pair_dict_loaded)))
    
    # Generating a key using the command line arguments passed for n1 and n2
    # Key Format: G1(<n1>)_G2(<n2>)_iso
    key = "G1(" + args.n1 + ")_G2(" + args.n2 + ")_iso"  # "G1(20)_G2(19)_iso"
    G1, G2 = G1_G2_pair_dict_loaded[key]
    # G1_G2_pair_list = G1_G2_pair_dict_loaded[key]
    # print("Graph - SubGraph Pair:{}".format(G1_G2_pair_list))

    # Plot the graph-subgraph pair corresponding to the input key 
    # as retreived from the previously generated dataset
    graph_subgraph_pair_generator_script_for_experiment.plot_graphs(G1, G2)

    # THE SUBGRAPH ISOMORPHISM Problem Implmented for the Quantum Virtual Network Embedding Solved on the DWave
    sampleset = sgi_qvne_H1_for_experiment.find_isomorphism(G1, G2)
    # G1, G2, sampleset = sgi_qvne_H1_for_experiment.find_isomorphism(G1, G2)
    sgi_qvne_H1_for_experiment.present_results(G1, G2, sampleset)

    # Saving the Results 
    logging_result_data_folder_path = '/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/DQM Approach/Experiment Result Data(DQM)/'
    logging_result_data_file_path = logging_result_data_folder_path + 'qvne_H1_dqm_experiment_result_data.csv'
    
    n1 = args.n1 # 'number of nodes in G1'
    e1 = G1.number_of_edges() # 'number of edges in G1'
    n2 = args.n2 # 'number of nodes in G2'
    e2 = G2.number_of_edges() # 'number of edges in G2'
    isomorphic_pair = True # Whether the G1 G2 pair is isomorphic or not
    experiment_label =  'qvne_H1_dqm_experiment_result_data_' + key
    best_mapping = sampleset.first.sample # the resultant mapping obtained form the annealing process (Could store 5 of these for averging out the result by running the script 5 times)
    best_mapping_energy = sampleset.first.energy 
    runtime = sampleset.info["run_time"] # time, in microseconds, of the run time (see description of runtime in Notes and 'testing_and_evaluation_script')
    charge_time = sampleset.info["charge_time"] # time, in microseconds, of the charge time (see description of chargetime in Notes and 'testing_and_evaluation_script')
    qpu_access_time = sampleset.info["qpu_access_time"] # time, in microseconds, of the qpu access time (see description of qpu access time in Notes and 'testing_and_evaluation_script')
    dwave_leap_problem_id = sampleset.info["problem_id"] # get DWave problem ID from 'testing_and_evaluation_Script'
    no_annealing_cycles = 100 # placeholder valuefor 'number of annealing cycles'
    duration_per_anneal_cycle = 20 # placeholder value, in microseconds, for the 'duration of each anneal cycle'
    # solver_information = Commented out because there a re a lot of data points, they are present in the 'testing_and_evaluation_script' 
    
    columns = [
        'key',
        'experiment label', 
        'number of nodes in G1', 
        'number of edges in G1', 
        'number of nodes in G2', 
        'number of edges in G2',
        'isomorphic pair?',  
        'mapping from G2 to G1',
        'best mapping energy', 
        'runtime', 
        'charge time', 
        'qpu access time',
        'dwave leap problem id', 
        'number of annealing cycles',
        'duration per anneal cycle'
        ]
    
    result_data_for_csv = [{
        'key': key,
        'experiment label': experiment_label,  
        'number of nodes in G1': n1,
        'number of edges in G1': e1, 
        'number of nodes in G2': n2, 
        'number of edges in G2': e2,
        'isomorphic pair?': isomorphic_pair, 
        'mapping from G2 to G1': best_mapping, 
        'best mapping energy': best_mapping_energy,
        'runtime': runtime,
        'charge time': charge_time,
        'qpu access time': qpu_access_time, 
        'dwave leap problem id': dwave_leap_problem_id,
        'number of annealing cycles': no_annealing_cycles,
        'duration per anneal cycle': duration_per_anneal_cycle
        }]

    if not os.path.exists(logging_result_data_folder_path):
        os.makedirs(logging_result_data_folder_path)
        df_for_csv = pd.DataFrame(result_data_for_csv, columns = columns)
        df_for_csv.to_csv(logging_result_data_file_path, header = False, index = False)
        print("CSV Created!")
    
    
    # Create another row and append row (as df) to existing CSV
    # row = [{'A':'X1', 'B':'X2', 'C':'X3'}]
    df_new_row = pd.DataFrame(result_data_for_csv)
    df_new_row.to_csv(logging_result_data_file_path, mode='a', header=False, index=False)
    print("Row added to DataFrame!")
    
    # print out what you just stored into the `Experiment Result Dataset`
    df_from_csv = pd.read_csv(logging_result_data_file_path)
    print(df_from_csv)
    

    # ---------- x ---------- x ---------- x ---------- 

    """ For Running the entire pipeline in a loop over all the graph-subgraph pairs to generate 
    a series of results for the `Experiment Results Dataset 
    # Taking the upper limit of number of nodes in G1 as 50, lower limit as 5
    # Taking the upper limit of number of nodes in G2 as 49, lower limit as 3
    # Taking step of 5
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n1")
    args = parser.parse_args()
    # `n1` = Pick G1 graphs from the dataset that have uptill `n1` Number of Nodes. i.e. All the graphs in 
    # the Graph-SubGraph Paiir Dataset having 5 to n1 nodes. 

    # Retreiving the G-SG Pair Dataset from the Directory 
    G1_G2_pair_dict_loaded = graph_subgraph_pair_generator_script_for_experiment.load_G1_G2_pair_dictionary(dataset_filepath)
    print("The Dataset has {} Graph-SubGraph Pairs.".format(len(G1_G2_pair_dict_loaded)))

    columns = [
        'key',
        'experiment label', 
        'number of nodes in G1', 
        'number of edges in G1', 
        'number of nodes in G2', 
        'number of edges in G2',
        'isomorphic pair?',  
        'mapping from G2 to G1',
        'best mapping energy', 
        'runtime', 
        'charge time', 
        'qpu access time',
        'dwave leap problem id', 
        'number of annealing cycles',
        'duration per anneal cycle'
        ]

    # Saving the Results 
    logging_result_data_folder_path = '/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/DQM Approach/Experiment Result Data(DQM)/'
    logging_result_data_file_path = logging_result_data_folder_path + 'qvne_H1_dqm_experiment_result_data.csv'

    # Step is 5 nodes, hardocded fro simplicity. 
    n1 = int(args.n1)
    for i in range(5, n1, 5):
        n1 = i
        for j in range(3, n1 - 1, 5):
            n2 = j
            if n2 < n1:
                key = "G1(" + str(n1) + ")_G2(" + str(n2) + ")_iso"  # "G1(20)_G2(19)_iso"
                G1, G2 = G1_G2_pair_dict_loaded[key]
                sampleset = sgi_qvne_H1_for_experiment.find_isomorphism(G1, G2)
                
                # sgi_qvne_H1_for_experiment.present_results(G1, G2, sampleset)
               
                e1 = G1.number_of_edges() # 'number of edges in G1'
                e2 = G2.number_of_edges() # 'number of edges in G2'
                isomorphic_pair = True # Whether the G1 G2 pair is isomorphic or not
                experiment_label =  'qvne_H1_dqm_experiment_result_data_' + key
                best_mapping = sampleset.first.sample # the resultant mapping obtained form the annealing process (Could store 5 of these for averging out the result by running the script 5 times)
                best_mapping_energy = sampleset.first.energy 
                runtime = sampleset.info["run_time"] # time, in microseconds, of the run time (see description of runtime in Notes and 'testing_and_evaluation_script')
                charge_time = sampleset.info["charge_time"] # time, in microseconds, of the charge time (see description of chargetime in Notes and 'testing_and_evaluation_script')
                qpu_access_time = sampleset.info["qpu_access_time"] # time, in microseconds, of the qpu access time (see description of qpu access time in Notes and 'testing_and_evaluation_script')
                dwave_leap_problem_id = sampleset.info["problem_id"] # get DWave problem ID from 'testing_and_evaluation_Script'
                no_annealing_cycles = 100 # placeholder valuefor 'number of annealing cycles'
                duration_per_anneal_cycle = 20 # placeholder value, in microseconds, for the 'duration of each anneal cycle'
                
                result_data_for_csv = [{
                    'key': key,
                    'experiment label': experiment_label,  
                    'number of nodes in G1': n1,
                    'number of edges in G1': e1, 
                    'number of nodes in G2': n2, 
                    'number of edges in G2': e2,
                    'isomorphic pair?': isomorphic_pair, 
                    'mapping from G2 to G1': best_mapping, 
                    'best mapping energy': best_mapping_energy,
                    'runtime': runtime,
                    'charge time': charge_time,
                    'qpu access time': qpu_access_time, 
                    'dwave leap problem id': dwave_leap_problem_id,
                    'number of annealing cycles': no_annealing_cycles,
                    'duration per anneal cycle': duration_per_anneal_cycle
                    }]

                if not os.path.exists(logging_result_data_folder_path):
                    os.makedirs(logging_result_data_folder_path)
                    df_for_csv = pd.DataFrame(result_data_for_csv, columns = columns)
                    df_for_csv.to_csv(logging_result_data_file_path, header = False, index = False)
                    print("CSV Created!")

                # Create another row and append row (as df) to existing CSV
                # row = [{'A':'X1', 'B':'X2', 'C':'X3'}]
                df_new_row = pd.DataFrame(result_data_for_csv)
                df_new_row.to_csv(logging_result_data_file_path, mode='a', header=False, index=False)
                print("Row added to DataFrame!")

                # print out what you just stored into the `Experiment Result Dataset`
                df_from_csv = pd.read_csv(logging_result_data_file_path)
                print(df_from_csv)
                
# ---------- x ---------- x ---------- x ----------
"""
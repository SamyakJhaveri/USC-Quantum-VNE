"""
BQM Approach
With improvements using suggestions from ChatGPT 
Final Working Script to be used for Experiments.

Script that Conducts the experiments and coordinated the activities of all the scripts 

References:
- https://towardsdatascience.com/simple-trick-to-work-with-relative-paths-in-python-c072cdc9acb9
- Using Pickle - https://ianlondon.github.io/blog/pickling-basics/
- Appending a new row of information to an existing csv file -https://www.youtube.com/watch?v=sHf0CJU8y7U
- https://blog.finxter.com/how-to-append-a-new-row-to-a-csv-file-in-python/
- getting number of reads in sampleset - https://docs.dwavesys.com/docs/latest/c_solver_parameters.html#num-reads
"""

# Importing the right libraries and frameworks
import networkx as nx
import random 
import pickle
import sys
import itertools
import time 
import numpy as np  
import dimod
import time 
import argparse
import os
import pandas as pd
import src.gsgmorph.pyqubo_form as gsgm_pqf
import src.gsgmorph.matrix_form as gsgm_mf 


# Ignore errors importing matpotlib.pyplot
try:
    import matplotlib.pyplot as plt  
    import matplotlib.colors as mcolors
except ImportError:
    pass

# Importing the scripts and their functions

# importing `graph_subgraph_pair_generator_script_for_experiment.py` script
import graph_subgraph_pair_generator_script_for_experiment

# importing `testing_and_evaluation_script_for_experiment.py` script
import testing_and_evaluation_script_for_experiment

# Matrix Form 
# importing `sgi_bqm_matrix_form_script_for_experiment.py`
import sgi_bqm_matrix_form_script_for_experiment

# PyQUBO Form 
# importing `sgi_bqm_pyqubo_form_script_for_experiment.py`
import sgi_bqm_pyqubo_form_script_for_experiment


if __name__ == "__main__":

    dataset_filepath = "/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/BQM Approach/Dataset(BQM)/G1_G2_PAIR_DICTIONARY_DATASET.pickle"
    
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

    # Generating a key using the command line arguments passed for `n1` and `n2`
    # Key Format: G1(<n1>)_G2(<n2>)_iso
    key = "G1(" + args.n1 + ")_G2(" + args.n2 + ")_iso"  # "G1(20)_G2(19)_iso"
    G1_G2_pair_list = G1_G2_pair_dict_loaded[key]

    # Plot the graph-subgraph pair corresponding to the input key 
    # as retreived from the previously generated dataset
    graph_subgraph_pair_generator_script_for_experiment.plot_graphs(G1_G2_pair_list)
    
    # THE SUBGRAPH ISOMORPHISM Solution Implmented for the Quantum Virtual Network Embedding Solved on the DWave    
    # ----------

    # For BQM GSGMorph Matrix Form
    
    sampleset, sample_translation_dictionary, best_sample = sgi_bqm_matrix_form_script_for_experiment.find_isomorphism(G1_G2_pair_list)
    print("Sampleset is:{}".format(sampleset))
    sampleset_agg = sampleset.aggregate()
    print("Sub-GI Results are \n :{}".format(best_sample))    
    # We can use an annealing sample and the sample translation dictionary from before to
    # generate a dictionary that maps nodes from the graph to be embedded to the target graph
    result_mapping = gsgm_mf.translate_sample(best_sample, sample_translation_dictionary)
    sgi_bqm_matrix_form_script_for_experiment.plot_graphs(G1_G2_pair_list, result_mapping)
    
    # ----------

    # For BQM GSGMorph PyQUBO Form
    
    sampleset, sample_translation_dictionary, best_sample = sgi_bqm_pyqubo_form_script_for_experiment.find_isomorphism(G1_G2_pair_list)
    # We can use the PyQUBO-translated annealing sample and the sample translation dictionary
    # from before to generate a dictionary that maps nodes from the graph to be embedded to the
    # target graph
    result_mapping = gsgm_pqf.translate_sample(best_sample, sample_translation_dictionary)
    sgi_bqm_pyqubo_form_script_for_experiment.plot_graphs(G1_G2_pair_list, result_mapping)
    
    # ----------
    
    # Saving the Results 
    logging_result_data_folder_path = '/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/BQM Approach/Working Code to Use(BQM) 2.0/'
    # matrix_form_result_data_folder_path = os.path.join(logging_result_data_folder_path, 'Matrix Form Experiment Result Data')
    # matrix_form_result_data_file_path = os.path.join(matrix_form_result_data_folder_path, 'matrix_H1_bqm_experiment_result_data.csv') 
    # print('matrix_form_result_data_file_path:', matrix_form_result_data_file_path)
    pyqubo_form_result_data_folder_path = os.path.join(logging_result_data_folder_path, 'PyQUBO Form Experiment Result Data')
    pyqubo_form_result_data_file_path = os.path.join(pyqubo_form_result_data_folder_path, 'pyqubo_H1_bqm_experiment_result_data.csv')
    print('pyqubo_form_result_data_file_path:', pyqubo_form_result_data_file_path)
    
    # Assigning data to variables store as dataset result columns
    n1 = args.n1 # 'number of nodes in G1'
    e1 = G1_G2_pair_list[0].number_of_edges() # 'number of edges in G1'
    n2 = args.n2 # 'number of nodes in G2'
    e2 = G1_G2_pair_list[1].number_of_edges() # 'number of edges in G2'
    isomorphic_pair = True # Whether the G1 G2 pair is isomorphic or not
    
    experiment_label =  'matrix_H1_bqm_experiment_result_data' + key
    # experiment_label =  'pyqubo_H1_bqm_experiment_result_data' + key
    
    best_mapping_energy = sampleset.first.energy 
    dwave_leap_problem_id = sampleset.info["problem_id"] # get DWave problem ID from 'testing_and_evaluation_Script'
    no_annealing_cycles = 100 # placeholder valuefor 'number of annealing cycles'
    duration_per_anneal_cycle = 20 # placeholder value, in microseconds, for the 'duration of each anneal cycle'
    qpu_access_time = sampleset.info["timing"]["qpu_access_time"] # time, in microseconds, of the qpu access time (see description of qpu access time in Notes and 'testing_and_evaluation_script')
    qpu_programming_time = sampleset.info["timing"]["qpu_programming_time"]
    qpu_sampling_time = sampleset.info["timing"]["qpu_sampling_time"]
    qpu_anneal_time_per_sample = sampleset.info["timing"]["qpu_anneal_time_per_sample"]
    qpu_readout_time_per_sample = sampleset.info["timing"]["qpu_readout_time_per_sample"]
    qpu_access_overhead_time = sampleset.info["timing"]["qpu_access_overhead_time"]

    # charge_time = sampleset.info["charge_time"] # time, in microseconds, of the charge time (see description of chargetime in Notes and 'testing_and_evaluation_script')
    # total_post_processing_time = sampleset.info["total_post_processing_time"]
    # post_processing_overhead_time = sampleset.info["post_processing_overhead_time"]
    
    
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
        'number of annealing cycles',
        'qpu access time',
        'qpu programming time',
        'qpu sampling time',
        'qpu anneal time per sample',
        'qpu readout time per sample',
        'qpu access overhead time',
        'problem id',
        #'runtime', 
        #'charge time', 
        ]
    
    result_data_for_csv = [{
        'key': key,
        'experiment label': experiment_label,  
        'number of nodes in G1': n1,
        'number of edges in G1': e1, 
        'number of nodes in G2': n2, 
        'number of edges in G2': e2,
        'isomorphic pair?': isomorphic_pair, 
        'mapping from G2 to G1': result_mapping, 
        'best mapping energy': best_mapping_energy,
        'number of annealing cycles': no_annealing_cycles,
        'qpu access time': qpu_access_time, 
        'qpu programming time': qpu_programming_time,
        'qpu sampling time': qpu_sampling_time,
        'qpu anneal time per sample': qpu_anneal_time_per_sample,
        'qpu readout time per sample': qpu_readout_time_per_sample,
        'qpu access overhead time': qpu_access_overhead_time,
        'problem id': dwave_leap_problem_id,
        }]
    
    # Saving to Matrix Form Experiment Result Data
    if not os.path.exists(matrix_form_result_data_folder_path):
        os.makedirs(matrix_form_result_data_folder_path)
        df_for_csv = pd.DataFrame(result_data_for_csv, columns = columns)
        df_for_csv.to_csv(matrix_form_result_data_file_path, header = False, index = False)
        print("Matrix Form CSV Created!")
    
    # Create another row and append row (as df) to existing CSV
    # row = [{'A':'X1', 'B':'X2', 'C':'X3'}]
    df_new_row = pd.DataFrame(result_data_for_csv)
    df_new_row.to_csv(matrix_form_result_data_file_path, mode='a', header=False, index=False)
    print("Row added to Matrix From DataFrame!")

    # print out what you just stored into the `Matrix Form Experiment Result Dataset`
    df_from_csv = pd.read_csv(matrix_form_result_data_file_path)
    print(df_from_csv)    
    
    
    # Saving to PyQUBO Form Experiment Result Data
    if not os.path.exists(pyqubo_form_result_data_folder_path):
        os.makedirs(pyqubo_form_result_data_folder_path)
        df_for_csv = pd.DataFrame(result_data_for_csv, columns = columns)
        df_for_csv.to_csv(pyqubo_form_result_data_file_path, header = False, index = False)
        print("PyQUBO Form CSV Created!")
    
    # Create another row and append row (as df) to existing CSV
    # row = [{'A':'X1', 'B':'X2', 'C':'X3'}]
    df_new_row = pd.DataFrame(result_data_for_csv)
    df_new_row.to_csv(pyqubo_form_result_data_file_path, mode='a', header=False, index=False)
    print("Row added to PyQUBO From DataFrame!")

    # print out what you just stored into the `PyQUBO Form Experiment Result Dataset`
    df_from_csv = pd.read_csv(pyqubo_form_result_data_file_path)
    print(df_from_csv)
    

    # Original code for saving the file
    # if not os.path.exists(logging_result_data_folder_path):
    #     os.makedirs(logging_result_data_folder_path)
    #     df_for_csv = pd.DataFrame(result_data_for_csv, columns = columns)
    #     df_for_csv.to_csv(logging_result_data_file_path, header = False, index = False)
    #     print("CSV Created!")
    
    # Create another row and append row (as df) to existing CSV
    # row = [{'A':'X1', 'B':'X2', 'C':'X3'}]
    # df_new_row = pd.DataFrame(result_data_for_csv)
    # df_new_row.to_csv(logging_result_data_file_path, mode='a', header=False, index=False)
    # print("Row added to DataFrame!")

    # print out what you just stored into the `Experiment Result Dataset`
    # df_from_csv = pd.read_csv(logging_result_data_file_path)
    # print(df_from_csv)
    
    # ---------- x ---------- x ---------- x ----------

    """ For Running the entire pipeline in a loop over all the graph-subgraph pairs to generate 
    a series of results for the `Experiment Results Dataset` """
    """
    # Taking the upper limit of number of nodes in G1 as 50, lower limit as 5
    # Taking the upper limit of number of nodes in G2 as 49, lower limit as 3
    # Taking step of 5

    
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
        'number of annealing cycles',
        'qpu access time',
        'qpu programming time',
        'qpu sampling time',
        'qpu anneal time per sample',
        'qpu readout time per sample',
        'qpu access overhead time',
        'problem id',
        #'runtime', 
        #'charge time', 
        ]

    # Saving the Results 
    logging_result_data_folder_path = '/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/BQM Approach/Working Code to Use(BQM) 2.0/'
    matrix_form_result_data_folder_path = os.path.join(logging_result_data_folder_path, 'Matrix Form Experiment Result Data')
    matrix_form_result_data_file_path = os.path.join(matrix_form_result_data_folder_path, 'matrix_H1_bqm_experiment_result_data.csv') 
    print('matrix_form_result_data_file_path:', matrix_form_result_data_file_path)
    # pyqubo_form_result_data_folder_path = os.path.join(logging_result_data_folder_path, 'PyQUBO Form Experiment Result Data')
    # pyqubo_form_result_data_file_path = os.path.join(pyqubo_form_result_data_folder_path, 'pyqubo_H1_bqm_experiment_result_data.csv')
    # print('pyqubo_form_result_data_file_path:', pyqubo_form_result_data_file_path)

    for i in range(8, 30, 2):
        n1 = i
        for j in range(4, 28, 2):
            n2 = j
            if n2 < n1:
                key = "G1(" + str(n1) + ")_G2(" + str(n2) + ")_iso"  # "G1(20)_G2(19)"
                G1_G2_pair_list = G1_G2_pair_dict_loaded[key]
                print("Graph - SubGraph Pair:{}".format(G1_G2_pair_list))
                
                # For BQM Matrix Form
                
                sampleset, sample_translation_dictionary, best_sample = sgi_bqm_matrix_form_script_for_experiment.find_isomorphism(G1_G2_pair_list)
                # print("Sampleset is:{}".format(sampleset))
                sampleset_agg = sampleset.aggregate()
                print("Sub-GI Results are \n :{}".format(best_sample))    
                # We can use an annealing sample and the sample translation dictionary from before to
                # generate a dictionary that maps nodes from the graph to be embedded to the target graph
                result_mapping = gsgm_mf.translate_sample(best_sample, sample_translation_dictionary)
                            
                # ----------
                
                # For BQM PyQUBO Form
                
                sampleset, sample_translation_dictionary, best_sample = sgi_bqm_pyqubo_form_script_for_experiment.find_isomorphism(G1_G2_pair_list)
                # We can use the PyQUBO-translated annealing sample and the sample translation dictionary
                # from before to generate a dictionary that maps nodes from the graph to be embedded to the
                # target graph
                result_mapping = gsgm_pqf.translate_sample(best_sample, sample_translation_dictionary)   
                
                # ----------     

                e1 = G1_G2_pair_list[0].number_of_edges() # 'number of edges in G1'
                e2 = G1_G2_pair_list[1].number_of_edges() # 'number of edges in G1'
                isomorphic_pair = True # Whether the G1 G2 pair is isomorphic or not
                
                experiment_label =  'matrix_H1_bqm_experiment_result_data' + key
                # experiment_label =  'pyqubo_H1_bqm_experiment_result_data' + key
                
                best_mapping_energy = sampleset.first.energy 
                dwave_leap_problem_id = sampleset.info["problem_id"] # get DWave problem ID from 'testing_and_evaluation_Script'
                no_annealing_cycles = 100 # placeholder valuefor 'number of annealing cycles'
                duration_per_anneal_cycle = 20 # placeholder value, in microseconds, for the 'duration of each anneal cycle'
                qpu_access_time = sampleset.info["timing"]["qpu_access_time"] # time, in microseconds, of the qpu access time (see description of qpu access time in Notes and 'testing_and_evaluation_script')
                qpu_programming_time = sampleset.info["timing"]["qpu_programming_time"]
                qpu_sampling_time = sampleset.info["timing"]["qpu_sampling_time"]
                qpu_anneal_time_per_sample = sampleset.info["timing"]["qpu_anneal_time_per_sample"]
                qpu_readout_time_per_sample = sampleset.info["timing"]["qpu_readout_time_per_sample"]
                qpu_access_overhead_time = sampleset.info["timing"]["qpu_access_overhead_time"]
                
                result_data_for_csv = [{
                    'key': key,
                    'experiment label': experiment_label,  
                    'number of nodes in G1': n1,
                    'number of edges in G1': e1, 
                    'number of nodes in G2': n2, 
                    'number of edges in G2': e2,
                    'isomorphic pair?': isomorphic_pair, 
                    'mapping from G2 to G1': result_mapping, 
                    'best mapping energy': best_mapping_energy,
                    'number of annealing cycles': no_annealing_cycles,
                    'qpu access time': qpu_access_time, 
                    'qpu programming time': qpu_programming_time,
                    'qpu sampling time': qpu_sampling_time,
                    'qpu anneal time per sample': qpu_anneal_time_per_sample,
                    'qpu readout time per sample': qpu_readout_time_per_sample,
                    'qpu access overhead time': qpu_access_overhead_time,
                    'problem id': dwave_leap_problem_id,
                }]

                # Saving to Matrix Form Experiment Result Data
                if not os.path.exists(matrix_form_result_data_folder_path):
                    os.makedirs(matrix_form_result_data_folder_path)
                    df_for_csv = pd.DataFrame(result_data_for_csv, columns = columns)
                    df_for_csv.to_csv(matrix_form_result_data_file_path, header = False, index = False)
                    print("Matrix Form CSV Created!")
                
                # Create another row and append row (as df) to existing CSV
                # row = [{'A':'X1', 'B':'X2', 'C':'X3'}]
                df_new_row = pd.DataFrame(result_data_for_csv)
                df_new_row.to_csv(matrix_form_result_data_file_path, mode='a', header=False, index=False)
                print("Row added to Matrix From DataFrame!")

                # print out what you just stored into the `Matrix Form Experiment Result Dataset`
                df_from_csv = pd.read_csv(matrix_form_result_data_file_path)
                print(df_from_csv)    
                
                
                # Saving to PyQUBO Form Experiment Result Data
                
                if not os.path.exists(pyqubo_form_result_data_folder_path):
                    os.makedirs(pyqubo_form_result_data_folder_path)
                    df_for_csv = pd.DataFrame(result_data_for_csv, columns = columns)
                    df_for_csv.to_csv(pyqubo_form_result_data_file_path, header = False, index = False)
                    print("PyQUBO Form CSV Created!")
                
                # Create another row and append row (as df) to existing CSV
                # row = [{'A':'X1', 'B':'X2', 'C':'X3'}]
                df_new_row = pd.DataFrame(result_data_for_csv)
                df_new_row.to_csv(pyqubo_form_result_data_file_path, mode='a', header=False, index=False)
                print("Row added to PyQUBO From DataFrame!")

                # print out what you just stored into the `PyQUBO Form Experiment Result Dataset`
                df_from_csv = pd.read_csv(pyqubo_form_result_data_file_path)
                print(df_from_csv)
                

                # Original Code for Saving the File
                # if not os.path.exists(logging_result_data_folder_path):
                #     os.makedirs(logging_result_data_folder_path)
                #     df_for_csv = pd.DataFrame(result_data_for_csv, columns = columns)
                #     df_for_csv.to_csv(logging_result_data_file_path, header = False, index = False)
                #     print("CSV Created!")

                # Create another row and append row (as df) to existing CSV
                # row = [{'A':'X1', 'B':'X2', 'C':'X3'}]
                # df_new_row = pd.DataFrame(result_data_for_csv)
                # df_new_row.to_csv(logging_result_data_file_path, mode='a', header=False, index=False)
                # print("Row added to DataFrame!")

                # print out what you just stored into the `Experiment Result Dataset`
                # df_from_csv = pd.read_csv(logging_result_data_file_path)
                # print(df_from_csv)
                time.sleep(2)
    
    # ---------- x ---------- x ---------- x ----------
    """
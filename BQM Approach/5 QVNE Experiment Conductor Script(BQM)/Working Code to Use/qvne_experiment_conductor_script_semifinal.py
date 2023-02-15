"""
BQM Approach 
Semi-Final Script. Works perfectly well but not used in the Final version of the Experiment.
Final Script is 'Working Code to Use' Folder with the other scripts and the conductor script.

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
import numpy as np  
import dimod
import time 
import argparse
import os
import pandas as pd
import src.gsgmorph.pyqubo_form as gsgm_pqf


# Ignore errors importing matpotlib.pyplot
try:
    import matplotlib.pyplot as plt  
    import matplotlib.colors as mcolors
except ImportError:
    pass

# Importing the scripts and their functions

# importing `graph_subgraph_pair_generator_script_semifinal.py` script
graph_subgraph_pair_generator_script_semifinal_path = '/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/BQM Approach/1 Generate Graph - SubGraph Pair Dataset(BQM) /Working Code to Use/'
sys.path.insert(1, graph_subgraph_pair_generator_script_semifinal_path)
import graph_subgraph_pair_generator_script_semifinal


"""
# importing `sgi_bqm_matrix_form_semifinal.py`
sgi_bqm_matrix_form_semifinal_path = '/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/BQM Approach/2 Sub-Graph Isomorphism Virtual Network Embedding(BQM)/Working Code to Use/'
sys.path.insert(1, sgi_bqm_matrix_form_semifinal_path)
# Matrix Form 
import sgi_bqm_matrix_form_semifinal
"""

# importing `sgi_bqm_pyqubo_form_semifinal.py`
sgi_bqm_pyqubo_form_semifinal_path = '/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/BQM Approach/2 Sub-Graph Isomorphism Virtual Network Embedding(BQM)/Working Code to Use/'
sys.path.insert(1, sgi_bqm_pyqubo_form_semifinal_path)
# PyQUBO Form 
import sgi_bqm_pyqubo_form_semifinal


# importing `testing_and_evaluation_seminfinal.py` script
testing_and_evaluation_seminfinal_path = '/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/BQM Approach/3 Testing and Evaluation(BQM)/Working Code to Use/'
sys.path.insert(1, testing_and_evaluation_seminfinal_path)
import testing_and_evaluation_seminfinal

"""
# importing `logging_results_data_working_semifinal.py` script 
logging_results_data_working_semifinal_path = '/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/BQM Approach/4 Logging Results Data(BQM)/Working Code to Use/'
sys.path.insert(1, logging_results_data_working_semifinal_path)
import logging_results_data_working_semifinal
"""

if __name__ == "__main__":

    """ Generating Graph-SubGraph pair Dataset """
    dataset_filepath = "/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/BQM Approach/Dataset(BQM)/G1_G2_PAIR_DICTIONARY_DATASET.pickle"
    """
    print("Enter the Number of nodes in Parent Graph (G1) as a command-line argument (n1)")
    n1 = int(sys.argv[1])

    print("Default p (probability of there being an edge between two nodes) is 0.5 i.e. 50%")
    # p is chosen to be 0.5 as per Zick's paper.  
    p = 0.5
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
        G1 = graph_subgraph_pair_generator_script.generate_random_parent_graph(i, p)
        for j in range(3, n2 + 1):
            # This loop iterates over the number of nodes G2 can have. 
            # range() is written this way because it is only meaningful to start having subgraphs 
            # that at least have 3 nodes and make a triangle.
            # the reason behind chosing `n2 + 1` as the upper limit of the range() is the same as 
            # that for the above loop for G1. 
            
            G2 = graph_subgraph_pair_generator_script.generate_child_subgraph(G1, j)
            key = "G1(" + str(i) + ")_G2(" + str(j) + ")_iso" # G1(20)_G2(19)_iso
            G1_G2_pair_dict[key] = [G1, G2]


    # Saving the Graph-SubGraph Pair Dataset in the designated directory
    graph_subgraph_pair_generator_script.save_G1_G2_pair_dict(dataset_filepath, G1_G2_pair_dict)
    print("DONE!")
    """

    """ Retreiving the G-SG Pair Dataset from the Directory """
    # Commented out because this script is only for saving the graph pairs generated
    
    G1_G2_pair_dict_loaded = graph_subgraph_pair_generator_script_semifinal.load_G1_G2_pair_dictionary(dataset_filepath)
    # print("Graph-SubGraph Pair Dictionary Dataset:")
    # print(G1_G2_pair_dict_loaded)
    print("The Dataset has {} Graph-SubGraph Pairs.".format(len(G1_G2_pair_dict_loaded)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--n1")
    parser.add_argument("--n2")
    args = parser.parse_args()

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
        'timing'
    ]

    logging_result_data_folder_path = '/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/BQM Approach/Experiment Result Data(BQM)/'
    
    n2 = int(args.n1) - 1
    for i in range(5, int(args.n1), 5):
        n1 = i
        for j in range(3, n2, 5):
            n2 = j
            if n2 < n1:
                # whatever key you want with the format: G1(<n1>)_G2(<n2>)_iso
                key = "G1(" + str(i) + ")_G2(" + str(j) + ")_iso"  # "G1(20)_G2(19)_iso"
                G1_G2_pair_list = G1_G2_pair_dict_loaded[key]
                print("Graph - SubGraph Pair:{}".format(G1_G2_pair_list))

                # FOR BQM MATRIX FORM 
                """ THE SUBGRAPH ISOMORPHISM Problem Implmented for the Quantum Virtual Network Embedding Solved on the DWave """
                """
                # sgi_qvne_H1.present_results(sgi_qvne_H1.find_isomorphism(sgi_qvne_H1.edges_to_graph()))
                sampleset_std = sgi_bqm_matrix_form_semifinal.find_isomorphism(G1_G2_pair_list)
                sampleset = sampleset_std[0] # sampleset
                sample_translation_dict = sampleset_std[1] # sample_transalation_dictionary
                
                sampleset = sampleset.aggregate()
                best_sample = sampleset.first
                print("Sub-GI Results are \n :{}".format(best_sample))    
                # We can use an annealing sample and the sample translation dictionary from before to
                # generate a dictionary that maps nodes from the graph to be embedded to the target graph
                result_mapping = gsgm_mf.translate_sample(best_sample, sample_translation_dict)
                """
                
                # FOR BQM PyQUBO FORM
                """ THE SUBGRAPH ISOMORPHISM Problem Implmented for the Quantum Virtual Network Embedding Solved on the DWave """
                sampleset_std_bs = sgi_bqm_pyqubo_form_semifinal.find_isomorphism(G1_G2_pair_list)
                # We can use the PyQUBO-translated annealing sample and the sample translation dictionary
                # from before to generate a dictionary that maps nodes from the graph to be embedded to the
                # target graph
                sampleset = sampleset_std_bs[0]
                sample_translation_dict = sampleset_std_bs[1]
                best_sample = sampleset_std_bs[2]
                result_mapping = gsgm_pqf.translate_sample(best_sample, sample_translation_dict)

                
                e1 = G1_G2_pair_list[0].number_of_edges() # 'number of edges in G1'
                e2 = G1_G2_pair_list[1].number_of_edges() # 'number of edges in G2'
                isomorphic_pair = True # Whether the G1 G2 pair is isomorphic or not
                experiment_label =  'qvne_bqm_matrix_form_experiment_result_data_' + key
                # experiment_label =  'qvne_bqm_pyqubo_form_experiment_result_data_' + key
                #(for matrix form) best_mapping_energy = sampleset.first.energy
                best_mapping_energy = best_sample.energy 
                timing = sampleset.info["timing"] # time, in microseconds, of the run time (see description of runtime in Notes and 'testing_and_evaluation_script')

                result_data_for_csv = [{
                    'key': key,
                    'experiment label': experiment_label,  
                    'number of nodes in G1': n1,
                    'number of edges in G1': e1, 
                    'number of nodes in G2': n2, 
                    'number of edges in G2': e2,
                    'isomorphic pair?': isomorphic_pair, 
                    'best mapping from G2 to G1': result_mapping, 
                    'best mapping energy': best_mapping_energy,
                    'timing': timing,
                }]

                logging_result_data_file_path = logging_result_data_folder_path + 'qvne_bqm_matrix_form_experiment_result_data.csv'

                if not os.path.exists(logging_result_data_folder_path):
                    os.makedirs(logging_result_data_folder_path)
                    df_for_csv = pd.DataFrame(result_data_for_csv, columns = columns)
                    df_for_csv.to_csv(logging_result_data_file_path, header = False, index = False)
                    print("CSV Created!")
                
                # Create another row and append row (as df) to existing CSV
                #row = [{'A':'X1', 'B':'X2', 'C':'X3'}]
                df_new_row = pd.DataFrame(result_data_for_csv)
                df_new_row.to_csv(logging_result_data_file_path, mode='a', header=False, index=False)
                print("Row added to DataFrame!")

                df_from_csv = pd.read_csv(logging_result_data_file_path)
                print(df_from_csv)
        
    

    
    """ USEFULL, COMMENTD FOR for() LOOP
    # whatever key you want with the format: G1(<n1>)_G2(<n2>)_iso
    key = "G1(" + args.n1 + ")_G2(" + args.n2 + ")_iso"  # "G1(20)_G2(19)_iso"
    G1_G2_pair_list = G1_G2_pair_dict_loaded[key]
    print("Graph - SubGraph Pair:{}".format(G1_G2_pair_list))
    graph_subgraph_pair_generator_script_semifinal.plot_graphs(G1_G2_pair_list)
    """

    """ THE SUBGRAPH ISOMORPHISM Problem Implmented for the Quantum Virtual Network Embedding Solved on the DWave """
    # sgi_qvne_H1.present_results(sgi_qvne_H1.find_isomorphism(sgi_qvne_H1.edges_to_graph()))
    """ USEFULL, COMMENTD FOR for() LOOP
    sampleset_std = sgi_bqm_matrix_form_semifinal.find_isomorphism(G1_G2_pair_list)
    sampleset = sampleset_std[0] # sampleset
    print("Sampleset is:{}".format(sampleset))

    
    sample_translation_dict = sampleset_std[1] # sample_transalation_dictionary
    sampleset = sampleset.aggregate()
    print("Sampleset type:{}".format(type(sampleset)))

    best_sample = sampleset.first
    print("Sub-GI Results are \n :{}".format(best_sample))    
    # We can use an annealing sample and the sample translation dictionary from before to
    # generate a dictionary that maps nodes from the graph to be embedded to the target graph
    result_mapping = gsgm_mf.translate_sample(best_sample, sample_translation_dict)
    sgi_bqm_matrix_form_semifinal.plot_graphs(G1_G2_pair_list, result_mapping )
 
    # G1_G2_sampleset = [G1, G2, results] - A list which has 
    # First element - G1 graph
    # Second element - G2 graph
    # results - Resultant mappings from 
    # G2 graph to G2 graph 
    """   
    
    """ Testing and Evaluating the Results """
    """ USEFULL, COMMENTD FOR for() LOOP
    n1 = args.n1 # 'number of nodes in G1'
    n2 = args.n2 # 'number of nodes in G2'
    e1 = G1_G2_pair_list[0].number_of_edges() # 'number of edges in G1'
    e2 = G1_G2_pair_list[1].number_of_edges() # 'number of edges in G2'
    isomorphic_pair = True # Whether the G1 G2 pair is isomorphic or not
    experiment_label =  'qvne_bqm_matrix_form_experiment_result_data_' + key
    # experiment_label =  'qvne_bqm_pyqubo_form_experiment_result_data_' + key

    best_mapping_energy = sampleset.first.energy 
    timing = sampleset.info["timing"] # time, in microseconds, of the run time (see description of runtime in Notes and 'testing_and_evaluation_script')
    dwave_leap_problem_id = sampleset.info["problem_id"] # get DWave problem ID from 'testing_and_evaluation_Script'
    no_annealing_cycles = len(sampleset) # placeholder valuefor 'number of annealing cycles'
    # solver_information = Commented out because there a re a lot of data points, they are present in the 'testing_and_evaluation_script' 
    """
    """
    charge_time = sampleset.info["charge_time"] # time, in microseconds, of the charge time (see description of chargetime in Notes and 'testing_and_evaluation_script')
    qpu_access_time = sampleset.info["qpu_access_time"] # time, in microseconds, of the qpu access time (see description of qpu access time in Notes and 'testing_and_evaluation_script')
    qpu_anneal_time_per_sample = sampleset.info["qpu_anneal_time_per_sample"]
    qpu_readout_time_per_sample = sampleset.info["qpu_readout_time_per_sample"]
    qpu_sampling_time = sampleset.info["qpu_sampling_time"]
    qpu_access_overhead_time = sampleset.info["qpu_access_overhead_time"]
    qpu_programming_time = sampleset.info["qpu_programming_time"]
    total_post_processing_time = sampleset.info["total_post_processing_time"]
    post_processing_overhead_time = sampleset.info["post_processing_overhead_time"]
    """
    

    """ Logging the results and generating a report of each experiment """
    """ USEFULL, COMMENTED FOR for() LOOP
    columns = [
        'key',
        'experiment_label', 
        'number of nodes in G1', 
        'number of edges in G1', 
        'number of nodes in G2', 
        'number of edges in G2',
        'isomorphic pair?',  
        'mapping from G2 to G1',
        'best mapping energy', 
        'timing'
        #'runtime', 
        #'charge time', 
        #'qpu access time',
        #'dwave leap problem id', 
        #'number of annealing cycles',
        #'qpu anneal time per sample',
        #'qpu readout time per sample',
        #'qpu sampling time',
        #'qpu access overhead time',
        #'qpu programming time', 
        #'total post processing time', 
        #'post processing overhead time'
    ]
    """    
     
    
    """
    result_data_for_csv = {
        'key': [key],
        'experiment_label': [experiment_label],  
        'number of nodes in G1': [n1],
        'number of edges in G1': [e1], 
        'number of nodes in G2': [n2], 
        'number of edges in G2': [e2],
        'isomorphic_pair?': [isomorphic_pair], 
        'best mapping from G2 to G1': [best_mapping],
        'best mapping energy': [best_mapping_energy], 
        'runtime': [runtime],
        'charge time': [charge_time],
        'qpu access time': [qpu_access_time], 
        'dwave_leap_problem_id': [dwave_leap_problem_id],
        'number of annealing cycles': [no_annealing_cycles],
        'duration per anneal cycle': [duration_per_anneal_cycle]
    }

    result_data_for_csv = [
        [key, experiment_label, n1, e1, n2,  
        e2, isomorphic_pair, best_mapping, best_mapping_energy, 
        runtime, charge_time, qpu_access_time, dwave_leap_problem_id, no_annealing_cycles, duration_per_anneal_cycle]
    ]
    """

    """ USEFULL, COMMENTED FOR for() LOOP
    result_data_for_csv = [{
        'key': key,
        'experiment label': experiment_label,  
        'number of nodes in G1': n1,
        'number of edges in G1': e1, 
        'number of nodes in G2': n2, 
        'number of edges in G2': e2,
        'isomorphic pair?': isomorphic_pair, 
        'best mapping from G2 to G1': result_mapping, 
        'best mapping energy': best_mapping_energy,
        'timing': timing,
        #'charge time': charge_time,
        #'qpu access time': qpu_access_time, 
        #'dwave leap problem id': dwave_leap_problem_id,
        #'number of annealing cycles': no_annealing_cycles,
        #'duration per anneal cycle': duration_per_anneal_cycle,
        #'qpu anneal time per sample': qpu_anneal_time_per_sample,
        #'qpu readout time per sample': qpu_readout_time_per_sample,
        #'qpu sampling time': qpu_sampling_time,
        #'qpu access overhead time': qpu_access_overhead_time,
        #'qpu programming time': qpu_programming_time, 
        #'total post processing time': total_post_processing_time,  
        #'post processing overhead time': post_processing_overhead_time
    }]
    
    logging_result_data_folder_path = '/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/BQM Approach/Experiment Result Data(BQM)/'
    logging_result_data_file_path = logging_result_data_folder_path + 'qvne_bqm_matrix_form_experiment_result_data.csv'

    if not os.path.exists(logging_result_data_folder_path):
        os.makedirs(logging_result_data_folder_path)
        df_for_csv = pd.DataFrame(result_data_for_csv, columns = columns)
        df_for_csv.to_csv(logging_result_data_file_path, header = False, index = False)
        print("CSV Created!")
    
    # Create another row and append row (as df) to existing CSV
    #row = [{'A':'X1', 'B':'X2', 'C':'X3'}]
    df_new_row = pd.DataFrame(result_data_for_csv)
    df_new_row.to_csv(logging_result_data_file_path, mode='a', header=False, index=False)
    print("Row added to DataFrame!")
    

    df_from_csv = pd.read_csv(logging_result_data_file_path)
    print(df_from_csv)
    """
"""
Script that Conducts the experiments and coordinated the activities of all the scripts 
Can be considered as the MASTER SCRIPT
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


# Ignore errors importing matpotlib.pyplot
try:
    import matplotlib.pyplot as plt  
    import matplotlib.colors as mcolors
except ImportError:
    pass

# Importing the scripts and their functions

# importing `graph_subgraph_pair_generator_script.py` script
graph_subgraph_pair_generator_script_path = '/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/1 Generate Graph - SubGraph Pairs/Working Code to Use'
sys.path.insert(1,graph_subgraph_pair_generator_script_path)
import graph_subgraph_pair_generator_script

# importing `sgi_qvne_H1.py`
sgi_qvne_H1_script_path = '/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/2 Sub-Graph Isomorphism Virtual Network Embedding/Working Code to Use'
sys.path.insert(1, sgi_qvne_H1_script_path)
# H1 is the baseline Hamiltonian, 
# H2 is the compact Hamiltonian (tweaked and fine tuned for the the Virtual Network Embedding problem)
import sgi_qvne_H1

# importing `testing_and_evaluation.py` script
testing_and_evaluation_script_path = '/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/3 Testing and Evaluation/Working Code to Use'
sys.path.insert(1,testing_and_evaluation_script_path )
import testing_and_evaluation


if __name__ == "__main__":

    """ Generating Graph-SubGraph pair Dataset """
    dataset_filepath = "/Users/samyakjhaveri/Desktop/Drive Folder/Research/USC Internship Quantum Virtual Network Embedding Project 2022/Dataset/G_SG_PAIR_DICTIONARY_DATASET.pickle"
    """
    print("Enter the Number of nodes in Parent Graph (G) as a command-line argument (n1)")
    n1 = int(sys.argv[1])

    print("Default p (probability of there being an edge between two nodes) is 0.5 i.e. 50%")
    # p is chosen to be 0.5 as per Zick's paper.  
    p = 0.5
    n2 = n1 - 1 # if n1 = 20, then n2 = 20 - 1 = 19, n2 is number of nodes possible in SubGraph
    g_sg_pair_dict = {} # dictionary to keep all the graph-subgraph pairs in one place. 
    # Will store only this one dictinoary instead of storing every graph-subgraph pair in 
    # a separate file that has to be stored and retreived everytime a new pair is 
    # generated which makes the whole process computationally heavy


    key = ""

    for i in range(5, n1 + 1):
        # This loop iterates over the number of nodes G can have.
        # range() is written this way as it only meaningful to have a G with at least 5 nodes
        # and the upper limit of the range is chosen as `n1+1` because range() with upper limit 
        # as `n1` does not include the last number (i.e. `n1` number of nodes)
        for j in range(3, n2 + 1):
            # This loop iterates over the number of nodes SG can have. 
            # range() is written this way because it is only meaningful to start having subgraphs 
            # that at least have 3 nodes and make a triangle.
            # the reason behind chosing `n2 + 1` as the upper limit of the range() is the same as 
            # that for the above loop for G. 
            G = graph_subgraph_pair_generator_script.generate_random_parent_graph(i, p)
            SG = graph_subgraph_pair_generator_script.generate_child_subgraph(G, j)
            key = "G" + str(i) + "_SG" + str(j)
            g_sg_pair_dict[key] = [G, SG]


    # Saving the Graph-SubGraph Pair Dataset in the designated directory
    graph_subgraph_pair_generator_script.save_g_sg_pair_dict(dataset_filepath, g_sg_pair_dict)
    print("DONE!")
    """

    """ Retreiving the G-SG Pair Dataset from the Directory """
    # Commented out because this script is only for saving the graph pairs generated
    g_sg_pair_dict_loaded = graph_subgraph_pair_generator_script.load_g_sg_pair_dictionary(dataset_filepath)
    print("Graph-SubGraph Pair Dictionary Dataset:")
    print(g_sg_pair_dict_loaded)
    print("The Dataset has {} Graph-SubGraph Pairs.".format(len(g_sg_pair_dict_loaded)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--n1")
    parser.add_argument("--n2")
    args = parser.parse_args()

    # whatever key you want with the format: G<n1>_SG<n2>
    key = "G" + args.n1 + "_SG" + args.n2  # "G20_SG19"
    g_sg_pair_list = g_sg_pair_dict_loaded[key]
    print("Graph - SubGraph Pair:{}".format(g_sg_pair_list))
    graph_subgraph_pair_generator_script.plot_graphs(g_sg_pair_list)
    print("Type of G-SG pair:{}".format(type(g_sg_pair_list[0])))

    """ THE SUBGRAPH ISOMORPHISM Problem Implmented for the Quantum Virtual Network Embedding  Solved on the DWave """
    # sgi_qvne_H1.present_results(sgi_qvne_H1.find_isomorphism(sgi_qvne_H1.edges_to_graph()))
    sgi_qvne_H1.present_results(sgi_qvne_H1.find_isomorphism(g_sg_pair_list))

    # annealing_results = sgi_qccd_modified_H1.find_isomorphism([NXGraph_G, NXGraph_SG])
    # Here, `annealing_results` in the form of 
    # [G1, G2, results] - A list which has 
    # First element - G1 graph
    # Second element - G2 graph
    # results - Resultant mappings from 
    # G2 graph to G2 graph
    # Made a separate `annealing_results` object so that I could use it with the last step of 
    # 'Storing the results and generating a report of each experiment`
    #sgi_qccd_modified_H1.present_results(annealing_results)
    """
    """ # Testing and Evaluating the Results """



    """ Storing the results and generating a report of each experiment """


    """ References:
    - Using Pickle - https://ianlondon.github.io/blog/pickling-basics/
    """
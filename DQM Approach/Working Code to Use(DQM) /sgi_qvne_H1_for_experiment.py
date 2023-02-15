"""
Final Working Script to be used for Experiments. 

Main Script that has the SubGraph Isomorphism Algorithm that creates a DQM Hamiltonian 
Equation to solve on the DWave LeapHybridDQMSampler(). 
This script also has funtions that present the results of the Annealing visually as 
plotted graphs and in text form on the Terminal.
This script is named with 'H1' as it has the baseline Hamiltonian algorithm and not 
the compact one, as inspired from Zick's paper on 'Experimental Quantum Annealing...'
The script with the modified compact Hamilotnian equation will be with 'H2' in its name. 

"""

# Import Stuff
import itertools
import numpy as np  
import networkx as nx  
import dimod
from dwave.system import LeapHybridDQMSampler
import time 
from dwave.cloud.solver import DQMSolver
import sys
from dwave.cloud import Client

# importing `testing_and_evaluation.py` script
import testing_and_evaluation_for_experiment

# Ignore errors importing matpotlib.pyplot
try:
    import matplotlib.pyplot as plt  
    import matplotlib.colors as mcolors
except ImportError:
    pass


def create_graph_helper_function(edges):
    """
    Generate Graph from edges list

    Args:
    - edges: List of node pairs that have edges between them 
    Returns: 
        Graph (NetworkX)
    """ 

    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def create_dqm_helper_function(G1, G2):
    """  
    Construct DQM based on two graphs 

    Create a discrete quadratic model
    to represent the problem of finding an isomorphism between the two graphs

    Args:
    - G1 Target Graph (networkx.Graph)
    - G2 Graph to Embed (networkx.Graph)
    
    Returns:
        DiscreteQuadraticModel
    """  

    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()
    
    if n1 < n2:
        raise ValueError
    
    G1_nodes = list(G1.nodes)
    G1_edges = list(G1.edges)
    G2_nodes = list(G2.nodes)
    G2_edges = list(G2.edges)
    
    print("G1 nodes:{}".format(G1_nodes))
    print("G1 edges:{}".format(G1_edges))
    print("G2 nodes:{}".format(G2_nodes))
    print("G2 edges:{}".format(G2_edges))

    dqm = dimod.DiscreteQuadraticModel()

    # In this formulation, the offset is equal to the number of nodes (multiplied by 
    # the penalty coefficient A, currently taken to be 1. 
    # Note that in a BQM formualtion, the offset would be twice as much due to the 
    # need to use a penalty model for each direction of the bijection constraint. 
  
    # The loop below ensures that each node in the graph G1 is chosen, 
    # and it is chosen once - i.e. it takes care of bijectivity. 
    # However, for Sub-Graph Isomorphism, bijectivity does not work. Injectivity works. 
    # Therefore, every node in the smaller graph G2 has to be mapped to a node in the 
    # larger graph G1, but every node in graph G1 need not be mapped to a node in 
    # graph G2. 
    
    """ updated code for H_A part 1 as per sub-graph isomorphism's injectivity rule """

    dqm.offset = n2

    for node in G2.nodes:
        # Looping through all the nodes in G2
        
        # Discrete variable for 'node' i in graph G2, with cases
        # representing the nodes in graph G1

        # In the line below, the first argument is the number of cases that are
        # the nodes in G1 that are options for nodes of G2 to choose from 
        # to be mapped to, since all nodes from G2 have to be mapped to some 
        # node in G1, but all nodes of G1 need not be mapped to a node of G2 
        
        dqm.add_variable(n1, node)
        
        print("Added {} as a discrete variable".format(node))

    """ updated code for H_A part 2 the SGI problem. """
    # Set up the coefficients associated with the constraints that
    # each node in G2 is chosen once.  This represents the H_A
    # component of the energy function.

    for node in G2.nodes:
        dqm.set_linear(node, np.repeat(-1.0, n1))
    for itarget in range(n1):
        for ivar, node1 in enumerate(G2_nodes):
            for node2 in G2_nodes[ivar+1:]:
                dqm.set_quadratic_case(node1, itarget, node2, itarget, 2.0)

    # Set up the coefficients associated with the constraints that
    # selected edges must appear in both graphs, which is the H_B
    # component of the energy function. The penalty coefficient B
    # controls the weight of the H_B relative to H_A in the energy 
    # function. The value was selected by trial and error using simple
    # test problems. 

    B = 2.0

    """ updated code for H_B, with 'list index out of range' issue handled for SGI """
    print("\n\n")
    print("For Mapping from G2 to G1")
    # Commented out the old versoin of the sgi code as it did not work well 
    """
    # For all edges in G2, penalizes mappings to edges not in G1
    for e1 in itertools.combinations(G1.nodes, 2):
        for e2 in G2.edges:
            
            e2_indices = (G2_nodes.index(e2[0]), G2_nodes.index(e2[1]))

            print("e1 of G1 node combinations: {}".format(e1))
            print("e2 of G2 edge: {}".format(e2))
            print("e2_indices: {}".format(e2_indices))
            
            if e1 in G1.edges:
                print("e1 - {} - node-node combination is present in G1, so continue".format(e1))
                print("------------")
            elif e1 not in G1.edges:
                print("e1 - {} - combination is not present in G1 edges, so penalize".format(e1))
                print("e2_indices[0] (u) = {}".format(e2_indices[0]))
                print("e1[1] (u_case) = {}".format(e1[0]))
                print("e2_indices[1] (v) = {}".format((e2_indices[1])))
                print("e1[1] (v_case) = {}".format(e1[1]))
                dqm.set_quadratic_case(e2_indices[0], e1[0], e2_indices[1], e1[1], B)
                dqm.set_quadratic_case(e2_indices[0], e1[1], e2_indices[1], e1[0], B)
                print("------------")
    """
    # this is the better version 
    for e1 in itertools.combinations(G1.nodes, 2):
        for e2 in G2.edges:
            
            e2_indices = (G2_nodes.index(e2[0]), G2_nodes.index(e2[1]))
            e1_indices = (G1_nodes.index(e1[0]), G1_nodes.index(e1[1]))

            # print("e1 of G1 node combinations: {}".format(e1))
            # print("e2 of G2 edge: {}".format(e2))
            # print("e2_indices: {}".format(e2_indices))
            
            if e1 in G1.edges:
                # print("e1 - {} - node-node combination is present in G1, so continue".format(e1))
                # print("------------")
                continue
            
            elif e1 not in G1.edges:
                print("Edge {} of G2 could not be mapped to any edge in G1".format(e2))
                print("e1 - {} - combination is not present in G1 edges, so penalize".format(e1))
                # print("e2[0] (u) = {}".format(e2[0]))
                # print("e1_indices[0] (u_case) = {}".format(e1_indices[0]))
                # print("e2[1] (v) = {}".format((e2[1])))
                # print("e1_indices[1] (v_case) = {}".format(e1_indices[1]))# print("e2_indices[0] (u) = {}".format(e2_indices[0]))
                # print("e1[0] (u_case) = {}".format(e1[0]))
                # print("e2_indices[1] (v) = {}".format((e2_indices[1])))
                # print("e1[1] (v_case) = {}".format(e1[1]))
                dqm.set_quadratic_case(e2[0], e1_indices[0], e2[1], e1_indices[1], B)
                dqm.set_quadratic_case(e2[1], e1_indices[0], e2[0], e1_indices[1], B)
                #dqm.set_quadratic_case(e2_indices[0], e1[0], e2_indices[1], e1[1], B)
                #dqm.set_quadratic_case(e2_indices[1], e1[0], e2_indices[0], e1[1], B)
                print("----------")
            # print("e1 - {} - combination is not present in G1 edges, so penalize".format(e1))
            # print("e2[0] (u) = {}".format(e2[0]))
            # print("e1_indices[0] (u_case) = {}".format(e1_indices[0]))
            # print("e2[1] (v) = {}".format((e2[1])))
            # print("e1_indices[1] (v_case) = {}".format(e1_indices[1]))# print("e2_indices[0] (u) = {}".format(e2_indices[0]))
            # print("e1[0] (u_case) = {}".format(e1[0]))
            # print("e2_indices[1] (v) = {}".format((e2_indices[1])))
            # print("e1[1] (v_case) = {}".format(e1[1]))
            # dqm.set_quadratic_case(e2[0], e1_indices[0], e2[1], e1_indices[1], B)
            # dqm.set_quadratic_case(e2[1], e1_indices[0], e2[0], e1_indices[1], B)
            # dqm.set_quadratic_case(e2_indices[0], e1[0], e2_indices[1], e1[1], B)
            # dqm.set_quadratic_case(e2_indices[1], e1[0], e2_indices[0], e1[1], B)
            # print("----------")            
    return dqm


def plot_graphs_helper_function(G1, G2, node_mapping):
    """ 
    Plot the Larger, Parent Graph G1 with the 
    resultant mappings from G2 -> G1 derived from the annealing process
    drwn on top of G1.  

    The provided mapping specifies how the nodes in Graph G2 correspond
    to the nodes in Graph G1. The nodes in each graph are colored using matcing colors
    based on the specififed mappings. 

    Args:
        G1 (NetworkX Graph) - Target Graph
        G2 (NetworkX Graph) - Graph to Embed
        node_mapping (dict) - Dictionary that defines the correspondence between nodes
                            in graph 1 and nodes in graph 2. The keys are the names of nodes in
                            graph 1, and the values are the names of nodes in graph 2. 
    """
    f, ax = plt.subplots(1, 1, figsize = [10, 4.5])

    color_map = []
    G1_targets = node_mapping.values()

    # inv_mapping = {v: k for k, v in node_mapping.items()}
    # updated_nodes_of_G1 = nx.relabel_nodes(G1, inv_mapping, copy = True)

    print("G1_targets:{}".format(G1_targets))
    for i in G1.nodes:
        if i in G1_targets:
            color_map.append("red")
        else:
            color_map.append("blue")
    print("Color map:{}".format(color_map))
    # pos = nx.spring_layout(updated_nodes_of_G1, seed = 9999)
    # nx.draw(updated_nodes_of_G1, pos, with_labels = True, font_color = 'w', arrows = False, 
    # nodelist = [key for key in node_mapping.keys()], 
    # node_color = "red")

    # nx.draw(updated_nodes_of_G1, pos, with_labels = True, font_color = 'w', arrows = False, 
    # nodelist = list(set(updated_nodes_of_G1.nodes()) - set(node_mapping.keys())))

    """
    nx.draw_networkx(G1, 
                    node_color = color_map, 
                    pos = nx.spring_layout(G1, iterations = 1000),
                    arrows = False, 
                    with_labels = True, 
                    font_color = 'w')
    plt.show()
    """

    """
    f, axes = plt.subplots(1, 2, figsize = [10, 4.5])

    colors = itertools.cycle(mcolors.TABLEAU_COLORS)
    G1_colors = [c for c, i in zip(colors, G1.nodes)]
    G2_targets = [node_mapping[n] for n in G2.nodes]
    G2_colors = [G1_colors[G2_targets.index(n)] for n in G2.nodes] 

    nx.draw(G1, with_labels = True, ax = axes[0], node_color = G1_colors)
    nx.draw(G2, with_labels = True, ax = axes[1], node_color = G2_colors)
    
    return axes    
    """    

def edges_to_graph():
    """
    Function to convert edges into a NetworkX Graph

    Returns: 
        - [G1, G2] - Pair of Target Graph G1 and Graph to Embed G2 in lsit form 
    """
    
    
    """ GRAPH G1 - TARGET GRAPH OPTIONS"""
    # 1. Square wwith four vertices, an edge connecting all vertices to make a square and a diagonal edge that connects two opposite corner vertices
    # edges_1 = [(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)]

    # 2. Pentagon with 5 vertices, an edge connecting al the vertices to make a pentagon, and a horizontal edge between two opposite verices such that there are two distinct visible
    # geometrical shapes - one triangle and one trapezium
    # edges_1 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)]

    # 3. A Graph I found on the internet with 
    # edges_1 = [(0, 1), (0, 2), (0, 3),  (0, 4), (1, 2), (2, 3), (3, 4), (1, 5), (1, 6), (3, 5), (4, 6), (5, 6)]

    # 4. Graph G1
    # edges_1 = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 5)]

    # 5 Graph G1 i got from internet
    edges_1 = [(0, 1),(1, 2),(2, 3),(3, 4),(4, 5),(5, 0),(1, 3),(1, 4),(1, 5),(2, 4)]

    """ GRAPH G2 - GRAPH TO EMBED OPTIONS"""
    # 1. Triangle graph that IS an Isomorphic Sub-Graph of Graph G1 Target Graph
    edges_2 = [(0, 1), (1, 2), (2, 0)]
    
    # 2. Square Graph with four vertices, an edge connecting all vertices to make a square
    # edges_2 = [(0, 1), (1, 2), (2, 3), (3, 0)]

    # 3. Sub graph of Grpah 1 of number 3
    # edges_2 = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (1, 4), (1, 5), (3, 4), (3, 5)]

    # 4. Triangle graph that IS an Isomorphic Sub-Graph of Graph G1 Target Graph
    # edges_2 = [(0, 1), (1, 2), (2, 0), (1, 3)]

    # 5. Graph G2 Sub Graph of G1 of #5
    # edges_2 = [(0, 1),(1, 2),(1, 3),(1, 4),(4, 0)]

    G1 = create_graph_helper_function(edges_1)
    G2 = create_graph_helper_function(edges_2)

    nx.draw(G1, with_labels = True)
    plt.show()
    nx.draw(G2, with_labels = True)
    plt.show()

    return [G1, G2]

# def find_isomorphism(G1, G2):
def find_isomorphism(G_list):
    """  
    Search for isomorpshim between two graphs

    Args:
        G_list - Pair of G1 and G2 in list form G1 Target Graph 
        (NetworkX Graph), G2 Graph to Embed (NetworkX Graph)
    
    Returns:
        [G1, G2, results] - A list which has 
                            First element - G1 graph
                            Second element - G2 graph
                            results - Resultant mappings from 
                            G2 graph to G2 graph

        If no isomorphism is found, returns None. 
        Otherwise, returns dict with keys as nodes
        from graph 1 and values as corresponding nodes from graph 2. 
    """  
    
    G1, G2 = G_list[0], G_list[1] 
    if G1.number_of_nodes() < G2.number_of_nodes():
        return None
    
    dqm = create_dqm_helper_function(G1, G2)
    sampler = LeapHybridDQMSampler()
    sampleset = sampler.sample_dqm(dqm, label = 'Example - Quantum Virtual Network Embedding Algorithm using Sub Graph Isomorphism')

    # Generating and Presenting testing and evaluation data using the separate script
    testing_and_evaluation_for_experiment.get_solver_sampler_information(sampler)
    testing_and_evaluation_for_experiment.get_current_solution_sampleset_evaluation_data(sampleset)

    return G1, G2, sampleset
    """
    best = results.first
    if np.isclose(best.energy, 0.0):
        G2_nodes = list(G2.nodes)
        return {k: G2_nodes[i] for k, i in best.sample.items()}
    else:
        # Isomorphism not found
        return None
    """

def present_results(G1, G2, sampleset):
    """
    Function to present the resultant mappings 
    and plot the graphs

    Args: 
        G1_G2_sampleset - A list which has 
                        First element - G1 graph
                        Second element - G2 graph
                        sampleset - Resultant mappings and other important informationabout the 
                        execution of the annealing process from G2 graph to G1 graph
    """
    # G1, G2 = G1_G2_sampleset[0], G1_G2_sampleset[1]
    # results = G1_G2_sampleset[2]
    results = sampleset
    print("Sub-GI Mappings Results from G2 --> G1 are \n :{}".format(results))
    
    if np.isclose(results.first.energy, 0.0):
        # 'best_mapping' is the top most mapping that has 0.0 energy, from nodes in G2 to nodes in G1
        print("G2 is a Sub-Graph of G1. Yay!")
        best_mapping =  results.first.sample
        print("best_mapping:{}".format(best_mapping))
        print("Type of 'best_mapping':{}",format(type(best_mapping)))
        plot_graphs_helper_function(G1, G2, best_mapping)
    else:
        print("NO, G2 is not a Subgraph of G1")
        best_mapping =  results.first.sample
        print("best_mapping:{}".format(best_mapping))
        print("Type of 'best_mapping':{}",format(type(best_mapping)))
        plot_graphs_helper_function(G1, G2, best_mapping)


if __name__ == "__main__":
    start = time.time()
    # G1, G2 = edges_to_graph()

    present_results(find_isomorphism(edges_to_graph()))
    
    end = time.time()
    print("\nTotal Time taken by Script to run(including the visuals): {}".format(str(end - start)))
    # print("\nTotal Time taken by Script to run(excluding the visuals): {}".format(str(end - start)))

    # axes = plot_graphs(G1, G2, best)
    # axes[0].set_title('Graph G2 (Graph to Embed) mapped onto Graph G1 (Target Graph)')

    # only finding isomorphism between two graphs for now
    """
    results = find_isomorphism(G1, G2)
    if results is None:
        print("No Isomorphism Found")
    else:
        print("The Two Graphs are Isomorphic!")
        for r1, r2 in results.items():
            print("{} --> {}".format(r2, r1))
        
        axes = plot_graphs(G1, G2, results)
        axes[0].set_title('Graph 1 - Target Graph')
        axes[1].set_title('Graph 2 - Graph to Embed')
        
        plt.show()
        filename = 'sub-gi.png'
        plt.savefig(filename, bbox_inches = 'tight')
        print("Plot saved to:", filename)  
        if args.save_plot:
            filename = 'circuit_equivalence.png'
            plt.savefig(filename, bbox_inches='tight')
            print('Plot saved to:', filename)
        """
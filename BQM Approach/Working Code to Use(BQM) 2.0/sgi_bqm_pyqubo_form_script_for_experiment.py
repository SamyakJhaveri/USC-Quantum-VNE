"""
BQM Approach
With improvements using suggestions from ChatGPT 
Final Working Script to be used for Experiments.

Using GSGMorph GitHub Repo's PyQUBO Form library.

References:
https://docs.dwavesys.com/docs/latest/c_solver_properties.html

"""
# Import Stuff
import itertools
import numpy as np  
import networkx as nx  
import dimod
import neal
import sys 
import src.gsgmorph.pyqubo_form as gsgm_pqf
from dwave.system import DWaveSampler, EmbeddingComposite
from pyqubo import Array
import networkx as nx
import warnings
# from .utils import IncompatibleGraphError

# importing `testing_and_evaluation_script_for_experiment.py` script
import testing_and_evaluation_script_for_experiment

# Ignore errors importing matpotlib.pyplot
try:
    import matplotlib.pyplot as plt  
    import matplotlib.colors as mcolors
except ImportError:
    pass

def create_graph(edges):
    """
    Generate Graph 
    """ 
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def find_isomorphism(G1_G2_pair_list):
    """  
    Search for isomorpshim between two graphs

    Args:
        G1 Target Graph (NetworkX Graph)
        G2 Graph to Embed (NetworkX Graph)
    
    Returns:
        If no isomorphism is found, returns None. 
        Otherwise, returns dict with keys as nodes
        from graph 1 and values as corresponding nodes from graph 2. 
    """  

    G1 = G1_G2_pair_list[0]
    G2 = G1_G2_pair_list[1]

    if G1.number_of_nodes() < G2.number_of_nodes():
        return None
    
    # get the pyQUBO expression and translation dictionary which allows us
    # to translate from the annealer results to the actual node mapping
    pyqubo_exp, sample_translation_dictionary = gsgm_pqf.subgraph_isomorphism(G2, G1)
    
    # Initiate Anealer
    qpu_advantage = DWaveSampler(solver={'topology__type': 'pegasus'})
    testing_and_evaluation_script_for_experiment.sampler_information(qpu_advantage)
    sampler = EmbeddingComposite(qpu_advantage)

    # Convert the PyQUBO expression to a BQM that can be fed to the annealer
    model = pyqubo_exp.compile()
    bqm = model.to_bqm()

    
    # Obtain samples from annealing and choose the one with the lowest energy
    # Keep in mind that there may exist multiple satisfactory, low-energy solutions!
    sampleset = sampler.sample(bqm, num_reads=100)
    decoded_samples = model.decode_sampleset(sampleset)
    best_sample=min(decoded_samples, key=lambda x: x.energy)

    # TODO: Find out what does the 'energy' here indicate? Does lower energy mean 
    # that there is a subgraph isomorphism mapping? what does it mean when the `energy` is
    # negative, zero or positive?
    
    return sampleset, sample_translation_dictionary, best_sample     

def plot_graphs(G1_G2_pair_list, result_mapping):
    """ 
    Plot graphs of two circuits

    The provided mapping specific how the nodes in graph1 correspond
    to the nodes in graph 2. The nodes in each graph are colored using matcing colors
    based on the specififed mappings. 

    Args:
        G1 (NetworkX Graph) - Target Graph
        G2 (NetworkX Graph) - Graph to Embed
        node_mapping (dict) - Dictionary that defines the correspondence between nodes
                            in graph 1 and nodes in graph 2. The keys are the names of nodes in
                            graph 1, and the values are the names of nodes in graph 2. 
    
    # We can use NetworkX to help us visualize the subgraph and how it maps to the target graph

    # Invert the mapping so it is from the target graph TO the graph to embed nodes
    # Snippet taken from: 
    # https://stackoverflow.com/a/483833
    inv_mapping = {v: k for k, v in result_mapping.items()}

    # relabel the nodes in the target graph with the node labels from the graph to embed
    updated_nodes = nx.relabel_nodes(G1, inv_mapping, copy=True)
    # Get the same fixed position used before
    pos = nx.spring_layout(updated_nodes, seed=9999)

    # Highlight the nodes that have been relabeled
    nx.draw(updated_nodes, pos, with_labels=True, 
            font_color='w', 
            nodelist=[key for key in result_mapping.keys()], 
            node_color="tab:red")

    # Difference between two lists:
    # https://stackoverflow.com/a/3462160
    nx.draw(updated_nodes, 
            pos, 
            with_labels=True, 
            font_color='w', 
            nodelist=list(set(updated_nodes.nodes()) - set(result_mapping.keys())), 
            node_color="tab:blue")

    plt.show()
    """
    G1 = G1_G2_pair_list[0]
    G2 = G1_G2_pair_list[1]
    
    f, ax = plt.subplots(1, 1, figsize = [10, 4.5])

    color_map = []
    G1_targets = result_mapping.values()

    print("G1_targets:{}".format(G1_targets))
    for i in G1.nodes:
        if i in G1_targets:
            color_map.append("red")
        else:
            color_map.append("blue")
    print("Color Map:{}".format(color_map))
    nx.draw_networkx(G1, 
                    node_color = color_map, 
                    #pos = nx.spring_layout(G1, iterations = 1000),
                    arrows = False, 
                    with_labels = True, 
                    font_color = 'w')
    plt.show()
    """
    colors = itertools.cycle(mcolors.TABLEAU_COLORS)
    G1_colors = [c for c, i in zip(colors, G1.nodes)]
    G2_targets = [node_mapping[n] for n in G1.nodes]
    G2_colors = [G1_colors[G2_targets.index(n)] for n in G2.nodes] 

    nx.draw(G1, with_labels = True, ax = axes[0], node_color = G1_colors)
    nx.draw(G2, with_labels = True, ax = axes[1], node_color = G2_colors)

    return axes        
    """


if __name__ == "__main__":
    """ GRAPH G1 - TARGET GRAPH """
    # 1. Square wwith four vertices, an edge connecting all vertices to make a square and a diagonal edge that 
    # connects two opposite corner vertices
    # edges_1 = [(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)]

    # 2. Pentagon with 5 vertices, an edge connecting al the vertices to make a pentagon, and a horizontal edge between two opposite verices such that there are two distinct visible
    # geometrical shapes - one triangle and one trapezium
    edges_1 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)]


    """ GRAPH G2 - GRAPH TO EMBED """
    # 1. Triangle graph that IS an Isomorphic Sub-Graph of Graph G1 Target Graph
    # edges_2 = [(0, 1), (1, 2), (2, 0)]
    
    # 2. Square Graph with four vertices, an edge connecting all vertices to make a square
    edges_2 = [(0, 1), (1, 2), (2, 3), (3, 0)]

    # 3. 
    # edges_2 = [()]


    G1 = create_graph(edges_1)
    G2 = create_graph(edges_2)

    print("Graph 1 - Target Graph is:")
    nx.draw(G1, with_labels = True)
    plt.show()
    print("Graph 2 - Graph to Embed is:")
    nx.draw(G2, with_labels = True)
    plt.show()

    # only finding isomorphism between two graphs for now
    result_mapping = find_isomorphism(G1, G2)
    if result_mapping is None:
        print("No Isomorphism Found")
    else:
        print("G2 is a subgraph isomorph of G1!")
        print(result_mapping)
        plot_graphs(G1, G2, result_mapping)
        """
        axes = 

        axes[0].set_title('Graph 1 - Target Graph')
        axes[1].set_title('Graph 2 - Graph to Embed')
        

        plt.show()  
        if args.save_plot:
            filename = 'circuit_equivalence.png'
            plt.savefig(filename, bbox_inches='tight')
            print('Plot saved to:', filename)
        """
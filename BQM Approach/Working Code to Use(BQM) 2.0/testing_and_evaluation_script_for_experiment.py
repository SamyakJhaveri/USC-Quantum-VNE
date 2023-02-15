"""
BQM Approach
With improvements using suggestions from ChatGPT 
Final Working Script to be used for Experiments. 

Script that prints (does not return) the infromation about the Solver in use and the meta information about the 
resultant sampleset from the DWave annealing of the problem.

(Optionally, we can set the time limit of running a job on the Solver/Sampler manually depending on our requirement. 
Will not do that actually test the timings but in a later stage could put that in just to play with the code)
Timings to account for 
(Based onthe Paper that compares the performance between 2000q and 5000 qubit Advantge System):
- `run time` is the total elapsed time including system overhead; 
- `charge time` is a subset of run time (omitting overhead) that is charged to the user’s account; and 
- `qpu access time` is the time spent accessing QPU. 

Note that the classical and quantum solver components operate asynchronously in parallel, so the total elapsed 
time does not necessarily equal the sum of component times.

(additional timings that could be looked into from my reading of the DWave documenation (may not be necessary 
and too detailed but lets just write them into the script so that we can use them any time later))
- 

"""

# Import stuff
import itertools
import numpy as np  
import networkx as nx  
import dimod
from dwave.system import LeapHybridDQMSampler 
import dwave.inspector


def get_current_solution_sampleset_evaluation_data(sampleset):
    """
    Function to present Evaluation data of the Current 
    sampleset being processed in the LeapHybridDQMSolver

    Args: 
        - sampleset - the sample of the best results we get from the solver
    """

    print("\nQPU Timing Information from DWave SAPI")
    print("\n-------------------------")

    """
    Information about the time the portfolio solver spent working on the problem: 
    - run_time is the total elapsed time including system overhead; 
    - charge_time is a subset of run time (omitting overhead) that is charged to 
    the user’s account; and 
    - qpu_access_time is the time spent accessing QPU. 
    
    Note that the classical and 
    quantum solver components operate asynchronously in parallel, so the total 
    elapsed time does not necessarily equal the sum of component times.
    """

    # CURRENT SOLUTION SAMPLESET INFORMATION
    print("CURRENT SOLUTION SAMPLESET INFORMATION")

    # Problem Label
    print("\nProblem Label: {}".format(sampleset.info["problem_label"])) 

    # Problem ID 
    print("Problem ID: {}".format(sampleset.info["problem_id"]))

    # Run Time - Total elapsed time including system overhead
    print("Run Time: Time, in microseconds, the hybrid solver spent working on the problem, i.e. Total elapsed time including system overhead:"
        "\n{} microseconds".format(sampleset.info["run_time"]))
    
    # Charge Time - A subset of run time (omitting overhead) that is charged to the user’s account
    print("Charge Time: A subset of run time (omitting overhead) that is charged to the user’s account:"
        "\n{} microseconds".format(sampleset.info["charge_time"]))
    
    # Actual QPU Access Time
    print("QPU Access Time (QPU time, in microseconds, used by the hybrid solver., i.e."
        "\nTime taken by the QPU to execute one QMI (one compute job))"
        "\nduring which the QPU is unavailable for any other QMI. This does not include"
        "\nthe service time and internet latency, only the time the code spends in the QPU" 
        "\nand not even the pre or post-processing parts of the sampler/solver."
        "\nTotal time in QPU):\n{} microseconds".format(sampleset.info["qpu_access_time"]))
    
    print("--------------------")

    """ Not Supported by Hybrid Solvers """
    # QPU Anneal Time Per Sample
    print("QPU Anneal Time Per Sample: Time Taken by QPU to complete one annealing cycle.:"
        "\n{} microseconds".format(sampleset.info["qpu_anneal_time_per_sample"]))

    # QPU Read Out Time Per Sample
    print("QPU Read Out Time Per Sample: Time Taken by QPU for one read.:"
        "\n{} microseconds".format(sampleset.info["qpu_readout_time_per_sample"]))
    
    # QPU Anneal Sampling Time
    print("QPU Anneal Sampling Time: Time Taken by QPU to complete Total time for 𝑅 samples where R is the number of reads(a.k.a samples).:"
        "\n{} microseconds".format(sampleset.info["qpu_sampling_time"]))
    
    # QPU Access Overhead Time 
    print("QPU Access Overhead Time: Initialization time spent in low-level operations (roughly 10 - 20 ms for Advantage Systems):"
        "\n{} microseconds".format(sampleset.info["qpu_access_overhead_time"]))

    # QPU Programming Time
    print("QPU Programming Time: Total time taken to program the QPU:"
        "\n{} microseconds".format(sampleset.info["qpu_programming_time"]))

    # Total Post-Processing Time (For the Post-processing that the Sampler does internally, automatically)
    print("Total Post Processing Time: Total time for post-processing:"
        "\n{} microseconds".format(sampleset.info["total_post_processing_time"]))
    
    # Post Processing Overhead Time post_processing_overhead_time
    print("Post Processing Overhead Time: Extra time needed to process the last batch:"
        "\n{} microseconds".format(sampleset.info["post_processing_overhead_time"]))
    
    
    # print("QPU Timing Information from DWave Cloud Client\n") 


def sampler_information(sampler):
    """
    Function to get information about the sampler / solver

    Args:
        - sampler - Sampler being used
        
    """

    # SOLVER/SAMPLER INFORMATION
    print("SOLVER/SAMPLER INFORMATION")

    # Annealing Time
    print("Annealing Time Range: {}".format(sampler.properties["annealing_time_range"]))

    # Category
    print("Category ['qpu', 'hybrid', 'software']: {}".format(sampler.properties["category"]))
    
    # Chip ID
    print("Chip ID: {}".format(sampler.properties["chip_id"]))

    # Default Annealing Time 
    print("Default Annealing Time: {}".format(sampler.properties["default_annealing_time"]))
    
    # Default Programming Thermalization
    print("Default Programming Thermalization: {}".format(sampler.properties["default_programming_thermalization"]))

    # Default Readout Thermalization
    print("Default Readout Thermalization: {}".format(sampler.properties["default_readout_thermalization"]))
    
    # Number of Qubits
    print("Number of Qubits: {}".format(sampler.properties["num_qubits"]))

    # Number of Reads Range
    print("Number of Reads Range: {}".format(sampler.properties["num_reads_range"]))

    # Paramters
    # print("Parameters: {}".format(sampler.properties["parameters"]))

    # Problem Timing Data
    print("Problem Timing Data: {}".format(sampler.properties["problem_timing_data"]))

    # Topology
    print("Topology: {}".format(sampler.properties["topology"]))


    print("--------------------")

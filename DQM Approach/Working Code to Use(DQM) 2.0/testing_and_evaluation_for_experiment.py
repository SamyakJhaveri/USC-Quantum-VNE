"""
DQM Approach
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

    # CURRENT SOLUTION SAMPLESET INFORMATION
    print("CURRENT SOLUTION SAMPLESET INFORMATION")

    # Problem Label
    print("\nProblem Label: {}".format(sampleset.info["problem_label"])) 

    # Problem ID 
    print("\nProblem ID: {}".format(sampleset.info["problem_id"]))

    # Run Time - Total elapsed time including system overhead
    print("\nRun Time: Time, in microseconds, the hybrid solver spent working on the problem, i.e. Total elapsed time including system overhead:"
        "\n{} microseconds".format(sampleset.info["run_time"]))
    
    # Charge Time - A subset of run time (omitting overhead) that is charged to the user’s account
    print("\nCharge Time: A subset of run time (omitting overhead) that is charged to the user’s account:"
        "\n{} microseconds".format(sampleset.info["charge_time"]))
    
    # Actual QPU Access Time
    print("\nQPU Access Time (QPU time, in microseconds, used by the hybrid solver., i.e."
        "\nTime taken by the QPU to execute one QMI (one compute job))"
        "\nduring which the QPU is unavailable for any other QMI. This does not include"
        "\nthe service time and internet latency, only the time the code spends in the QPU" 
        "\nand not even the pre or post-processing parts of the sampler/solver."
        "\nTotal time in QPU):\n{} microseconds".format(sampleset.info["qpu_access_time"]))
    
    print("--------------------")

    """ Not Supported by Hybrid Solvers
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
    
    """


def get_solver_sampler_information(sampler):
    """
    Function to get information about the sampler / solver

    Args:
        - sampler - Sampler being used
        
    """

    # SOLVER/SAMPLER INFORMATION
    print("SOLVER/SAMPLER INFORMATION")
    print("Leap Hybrid DQM Sampler Properties are:{}".format(sampler.properties))
    print("Leap Hybrid DQM Sampler Solver Parameters are:{}".format(sampler.parameters))
    # Category
    print("Category ['qpu', 'hybrid', 'software']: {}".format(sampler.properties["category"]))

    # Maximum Number of Biases
    print("Maximum number of biases, both linear and quadratic in total, accepted by the solver: {}".format(sampler.properties["maximum_number_of_biases"]))

    # Maximum Number of Variables
    print("Maximum number of problem variables accepted by the solver: {}".format(sampler.properties["maximum_number_of_variables"]))

    # Maximum Time limit in Hours
    print("Maximum allowed run Time in Hours that can be specified for the solver: {}".format(sampler.properties["maximum_time_limit_hrs"]))

    # Minimum Time Limit
    print("Minimum required run time, in seconds, the solver must be allowed to work on the given problem." 
    "\nSpecifies the minimum time as a piecewise-linear curve defined by a set of floating-point pairs." 
    "\nThe second element is the minimum required time; the first element in each pair is some measure of the" 
    "\nproblem, dependent on the solver:"
        "\n\t- For hybrid BQM solvers, this is the number of variables."
        "\n\t- For hybrid DQM solvers, this is a combination of the numbers of interactions, variables," 
        "\nand cases that reflects the “density” of connectivity between the problem’s variables."
    "\nThe minimum time for any particular problem is a linear interpolation calculated on two pairs that represent" 
    "\nthe relevant range for the given measure of the problem. For example, if minimum_time_limit for a hybrid BQM" 
    "\nsolver were [[1, 0.1], [100, 10.0], [1000, 20.0]], then the minimum time for a 50-variable problem would be" 
    "\n5 seconds, the linear interpolation of the first two pairs that represent problems with between 1 to 100" 
    "\nvariables.")
    print("\nMinimum Time limit:{}".format(sampler.properties["minimum_time_limit"]))
    
    # Paramters
    print("Parameters: {}".format(sampler.properties["parameters"]))

    # Quota Conversion Rate 
    print("\nRate at which user or project quota is consumed for the solver as a ratio to QPU solver usage." 
    "\nDifferent solver types may consume quota at different rates.")
    print("\nQuota Conversion Rate:{}".format(sampler.properties["quota_conversion_rate"]))
    
    # Version 
    print("Version number of solver/sampler: {}".format(sampler.properties["version"]))

    print("--------------------")

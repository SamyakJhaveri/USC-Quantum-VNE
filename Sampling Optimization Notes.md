**MATLAB Code**:
function [ RelativeEntropy,BestSolutions,BestObjective,BestSolutionFound,BestObjectiveFound,EmbeddingData] = SEBREMforQUBO( Qfull,beta,Niterations,EmbeddingFlag,Step)
% SEBREMforQUBO Summary of this function goes here
%   Qfull is the QUBO matrix of the problem we want to embed





% Parameters


MAX_N_SOLUTIONS = 40000;

Nqubits = 512;

Nvariables = size(Qfull,1);

params.num_reads = 10000; % Number of reads per run
params.auto_scale = false;
%params.annealing_time = 100;
absJmax = 1;
abshmax = 2;

SampleScaleFactor = 0.99;


RelativeEntropy = zeros(1,Niterations);
BestObjective = zeros(1,Niterations);
%BestSolution = zeros(Nvariables,Niterations);



% Obtain adjacency matrix and list of qubits used




Aff = 0; % Affiliate connection flag

% Choose a connection depending on the time of day

clk = clock;

if clk(4) == 4 || clk(4) == 9 || clk(4) == 14 || clk(4) == 19 
    
    % Creates a remote SAPI connection handle to the affiliate server
    conn = sapiRemoteConnection('https://206.117.25.108/sapi/','92cc484c78372afc3810749cf106e95c76540571');
    Aff =1;
else
    % Creates a remote SAPI connection handle to the ISI server
    conn = sapiRemoteConnection('https://206.117.25.107/sapi/','d6d46de897c148729afb655909aff78d0eb125ea');
end

% Create a SAPI solver handle

if Aff==1
    solver = sapiSolver(conn,'V6-LP'); % Hardware Low Priority (affiliate)
else
    solver = sapiSolver(conn,'V6-MP'); % Hardware Medium Priority
end

AdjacencyMatrix = getHardwareAdjacency(solver);

QubitsUsed = GenerateQubitsUsed(AdjacencyMatrix);

% Shift QubitsUsed to use a different part of the chip

% QubitsUsed = QubitsUsed - 128; % Shifts everything 2 cells up

----------
**ChatGPT Response**:

This function appears to be implementing a method called "SEBREM" (Short Embedding using Relative Entropy Maximization) for solving a problem represented by a QUBO (Quadratic Unconstrained Binary Optimization) matrix, using a quantum annealer. 
The function takes several input parameters including the 
- QUBO matrix (Qfull), 
- a temperature parameter (beta), 
- the number of iterations to run the algorithm (Niterations), 
- a flag to indicate whether to perform an embedding step (EmbeddingFlag), and 
- a step size (Step). 

The function also has several internal parameters such as the 
- number of qubits to use (Nqubits), 
- the maximum number of solutions to return (MAX_N_SOLUTIONS),
- and the number of reads per run (params.num_reads). 

The function also creates a connection to a remote quantum annealer using the SAPI (Solver API) library and generates a list of qubits used based on the hardware's adjacency matrix. The function appears to perform several steps including calculating relative entropy, finding the best solutions and objectives, and potentially an embedding step. The function also appears to have an if-else block that changes the quantum annealer used based on the current time.

It is not possible to convert this code into python without more information about the implementation of the SEBREM algorithm, the GenerateQubitsUsed() and getHardwareAdjacency() functions, and the sapiRemoteConnection and sapiSolver classes. Additionally, the code is written in MATLAB, which is a different programming language from Python, so a direct translation is not possible. However, I can give you an idea of how the code could be implemented in python.

import numpy as np

def SEBREMforQUBO(Qfull, beta, Niterations, EmbeddingFlag, Step):
    MAX_N_SOLUTIONS = 40000
    Nqubits = 512
    Nvariables = Qfull.shape[0]
    params = {'num_reads': 10000, 'auto_scale': False}
    absJmax = 1
    abshmax = 2
    SampleScaleFactor = 0.99

    RelativeEntropy = np.zeros(Niterations)
    BestObjective = np.zeros(Niterations)

    Aff = 0
    # Choose a connection depending on the time of day
    import datetime
    now = datetime.datetime.now()
    if now.hour in [4, 9, 14, 19]:
        # Creates a remote SAPI connection handle to the affiliate server
        conn = sapiRemoteConnection('https://206.117.25.108/sapi/','92cc484c78372afc3810749cf106e95c76540571')
        Aff =1
    else:
        # Creates a remote SAPI connection handle to the ISI server
        conn = sapiRemoteConnection('https://206.117.25.107/sapi/','d6d46de897c148729afb655909aff78d0eb125ea')

    # Create a SAPI solver handle
    if Aff==1:
        solver = sapiSolver(conn,'V6-LP') # Hardware Low Priority (affiliate)
    else:
        solver = sapiSolver(conn,'V6-MP') # Hardware Medium Priority

    AdjacencyMatrix = getHardwareAdjacency(solver)
    QubitsUsed = GenerateQubitsUsed(AdjacencyMatrix)

    # Shift QubitsUsed to use a different part of the chip
    # QubitsUsed = QubitsUsed - 128; # Shifts everything 2 cells up
    
    # the rest of the algorithm

Please be aware that this is just a rough idea of how the code could be implemented in python, as the specific implementations of sapiRemoteConnection, sapiSolver, getHardwareAdjacency, and GenerateQubitsUsed are not available. Also, you would need to implement the rest of the algorithm and the missing functions.


---------- x ---------- x ---------- x ---------- x ---------- x ----------

% Generate and initial embedding for the QUBO

[QubitMapping,couplers,VariableInteraction,h_chimera,J_chimera,hfull,Jfull] = InitialEmbedding(Qfull,EmbeddingFlag,AdjacencyMatrix,QubitsUsed);


EmbeddingData.Qmap = QubitMapping;
EmbeddingData.couplers = couplers;

Python:
QubitMapping, couplers, VariableInteraction, h_chimera, J_chimera, hfull, Jfull = InitialEmbedding(Qfull, EmbeddingFlag, AdjacencyMatrix, QubitsUsed)

EmbeddingData = {}
EmbeddingData['Qmap'] = QubitMapping
EmbeddingData['couplers'] = couplers


Response:
Here, InitialEmbedding(Qfull, EmbeddingFlag, AdjacencyMatrix, QubitsUsed) is a function that takes the QUBO matrix, a flag indicating whether to perform an embedding, the adjacency matrix, and the list of qubits used as input and returns multiple outputs, which are unpacked and assigned to the corresponding variables. Also, in python, we can use dictionary to store the data instead of struct.

---------- x ---------- x ---------- x ---------- x ---------- x ----------
% Define the function that computes the Ising energy on spin configurations

IsingEnergy = @(S) (S'*Jfull*S + S'*hfull);

Python: 
import numpy as np
from functools import partial

IsingEnergy = partial(np.dot, np.dot(Jfull, np.transpose(S)) + np.dot(np.transpose(S), hfull))

def IsingEnergy(S):
    return np.dot(np.dot(Jfull, np.transpose(S)) + np.dot(np.transpose(S), hfull))

Response:
Here, partial(np.dot, np.dot(Jfull, np.transpose(S)) + np.dot(np.transpose(S), hfull)) is a function that takes a variable S and returns the dot product of S'*Jfull*S + S'*hfull, where S is a spin configuration.


---------- x ---------- x ---------- x ---------- x ---------- x ----------


% Start the iterative procedure

for iter=1:Niterations
    
    
    
    % Solve the Ising model with quantum processor 
    
    [ answer] = IsingConnectSolve(SampleScaleFactor*h_chimera,SampleScaleFactor*J_chimera,params);
    
    %[ answer] = GaugedIsingConnectSolve(SampleScaleFactor*h_chimera,SampleScaleFactor*J_chimera,params,10);
    
    NumberOfSamples = size(answer.solutions,2);
    
    
    % Extract a distribution from the samples
    
    [SpinSolutions,ProbabilityOfSamples,EnergiesOfSamples{iter}] = ExtractDistribution(answer,QubitMapping);
    
    
    % Compute objective function on samples
    
    Gvec = GvecComputation(SpinSolutions,IsingEnergy); % Compute the values of the normalized original function on the samples
    G{iter} = Gvec;    
    
    
    % Extract best solutions
    
    [BestObjective(iter),BestSolutions{iter}] = FindBestSolution(SpinSolutions,Gvec,Qfull);
    
    
    
    
    % Compute the relative entropy 
    
    RelativeEntropy(iter) = -ProbabilityOfSamples*log(ProbabilityOfSamples') + beta*(Gvec*ProbabilityOfSamples');
    
    
    % Compute the gradient of the relative entropy
    
    [GradRE_h,GradRE_J] = REGradient(SpinSolutions,ProbabilityOfSamples,Gvec,beta,VariableInteraction);
    Grad(iter,:) = [GradRE_h GradRE_J];
    if iter > 1
        GradInnerProduct = Grad(iter,:)*Grad(iter-1,:)/(norm(Grad(iter-1,:))*norm(Grad(iter,:)));
    else
        GradInnerProduct = 0;
    end
    
    
    % Update Ising model
    
    if GradInnerProduct < -0.1
        Step = max(0.005,Step/3);
    elseif GradInnerProduct > 0.5
        Step = min(0.1,Step*3);    
    end
    
    
    [h_chimera,J_chimera] = UpdateIsingModel( h_chimera,J_chimera,GradRE_h,GradRE_J,Step,QubitMapping,couplers,BestObjective,BestSolutions,iter);
    
    
    
    h{iter} = h_chimera;
    J{iter} = J_chimera;
    num_samples{iter} = NumberOfSamples;
    step{iter} = Step;
    
    
    
    
    
    
    % Start printing information
    
    DisplayInformation(RelativeEntropy,iter,BestObjective,BestSolutions,beta,Niterations,GradInnerProduct,NumberOfSamples);
    
    
    % Save data
    
    save SEBREMdata BestObjective BestSolutions G EnergiesOfSamples Grad RelativeEntropy h J ;
    
    EmbeddingData.h = h;
    EmbeddingData.J = J;
    EmbeddingData.num_samples = num_samples;
    EmbeddingData.step = step;
    
    
end


Python: 
for iter in range(Niterations):

    # Solve the Ising model with quantum processor
    answer = IsingConnectSolve(SampleScaleFactor*h_chimera, SampleScaleFactor*J_chimera, params)

    # NumberOfSamples = size(answer.solutions,2);
    NumberOfSamples = answer.solutions.shape[1]

    # Extract a distribution from the samples
    SpinSolutions, ProbabilityOfSamples, EnergiesOfSamples[iter] = ExtractDistribution(answer, QubitMapping)

    # Compute objective function on samples
    Gvec = GvecComputation(SpinSolutions, IsingEnergy)  # Compute the values of the normalized original function on the samples
    G[iter] = Gvec

    # Extract best solutions
    BestObjective[iter], BestSolutions[iter] = FindBestSolution(SpinSolutions, Gvec, Qfull)

    # Compute the relative entropy
    RelativeEntropy[iter] = -ProbabilityOfSamples * np.log(ProbabilityOfSamples) + beta * (Gvec * ProbabilityOfSamples)

    # Compute the gradient of the relative entropy
    GradRE_h, GradRE_J = REGradient(SpinSolutions, ProbabilityOfSamples, Gvec, beta, VariableInteraction)
    Grad[iter, :] = [GradRE_h, GradRE_J]
    if iter > 1:
        GradInnerProduct = Grad[iter, :] * Grad[iter - 1, :] / (np.linalg.norm(Grad[iter - 1, :]) * np.linalg.norm(Grad[iter, :]))
    else:
        GradInnerProduct = 0

    # Update Ising model
    if GradInnerProduct < -0.1:
        Step = max(0.005, Step / 3)
    elif GradInnerProduct > 0.5:
        Step = min(0.1, Step * 3)

    h_chimera, J_chimera = UpdateIsingModel(h_chimera, J_chimera, GradRE_h, GradRE_J, Step, QubitMapping, couplers, BestObjective, BestSolutions, iter)

    h[iter] = h_chimera
    J[iter] = J_chimera
    num_samples[iter] = NumberOfSamples
    step[iter] = Step

    # Start printing information
    DisplayInformation(RelativeEntropy, iter, BestObjective, BestSolutions, beta, Niterations, GradInnerProduct, NumberOfSamples)

    # Save data
    np.savez("SEBREMdata", BestObjective=BestObjective, BestSolutions=BestSolutions, G=G, EnergiesOfSamples=EnergiesOfSamples, Grad=Grad, RelativeEntropy=RelativeEntropy, h=h, J=J)

    EmbeddingData['h'] = h
    EmbeddingData['J'] = J
    EmbeddingData['num_samples'] = num_samples
    EmbeddingData['step'] = step
# Extract best solution
bestindex = np.argmin(BestObjective)
BestObjectiveFound = BestObjective[bestindex]
BestSolutionFound = BestSolutions[bestindex][:, 0]


Response: 
Here, np.argmin(BestObjective) returns the index of the minimum element of the BestObjective array, and BestObjective[bestindex] returns the minimum element of the BestObjective array. BestSolutions[bestindex][:, 0] returns the first column of the best solution stored in BestSolutions at the index of the minimum element of the BestObjective array.

---------- x ---------- x ---------- x ---------- x ---------- x ----------

function [ QubitMapping,couplers,VariableInteraction,h_chimera,J_chimera,hfull,Jfull] = InitialEmbedding( Qfull,EmbeddingFlag,AdjacencyMatrix,QubitsUsed )
%INITIALEMBEDDING Provides the starting embedding for SEBREM
%   It takes as input the full matrix of a QUBO, converts it to upper
%   diagonal (it assumes it is symmetric and discards the lower triangle),
%   converts it to an Ising model (h and J), and normalizes it. Then uses
%   some embedding heuristic controlled by EmbeddingFlag:
%   EmbeddingFlag = 1: Greedy embedding
%   EmbeddingFlag = 2: Stochastic greedy embedding
%   EmbeddingFlag = 3: Randomized direct embedding
%                                                                         
%   As output if provides the mapping between variables and qubits
%   (QubitMapping), a matrix with all the pairs of qubits assigned to
%   variables that share a coupler (couplers), and the same information but
%   written in terms of the variables those qubits are assigned. It also
%   provides the embedding as h_chimera and J_chimera, and the normalized
%   ising model as hfull and Jfull



% Parameters

Nqubits = 512;
absJmax = 1;
abshmax = 2;


StartingQubit = 217; % First qubit of one of the central cells

Python: 
def InitialEmbedding(Qfull, EmbeddingFlag, AdjacencyMatrix, QubitsUsed):
    # Parameters
    Nqubits = 512
    absJmax = 1
    abshmax = 2

    StartingQubit = 217  # First qubit of one of the central cells

    # Code for InitialEmbedding function
    # ...

    return QubitMapping, couplers, VariableInteraction, h_chimera, J_chimera, hfull, Jfull

Response: 
It takes the same inputs as the MATLAB function, Qfull, EmbeddingFlag, AdjacencyMatrix and QubitsUsed, and returns the same outputs, QubitMapping, couplers, VariableInteraction, h_chimera, J_chimera, hfull, Jfull in the same order. The function body should include the code for performing the embedding based on the value of the EmbeddingFlag input as described in the comments of the MATLAB function.

---------- x ---------- x ---------- x ---------- x ---------- x ----------

% Convert Qfull to upper triangular

Qfull = triu(Qfull) + triu(Qfull',1);

% Convert to Ising model to allow proper normalization

[hfull, Jfull,~] = quboToIsing(Qfull); % Now Jfull is also upper triangular

% Normalize 

NormFactor = max([(1/abshmax)*max(abs(hfull)) (1/absJmax)*max(abs(Jfull))]); 

hfull = hfull/NormFactor;
Jfull = Jfull/NormFactor;

% Construct initial embedding

if EmbeddingFlag == 1  % Apply the greedy embedding algorithm to find a starting embedding

    
    [ QubitMapping,AdjMat ] = GreedyEmbedding( Jfull,AdjacencyMatrix,StartingQubit );

    
    
elseif EmbeddingFlag == 2  % Apply stochastic version of greedy embedding

    
    [ QubitMapping,AdjMat ] = StochasticGreedyEmbedding( Jfull,AdjacencyMatrix,StartingQubit );
    

else
    [ QubitMapping,AdjMat ] = RandomizedDirectEmbedding( Jfull,AdjacencyMatrix,QubitsUsed );
end




Python: 
# Convert Qfull to upper triangular
Qfull = np.triu(Qfull) + np.triu(Qfull.T, 1)

# Convert to Ising model to allow proper normalization
hfull, Jfull, _ = quboToIsing(Qfull)  # Now Jfull is also upper triangular

# Normalize
NormFactor = max([(1/abshmax)*max(abs(hfull)) (1/absJmax)*max(abs(Jfull))])
hfull = hfull/NormFactor
Jfull = Jfull/NormFactor

# Construct initial embedding
if EmbeddingFlag == 1:  # Apply the greedy embedding algorithm to find a starting embedding
    QubitMapping, AdjMat = GreedyEmbedding(Jfull, AdjacencyMatrix, StartingQubit)

elif EmbeddingFlag == 2:  # Apply stochastic version of greedy embedding
    QubitMapping, AdjMat = StochasticGreedyEmbedding(Jfull, AdjacencyMatrix, StartingQubit)

else:
    QubitMapping, AdjMat = RandomizedDirectEmbedding(Jfull, AdjacencyMatrix, QubitsUsed)


---------- x ---------- x ---------- x ---------- x ---------- x ----------

% Embedded J: AdjMat is only an adjacency matrix 
% J_chimera should be upper triangular (because Jfull is)


J_chimera = zeros(Nqubits);
h_chimera = zeros(Nqubits,1);

J_chimera(QubitMapping(:,2)',QubitMapping(:,2)') = Jfull.*AdjMat; 
J_chimera = triu(J_chimera); % Keep J_chimera upper triangular

h_chimera(QubitMapping(:,2)') = hfull; % Assign local fields

% Check if there is a zero row/column in J_chimera for which
% h_chimera is also zero, and add a very small local field if that is the
% case

ZeroDetect = sum(abs(full([h_chimera(QubitMapping(:,2)') (sum(Jfull.*AdjMat + (Jfull.*AdjMat)'))'])),2);
if ~isempty(find(ZeroDetect == 0,1))
    h_chimera(QubitMapping(find(ZeroDetect == 0),2)) = 0.001;
end

% Inverse QubitMapping
InverseQubitMapping = zeros(Nqubits,1);
InverseQubitMapping(QubitMapping(:,2)) = QubitMapping(:,1);

% Find all the available couplers between the qubits in Qchimera

Reduced_Adjacency_matrix = zeros(Nqubits);

Reduced_Adjacency_matrix(QubitMapping(:,2)',QubitMapping(:,2)') = AdjacencyMatrix(QubitMapping(:,2)',QubitMapping(:,2)');

MaxCouplers = 0;

Python: 
# Embedded J: AdjMat is only an adjacency matrix
# J_chimera should be upper triangular (because Jfull is)

J_chimera = np.zeros((Nqubits, Nqubits))
h_chimera = np.zeros(Nqubits)

J_chimera[np.ix_(QubitMapping[:, 2], QubitMapping[:, 2])] = Jfull * AdjMat
J_chimera = np.triu(J_chimera)  # Keep J_chimera upper triangular

h_chimera[QubitMapping[:, 2]] = hfull  # Assign local fields

# Check if there is a zero row/column in J_chimera for which
# h_chimera is also zero, and add a very small local field if that is the
# case

ZeroDetect = np.sum(np.abs(np.concatenate((h_chimera[QubitMapping[:, 2]].reshape(-1, 1), (np.sum(Jfull * AdjMat + (Jfull * AdjMat).T, axis=0)).reshape(-1

---------- x ---------- x ---------- x ---------- x ---------- x ----------
for i=1:Nqubits
    nonzero_couplers = find(Reduced_Adjacency_matrix(i,1:Nqubits) == 1);
    
    
    for j=1:length(nonzero_couplers)
        if i < nonzero_couplers(j)
            MaxCouplers = MaxCouplers+1;
            couplers(MaxCouplers,1:2) = [i nonzero_couplers(j)]; % matrix with all the couplers between the qubits in QubitMaping
            VariableInteraction(MaxCouplers,1:2) = [InverseQubitMapping(i) ...
                InverseQubitMapping(nonzero_couplers(j))]; % Same as couplers but in terms of the original variables
        end                                             % These couplers respect the upper triangular index ordering (couplers(ct,1) < couplers(ct,2))
        
                                                        
    end
end

couplers = couplers(1:MaxCouplers,:); % Keep only the assigned ones
VariableInteraction = VariableInteraction(1:MaxCouplers,:); % Keep only the assigned ones

end

Python: 

couplers = []
VariableInteraction = []
for i in range(Nqubits):
    nonzero_couplers = np.where(Reduced_Adjacency_matrix[i, :Nqubits] == 1)[0]
    for j in range(len(nonzero_couplers)):
        if i < nonzero_couplers[j]:
            MaxCouplers += 1
            couplers.append([i, nonzero_couplers[j]])
            VariableInteraction.append([InverseQubitMapping[i], InverseQubitMapping[nonzero_couplers[j]]])

couplers = np.array(couplers)
VariableInteraction = np.array(VariableInteraction)

---------- x ---------- x ---------- x ---------- x ---------- x ----------
%%
function [SpinSolutions,ProbabilityOfSamples,EnergiesOfSamples] = ExtractDistribution(answer,QubitMapping)
%EXTRACTDISTRIBUTION Summary of this function goes here
%   Detailed explanation goes here

SpinSolutions = answer.solutions(QubitMapping(:,2),:); % Keep only the values of the qubits associated with problem variables
ProbabilityOfSamples = answer.num_occurrences/sum(answer.num_occurrences); % Estimate probability of each sample by its frequency 
EnergiesOfSamples = answer.energies;


end

Python: 
def ExtractDistribution(answer, QubitMapping):
    SpinSolutions = answer["solutions"][QubitMapping[:, 2], :]
    ProbabilityOfSamples = answer["num_occurrences"] / sum(answer["num_occurrences"])
    EnergiesOfSamples = answer["energies"]
    return SpinSolutions, ProbabilityOfSamples, EnergiesOfSamples

Response: 
The above function will take in two inputs:

answer which is a dictionary that contains the solutions, number of occurrences, and energies of the samples.
QubitMapping which is a 2-column matrix that maps the qubits to the problem variables.
It will return three outputs:

SpinSolutions which is a matrix where each column represents a sample and the rows are the values of the qubits associated with the problem variables.
ProbabilityOfSamples which is a vector with the probability of each sample.
EnergiesOfSamples which is a vector with the energies of each sample.

---------- x ---------- x ---------- x ---------- x ---------- x ----------

%%
function [ QubitMapping,AdjMat ] = GreedyEmbedding( J,AdjacencyMatrix,StartingQubit )
%GREEDYEMBEDDING Generates an embedding that prioritizes keeping the
%strongest connections of J.

% Inputs:
%
%  J  : Interaction matrix to be approximated
%  AdjacencyMatrix : Adjacency matrix of the chip
%  StartingQubit : Initial qubit to start the embedding (picking one with 
%                  high connectivity and somwhere around the center of the
%                  chip recommended)
%
%
% Outputs: 
%
%  QuibtMapping : N x 2 matrix, first column is a list of variables of
%  the quadratic function, second column has the qubits assigned to those
%  variables
%  AdjMat : reduced adjacency matrix representing the connections between
%  the qubits chosen from the embedding


J = J - diag(diag(J));

Nvars = size(J,1);
Nqubits = 512;  % Total number of qubits in Vesuvius (used for indexing)    
Qubit = StartingQubit; 


MissingQubits = find((sum(AdjacencyMatrix,1)) == 0); % Missing qubits in processor

%Remove Missing qubits from AdjacencyMatrix (in case they're still there)
for i=1:length(MissingQubits)
    AdjacencyMatrix(MissingQubits(i),:) = 0*AdjacencyMatrix(MissingQubits(i),:);
    AdjacencyMatrix(:,MissingQubits(i)) = 0*AdjacencyMatrix(:,MissingQubits(i));
end




% MissingQubits = [];



[~,StrongCouplingIndex] = max(sum(abs(J))); % Find the index with the strongest coupling


VariablesLeft = 1:Nvars;
VariablesAssigned = [];

QubitsAssigned = [];
QubitsLeft = 1:Nqubits;
QubitsLeft(MissingQubits) = [];



 

VariablesAssigned = StrongCouplingIndex;
VariablesLeft(StrongCouplingIndex) = []; % Remove the assigned variable from list of variables
J(:,StrongCouplingIndex) = zeros(Nvars,1);


QubitsAssigned = [QubitsAssigned Qubit];

QubitNeighbors = find(AdjacencyMatrix(Qubit,:) ~= 0); % Qubits adjacent to first assigned qubit



for n=2:Nvars
    
    for i=1:length(VariablesLeft)
        for j=1:length(QubitNeighbors)
            Coupling = 0;
            for k=1:length(VariablesAssigned)
                Coupling = Coupling + abs(J(VariablesAssigned(k),VariablesLeft(i)))*...
                    AdjacencyMatrix(QubitNeighbors(j),QubitsAssigned(k));
            end
            
            CouplingStrength(i,j) = Coupling;
            
        end
    end
    
    
    
    [MaxCouplingStrength,NewQubitIndex] = max(max(CouplingStrength));
    %%%
    if size(CouplingStrength,1)> 1
        PossibleNewQubitIndices =  find(max(CouplingStrength) == MaxCouplingStrength); 
        % Find all indices that give maximum coupling 
    else
        PossibleNewQubitIndices =  find(CouplingStrength == MaxCouplingStrength); % For the case CS is a row vector
    end
    
    for nqi=1:length(PossibleNewQubitIndices)
        DistanceToLastAssignedQubit(nqi) = abs(QubitsAssigned(end)-QubitNeighbors(PossibleNewQubitIndices(nqi)));
    end
    [~,MinIndex] = min(DistanceToLastAssignedQubit); % Find the qubit closest to last assigned qubit
    clear DistanceToLastAssignedQubit;
    
    NewQubit = QubitNeighbors(PossibleNewQubitIndices(MinIndex));
    
    %%%
    %NewQubit = QubitNeighbors(NewQubitIndex);
    
    [~,NewVarIndex] = max(CouplingStrength(:,NewQubitIndex));
    NewVar = VariablesLeft(NewVarIndex);
    
    clear CouplingStrength;
    
    VariablesAssigned = [VariablesAssigned NewVar];
    VariablesLeft(find(VariablesLeft == NewVar)) = []; 
    
    QubitsAssigned = [QubitsAssigned NewQubit];
    
    QubitNeighbors = [];
    for k=1:length(QubitsAssigned)
        
        SingleQubitNeighbors = find(AdjacencyMatrix(QubitsAssigned(k),:) ~= 0);
        
        QubitNeighbors = unique([QubitNeighbors SingleQubitNeighbors]);
    end
    
    QubitNeighbors = setdiff(QubitNeighbors,QubitsAssigned);
    
       
        
    
    
end
    
QubitMapping = [VariablesAssigned' QubitsAssigned'];

QubitMapping = sortrows(QubitMapping,1);

AdjMat =  AdjacencyMatrix(QubitMapping(:,2)',QubitMapping(:,2)');







end




Python: 

def GreedyEmbedding(J, AdjacencyMatrix, StartingQubit):
    J = J - np.diag(np.diag(J))
    Nvars = J.shape[0]
    Nqubits = 512
    Qubit = StartingQubit
    MissingQubits = np.where(np.sum(AdjacencyMatrix, axis=1) == 0)[0]

    #Remove Missing qubits from AdjacencyMatrix (in case they're still there)
    for i in range(len(MissingQubits)):
        AdjacencyMatrix[MissingQubits[i],:] = 0*AdjacencyMatrix[MissingQubits[i],:]
        AdjacencyMatrix[:,MissingQubits[i]] = 0*AdjacencyMatrix[:,MissingQubits[i]]

    StrongCouplingIndex = np.argmax(np.sum(np.abs(J)))
    VariablesLeft = np.arange(Nvars)
    VariablesAssigned = []
    QubitsAssigned = []
    QubitsLeft = np.arange(Nqubits)
    QubitsLeft = np.delete(QubitsLeft,MissingQubits)

    VariablesAssigned = StrongCouplingIndex
    VariablesLeft = np.delete(VariablesLeft, StrongCouplingIndex)
   

---------- x ---------- x ---------- x ---------- x ---------- x ----------
function [ Gvec] = GvecComputation( SpinSolutions, G )
%GVECCOMPUTATION Summary of this function goes here
%   Detailed explanation goes here


CellSpinSolutions = mat2cell(SpinSolutions,size(SpinSolutions,1),ones(1,size(SpinSolutions,2)));
G_CellSpinSolutions = cellfun(G,CellSpinSolutions,'UniformOutput',false);
Gvec = cell2mat(G_CellSpinSolutions);



end


Python:
import numpy as np
def GvecComputation(SpinSolutions, G):
    Gvec = np.apply_along_axis(G, 1, SpinSolutions)
    return Gvec

Response: 
The function GvecComputation takes two inputs:

SpinSolutions: a 2D matrix where each row represents a spin solution.
G: a function that takes an array of spin solutions as its input and returns an array of values after applying the function on each element of the input array.
The function converts each row of the input matrix SpinSolutions into a cell and applies the function G on each cell. Then it converts the resultant cells into a matrix and returns the matrix as Gvec.

This code snippet uses numpy.apply_along_axis() function which applies the given function along the given axis of the input array. It is similar to the Matlab's cellfun() function.

---------- x ---------- x ---------- x ---------- x ---------- x ----------



function [ distance ] = HammingDistance( String1,String2 )
%HAMMINGDISTANCE Summary of this function goes here
%   Detailed explanation goes here

distance = sum(mod(String1+String2,2));

 end


Python: 
def HammingDistance(String1, String2):
distance = sum(np.mod(String1 + String2, 2))
return distance

---------- x ---------- x ---------- x ---------- x ---------- x ----------
function [GradRE_h,GradRE_J] = REGradient(SpinSolutions,ProbabilityOfSamples,Gvec,beta,VariableInteraction)
%RECOMPUTATION COmputes the relative entropy and its gradient
%   It takes the structure answer as input and computes the relative
%   entropy between the input QUBO and the embedded one. Note that internally 
%   the code works in the Ising framework, so solutions are strings of
%   {+1,-1}. 



Nvariables = size(SpinSolutions,1);

% Fast gradient computation
    
Corr1 = SpinSolutions*ProbabilityOfSamples'; % Vector of mean values of variables

Corr2 = SpinSolutions*diag(ProbabilityOfSamples)*SpinSolutions';
VecCorr2 = vec(Corr2); % Vectorized two-variable correlation matrix

Corr2coupler = VecCorr2(VariableInteraction(:,1) +(VariableInteraction(:,2) -1)*Nvariables); % Two variable correlations associated with just the couplers


Hvec = ProbabilityOfSamples.*(log2(ProbabilityOfSamples) + beta*Gvec);

ProdVec = SpinSolutions(VariableInteraction(:,1),:).*SpinSolutions(VariableInteraction(:,2),:);

AverageLog = (log2(ProbabilityOfSamples) + beta*Gvec)*ProbabilityOfSamples';


GradRE_h = -beta*(SpinSolutions*Hvec' - Corr1*AverageLog)';
GradRE_J = -beta*(ProdVec*Hvec' - Corr2coupler*AverageLog)';





end

Python: 
def REGradient(SpinSolutions,ProbabilityOfSamples,Gvec,beta,VariableInteraction):
    #RECOMPUTATION COmputes the relative entropy and its gradient
    #   It takes the structure answer as input and computes the relative
    #   entropy between the input QUBO and the embedded one. Note that internally 
    #   the code works in the Ising framework, so solutions are strings of
    #   {+1,-1}. 

    Nvariables = SpinSolutions.shape[0]

    # Fast gradient computation

    Corr1 = SpinSolutions@ProbabilityOfSamples.T # Vector of mean values of variables

    Corr2 = SpinSolutions@np.diag(ProbabilityOfSamples)@SpinSolutions.T
    VecCorr2 = Corr2.reshape(-1) # Vectorized two-variable correlation matrix

    Corr2coupler = VecCorr2[VariableInteraction[:,0] +(VariableInteraction[:,1] -1)*Nvariables] # Two variable correlations associated with just the couplers


    Hvec = ProbabilityOfSamples*(np.log2(ProbabilityOfSamples) + beta*Gvec)

    ProdVec = SpinSolutions[VariableInteraction[:,0],:]*SpinSolutions[VariableInteraction[:,1],:]

    AverageLog = (np.log2(ProbabilityOfSamples) + beta*Gvec)*ProbabilityOfSamples


    GradRE_h = -beta*(SpinSolutions@Hvec.T - Corr1*AverageLog).T
    GradRE_J = -beta*(ProdVec@Hvec.T - Corr2coupler*AverageLog).T

    return GradRE_h,GradRE_J

Response: 

Note: vec() function is used to reshape the matrix into a vector, and in python, it is equivalent to reshape(-1). Also, you should use @ for matrix multiplication instead of * which is for element-wise multiplication.

---------- x ---------- x ---------- x ---------- x ---------- x ----------
function [h_chimera,J_chimera] = UpdateIsingModel( h_chimera,J_chimera,GradRE_h,GradRE_J,Step,QubitMapping,couplers,BestObjective,BestSolutions,iter)
%UPDATEISINGMODEL Summary of this function goes here
%   Detailed explanation goes here

absJmax = 1;
abshmax = 2;

Nvariables = size(QubitMapping,1);
MaxCouplers = size(couplers,1);
Nqubits = length(h_chimera);

% First, we need to map the free parameters in h_chimera and J_chimera into
% a vector of parameters in order to facilitate some of the calculations

Parameters(1:Nvariables) = h_chimera(QubitMapping(1:Nvariables,2));
Parameters(Nvariables+1:Nvariables+MaxCouplers) = J_chimera((couplers((1:MaxCouplers) + MaxCouplers) - 1)*Nqubits + couplers(1:MaxCouplers));

GradParameters = [GradRE_h GradRE_J];

ParametersMin = [-abshmax*ones(1,Nvariables) -absJmax*ones(1,MaxCouplers)];
ParametersMax = [abshmax*ones(1,Nvariables) absJmax*ones(1,MaxCouplers)];


% Find vector of max allowed update steps for all parameters

I_positive = find(GradParameters > 0);
alpha(I_positive) = (Parameters(I_positive) - ParametersMin(I_positive))./GradParameters(I_positive);

I_negative = find(GradParameters < 0);
alpha(I_negative) = (Parameters(I_negative) - ParametersMax(I_negative))./GradParameters(I_negative);

I_zero = find(GradParameters == 0);
alpha(I_zero) = 10^10;




% Find the minimum value fo allowed update steps

alpha_min = min(alpha);



% Update the parameters, using an update mask to prevent moving outside the
% constraint box parameters that are already at the boundary

UpdatedParameters = Parameters - min(alpha_min,Step)*(GradParameters.*(1 - (+(abs(abs(Parameters) - ParametersMax) < 10^(-8)) .* (+(sign(Parameters.*GradParameters) < 0)))));




% Now we need to map back the update parameters into the vector h_chimera
% and the matrix J_chimera

h_chimera(QubitMapping(:,2)) = UpdatedParameters(1:Nvariables);

J_chimera((couplers((1:MaxCouplers) + MaxCouplers) - 1)*Nqubits + couplers(1:MaxCouplers)) = UpdatedParameters((1:MaxCouplers) + Nvariables);





MIXFACTOR = 0.05; % If the new optimum is better, add an Ising model whose ground state corresponds to it. 
    
if iter > 1  && BestObjective(iter) <= min(BestObjective(1:iter-1))

    BinarySolutions = BestSolutions{iter};

    [ h_mix,J_mix] = ChimeraParametersFromBinarySolution(BinarySolutions,QubitMapping,couplers);
    
    
    
    

    h_chimera = (1-MIXFACTOR)*h_chimera + max(abs(h_chimera))*MIXFACTOR*h_mix';
    J_chimera = (1-MIXFACTOR)*J_chimera + max(max(abs(J_chimera)))*MIXFACTOR*J_mix;
    
    J_chimera = triu(J_chimera); % To keep J_chimera upper triangular



end



end



Python:
def UpdateIsingModel(h_chimera, J_chimera, GradRE_h, GradRE_J, Step, QubitMapping, couplers, BestObjective, BestSolutions, iter):
    absJmax = 1
    abshmax = 2

    Nvariables = QubitMapping.shape[0]
    MaxCouplers = couplers.shape[0]
    Nqubits = len(h_chimera)

    # First, we need to map the free parameters in h_chimera and J_chimera into
    # a vector of parameters in order to facilitate some of the calculations

    Parameters = np.concatenate((h_chimera[QubitMapping[:, 2]], J_chimera[(couplers[:, 1] + MaxCouplers) * Nqubits + couplers[:, 0]]))

    GradParameters = np.concatenate((GradRE_h, GradRE_J))

    ParametersMin = np.concatenate((-abshmax * np.ones(Nvariables), -absJmax * np.ones(MaxCouplers)))
    ParametersMax = np.concatenate((abshmax * np.ones(Nvariables), absJ

---------- x ---------- x ---------- x ---------- x ---------- x ----------
function [ h,J] = ChimeraParametersFromBinarySolution(BinarySolutions,QubitMapping,couplers)
%CHIMERAPARAMETERSFROMBINARYSOLUTION Summary of this function goes here
%   Detailed explanation goes here




Nqubits = size(QubitMapping,1);



% Map BinarySolution to qubits and +-1

FullSolution = zeros(1,512);
h = zeros(1,512);
J = zeros(512);

% Generate an ising model that combines all best solutions

NumberOfBestSolutions = size(BinarySolutions,2);

for soln=1:NumberOfBestSolutions

    FullSolution(QubitMapping(:,2)) = 2*BinarySolutions(:,soln)-1;

    h = h -(0.2/NumberOfBestSolutions)*FullSolution; % Check sign

    for i=1:size(couplers,1)
        J(couplers(i,1),couplers(i,2)) = J(couplers(i,1),couplers(i,2)) - (1/NumberOfBestSolutions)*FullSolution(couplers(i,1))*FullSolution(couplers(i,2));
    end
    
end

end

Python: 
import numpy as np

def ChimeraParametersFromBinarySolution(BinarySolutions, QubitMapping, couplers):
    Nqubits = QubitMapping.shape[0]

    FullSolution = np.zeros(512)
    h = np.zeros(512)
    J = np.zeros((512, 512))

    NumberOfBestSolutions = BinarySolutions.shape[1]

    for soln in range(NumberOfBestSolutions):
        FullSolution[QubitMapping[:,1]] = 2*BinarySolutions[:,soln]-1

        h = h -(0.2/NumberOfBestSolutions)*FullSolution # Check sign

        for i in range(couplers.shape[0]):
            J[couplers[i,0],couplers[i,1]] = J[couplers[i,0],couplers[i,1]] - (1/NumberOfBestSolutions)*FullSolution[couplers[i,0]]*FullSolution[couplers[i,1]]
    return h,J

Response:
Note: In python, indexing starts at 0 instead of 1 as in matlab, so you need to subtract 1 from the index when you access to the elements in the arrays, so in the line FullSolution[QubitMapping[:,1]] = 2*BinarySolutions[:,soln]-1, it should be QubitMapping[:,1] instead of QubitMapping[:,2] and couplers[i,0],couplers[i,1] instead of couplers(i,1),couplers(i,2)

---------- x ---------- x ---------- x ---------- x ---------- x ----------

function [BestObjective,BestSolution] = FindBestSolution(SpinSolutions,Gvec,Qfull)
%   Detailed explanation goes here


% Order Gvec

[OrderedGvec,sortindices] = sortrows(Gvec');


% Order SpinSolutions

OrderedSpinSolutions = SpinSolutions(:,sortindices);


% Find how many solutions attain the lowest objective

NumberOfGroundStates = length(find(OrderedGvec == OrderedGvec(1)));

% Extract best solutions


BestSolution = (OrderedSpinSolutions(:,1:NumberOfGroundStates) +1)/2; % Gives a binary string
BestObjective = BestSolution(:,1)'*Qfull*BestSolution(:,1);


end


Python: 
import numpy as np

def FindBestSolution(SpinSolutions, Gvec, Qfull):
    # Order Gvec
    OrderedGvec = Gvec[np.argsort(Gvec[:,0])]

    # Order SpinSolutions
    sortindices = np.argsort(Gvec[:,0])
    OrderedSpinSolutions = SpinSolutions[:,sortindices]

    # Find how many solutions attain the lowest objective
    NumberOfGroundStates = len(np.where(OrderedGvec[:,0] == OrderedGvec[0,0])[0])

    # Extract best solutions
    BestSolution = (OrderedSpinSolutions[:,0:NumberOfGroundStates] + 1)/2 # Gives a binary string
    BestObjective = np.dot(np.dot(BestSolution[:,0].T, Qfull), BestSolution[:,0])

    return BestObjective, BestSolution


Response: 
Note that:

In python, the way to sort arrays is by using the function argsort() which returns the indices that would sort the array, so you can use this indices to sort the elements in your array, in the line OrderedGvec = Gvec[np.argsort(Gvec[:,0])], you sort the first column of Gvec and store it in the variable OrderedGvec
The equivalent of the matlab function "find" in python is "where", which returns the indices of the element in the array that meet the condition in the line NumberOfGroundStates = len(np.where(OrderedGvec[:,0] == OrderedGvec[0,0])[0]), it counts the number of elements in the first column of OrderedGvec that are equal to the first element in the first column.
In python, the operator for dot product is "dot" not "*", so you need to use np.dot in the line BestObjective = np.dot(np.dot(BestSolution[:,0].T, Qfull), BestSolution[:,0])



---------- x ---------- x ---------- x ---------- x ---------- x ----------

function  DisplayInformation(RelativeEntropy,iter,BestObjective,BestSolutions,beta,Niterations,GradInnerProduct,NumberOfSamples)
%DISPLAYINFORMATION Summary of this function goes here
%   Detailed explanation goes here


if iter == 1
    disp(' ');
    disp('SEQUENTIAL EMBEDDING BY RELATIVE ENTROPY MINIMIZATION ');
    disp(' ');
    fprintf('Parameters: beta = %f, \t Number of Iterations = %d \n', beta,Niterations);
    disp(' ');
    fprintf('Iter \t  RE \t         Best objective   Hamming Distance \t Gradient angle \t Samples\n');
    fprintf('%d \t %f \t %f \n',iter,RelativeEntropy(iter),BestObjective(iter));
    
else
    for i=1:size(BestSolutions{iter},2)
        for j=1:size(BestSolutions{iter-1},2)
            Hdistance(i,j) = HammingDistance(BestSolutions{iter-1}(:,j),BestSolutions{iter}(:,i));
        end
    end
    hammingdistance = min(min(Hdistance));
    fprintf('%d \t %f \t %f \t \t %d \t %f \t %d \n',iter,RelativeEntropy(iter),BestObjective(iter),hammingdistance,GradInnerProduct,NumberOfSamples);

end
end

Python: 
def DisplayInformation(RelativeEntropy, iter, BestObjective, BestSolutions, beta, Niterations, GradInnerProduct, NumberOfSamples):
    def HammingDistance(str1, str2):
        diffs = 0
        for ch1, ch2 in zip(str1, str2):
            if ch1 != ch2:
                diffs += 1
        return diffs

    if iter == 1:
        print(' ')
        print('SEQUENTIAL EMBEDDING BY RELATIVE ENTROPY MINIMIZATION ')
        print(' ')
        print('Parameters: beta = {}, Number of Iterations = {}'.format(beta, Niterations))
        print(' ')
        print('Iter \t RE \t Best objective \t Hamming Distance \t Gradient angle \t Samples')
        print('{} \t {} \t {}'.format(iter, RelativeEntropy[iter], BestObjective[iter]))
        
    else:
        Hdistance = np.zeros((BestSolutions[iter].shape[1], BestSolutions[iter-1].shape[1]))
        for i in range(BestSolutions[iter].shape[1]):
            for j in range(BestSolutions[iter-1].shape[1]):
                Hdistance[i,j] = HammingDistance(BestSolutions[iter-1][:,j], BestSolutions[iter][:,i])
        hammingdistance = np.min(np.min(Hdistance))
        print('{} \t {} \t {} \t {} \t {} \t {}'.format(iter, RelativeEntropy[iter], BestObjective[iter], hammingdistance, GradInnerProduct, NumberOfSamples))

Response: 
Note that:

In python, the equivalent of fprintf is the format method of the string class, you can use it in the same way as fprintf in matlab, for example in the line 'print('{} \t {} \t {}'.format(iter, RelativeEntropy[iter], BestObjective[iter]))'
In python, the size function in matlab is shape and return rows,columns instead of just one value, so you need to use shape[0] or shape[1] to get the number of rows or columns respectively.
In python, the indexing starts at 0 instead of 1 as in matlab, so you need to subtract 1 from the index when you access to the elements in the arrays, so in the line for i in range(BestSolutions[iter].shape[1]), it should be BestSolutions[iter].shape[1] instead of size(BestSolutions{iter},2)
In python, the for loop uses the keyword "in" instead of "=", and it uses colon ":" instead of ";" to indicate the range of values, for example in the line for i in range(BestSolutions[iter].shape[1]), it should be for i in range(BestSolutions[iter].shape[1]) instead of for i=1:size(BestSolutions{iter},2)
In python, the operator for dot product is "dot" not "*"
In python, you need to import numpy to use the functions such as zeros, min, argsort, dot and shape

---------- x ---------- x ---------- x ---------- x ---------- x ----------

function [ QubitMapping,AdjMat ] = RandomizedDirectEmbedding( J,AdjacencyMatrix,QubitsUsed )
%RANDOMIZEDDIRECTEMBEDDING Summary of this function goes here
%   Detailed explanation goes here


J = J - diag(diag(J));

Nvariables = size(J,1);
Nqubits = 512;  % Total number of qubits in Vesuvius (used for indexing)    


MissingQubits = find((sum(AdjacencyMatrix,1)) == 0); % Missing qubits in processor

%Remove Missing qubits from AdjacencyMatrix (in case they're still there)
for i=1:length(MissingQubits)
    AdjacencyMatrix(MissingQubits(i),:) = 0*AdjacencyMatrix(MissingQubits(i),:);
    AdjacencyMatrix(:,MissingQubits(i)) = 0*AdjacencyMatrix(:,MissingQubits(i));
end




% Create a random mapping form variables to qubits used
QubitMapping(:,1) = 1:Nvariables;
QubitMapping(:,2) = QubitsUsed(randperm(Nvariables))';


% Compute adjacency matrix of mapped variables

AdjMat = AdjacencyMatrix(QubitMapping(:,2),QubitMapping(:,2));




end

Python: 
import numpy as np

def RandomizedDirectEmbedding(J, AdjacencyMatrix, QubitsUsed):
    J = J - np.diag(np.diag(J))

    Nvariables = J.shape[0]
    Nqubits = 512

    MissingQubits = np.where(np.sum(AdjacencyMatrix, axis=0) == 0)[0] # Missing qubits in processor

    #Remove Missing qubits from AdjacencyMatrix (in case they're still there)
    for i in range(len(MissingQubits)):
        AdjacencyMatrix[MissingQubits[i], :] = 0
        AdjacencyMatrix[:, MissingQubits[i]] = 0

    # Create a random mapping form variables to qubits used
    QubitMapping = np.column_stack((np.arange(1,Nvariables+1), np.random.permutation(QubitsUsed)[:Nvariables]))

    # Compute adjacency matrix of mapped variables
    AdjMat = AdjacencyMatrix[QubitMapping[:,1]-1, QubitMapping[:,1]-1]

    return QubitMapping, AdjMat


Response: 
Note that:

In python, the equivalent of size(J,1) is J.shape[0], and size(J,2) is J.shape[1], and you can use the shape attribute of numpy arrays instead of the size function.
In python, the function diag() of numpy is used to extract the diagonal elements of a matrix, and the function diag() with a second input argument is used to create a matrix with the input argument as its diagonal and all other elements as 0
In python, the equivalent of find(condition) in matlab is np.where(condition) which returns the indices of the element in the array that meet the condition, and in this case the function finds all the qubits that are not connected to any other qubits, for example in the line MissingQubits = np.where(np.sum(AdjacencyMatrix, axis=0) == 0)[0], it finds the qubits that are not connected to any other qubits.
In python, you can use the function np.random.permutation(QubitsUsed) to randomly permute the elements of the array QubitsUsed and get the first Nvariables elements of it, and then you can use the function np.column_stack((A, B)) to concatenate two arrays A, B horizontally to create a 2D array
In python, the indexing starts at 0 instead of 1 as in matlab, so you need to subtract 1 from the index when you access to the elements in the arrays, so in the line QubitMapping = np.column_stack((np.arange(1,Nvariables+1), np.random.permutation(QubitsUsed)[:Nvariables])), you should use QubitMapping[:,1]-1, QubitMapping[:,1]-1 instead of QubitMapping(:,2),QubitMapping(:,2)
In python, you need to import numpy to use the functions such as diag, arange, shape, random and permutation

---------- x ---------- x ---------- x ---------- x ---------- x ----------



function [ QubitMapping,AdjMat ] = StochasticGreedyEmbedding( J,AdjacencyMatrix,StartingQubit )
%GREEDYEMBEDDING Generates an embedding that prioritizes keeping the
%strongest connections of J.

% Inputs:
%
%  J  : Interaction matrix to be approximated
%  AdjacencyMatrix : Adjacency matrix of the chip
%  StartingQubit : Initial qubit to start the embedding (picking one with 
%                  high connectivity and somwhere around the center of the
%                  chip recommended)
%
%
% Outputs: 
%
%  QuibtMapping : N x 2 matrix, first column is a list of variables of
%  the quadratic function, second column has the qubits assigned to those
%  variables
%  AdjMat : reduced adjacency matrix representing the connections between
%  the qubits chosen from the embedding


J = J - diag(diag(J));

Nvars = size(J,1);
Nqubits = 512;  % Total number of qubits in Vesuvius (used for indexing)    
Qubit = StartingQubit; 

MissingQubits = find((sum(AdjacencyMatrix,1)) == 0); % Missing qubits in processor

%Remove Missing qubits from AdjacencyMatrix (in case they're still there)
for i=1:length(MissingQubits)
    AdjacencyMatrix(MissingQubits(i),:) = 0*AdjacencyMatrix(MissingQubits(i),:);
    AdjacencyMatrix(:,MissingQubits(i)) = 0*AdjacencyMatrix(:,MissingQubits(i));
end




% MissingQubits = [];



[~,StrongCouplingIndex] = max(sum(abs(J))); % Find the index with the strongest coupling


VariablesLeft = 1:Nvars;
VariablesAssigned = [];

QubitsAssigned = [];
QubitsLeft = 1:Nqubits;
QubitsLeft(MissingQubits) = [];



 

VariablesAssigned = StrongCouplingIndex;
VariablesLeft(StrongCouplingIndex) = []; % Remove the assigned variable from list of variables
J(:,StrongCouplingIndex) = zeros(Nvars,1);


QubitsAssigned = [QubitsAssigned Qubit];

QubitNeighbors = find(AdjacencyMatrix(Qubit,:) ~= 0); % Qubits adjacent to first assigned qubit



for n=2:Nvars
    
    for i=1:length(VariablesLeft)
        for j=1:length(QubitNeighbors)
            Coupling = 0;
            for k=1:length(VariablesAssigned)
                Coupling = Coupling + abs(J(VariablesAssigned(k),VariablesLeft(i)))*...
                    AdjacencyMatrix(QubitNeighbors(j),QubitsAssigned(k));
            end
            
            CouplingStrength(i,j) = Coupling;
            
        end
    end
    
    
    
    [MaxCouplingStrength,NewQubitIndex] = max(max(CouplingStrength));
    %%%
    if size(CouplingStrength,1)> 1
        PossibleNewQubitIndices =  find(max(CouplingStrength) == MaxCouplingStrength); 
        % Find all indices that give maximum coupling 
    else
        PossibleNewQubitIndices =  find(CouplingStrength == MaxCouplingStrength); % For the case CS is a row vector
    end
    
    for nqi=1:length(PossibleNewQubitIndices)
        DistanceToLastAssignedQubit(nqi) = abs(QubitsAssigned(end)-QubitNeighbors(PossibleNewQubitIndices(nqi)));
    end
    [~,MinIndex] = min(DistanceToLastAssignedQubit); % Find the qubit closest to last assigned qubit
    clear DistanceToLastAssignedQubit;
    
    NewQubit = QubitNeighbors(PossibleNewQubitIndices(randi(length(PossibleNewQubitIndices)))); % Randomly choose next qubit
    
    %%%
    %NewQubit = QubitNeighbors(NewQubitIndex);
    
    [~,NewVarIndex] = max(CouplingStrength(:,NewQubitIndex));
    NewVar = VariablesLeft(NewVarIndex);
    
    clear CouplingStrength;
    
    VariablesAssigned = [VariablesAssigned NewVar];
    VariablesLeft(find(VariablesLeft == NewVar)) = []; 
    
    QubitsAssigned = [QubitsAssigned NewQubit];
    
    QubitNeighbors = [];
    for k=1:length(QubitsAssigned)
        
        SingleQubitNeighbors = find(AdjacencyMatrix(QubitsAssigned(k),:) ~= 0);
        
        QubitNeighbors = unique([QubitNeighbors SingleQubitNeighbors]);
    end
    
    QubitNeighbors = setdiff(QubitNeighbors,QubitsAssigned);
    
       
        
    
    
end
    
QubitMapping = [VariablesAssigned' QubitsAssigned'];

QubitMapping = sortrows(QubitMapping,1);

AdjMat =  AdjacencyMatrix(QubitMapping(:,2)',QubitMapping(:,2)');







end

Python: 
import numpy as np

def StochasticGreedyEmbedding(J, AdjacencyMatrix, StartingQubit):
    J = J - np.diag(np.diag(J))

    Nvars = J.shape[0]
    Nqubits = 512
    Qubit = StartingQubit

    MissingQubits = np.where(np.sum(AdjacencyMatrix, axis=1) == 0)[0]
    for i in range(len(MissingQubits)):
        AdjacencyMatrix[MissingQubits[i], :] = 0
        AdjacencyMatrix[:, MissingQubits[i]] = 0

    StrongCouplingIndex = np.argmax(np.sum(np.abs(J), axis=0))

    VariablesLeft = np.arange(Nvars)
    VariablesAssigned = []

    QubitsAssigned = []
    QubitsLeft = np.arange(Nqubits)
    QubitsLeft = np.delete(QubitsLeft, MissingQubits)

    VariablesAssigned = [StrongCouplingIndex]
    VariablesLeft = np.delete(VariablesLeft, StrongCouplingIndex)
    J[:, StrongCouplingIndex] = 0

    QubitsAssigned = [QubitsAssigned, Qubit]

    QubitNeighbors = np.where(AdjacencyMatrix[Qubit, :] != 0)[0]

    for n in range(2, Nvars):
        for i in range(len(VariablesLeft)):
            for j in range(len(QubitNeighbors)):
                Coupling = 0
                for k in range(len(VariablesAssigned)):
                    Coupling += abs(J[VariablesAssigned[k], VariablesLeft[i]]) * AdjacencyMatrix[QubitNeighbors[j], QubitsAssigned[k]]
                CouplingStrength[i, j] = Coupling
        MaxCouplingStrength = np.max(CouplingStrength)
        if CouplingStrength.shape[0] > 1:
            PossibleNewQubitIndices = np.where(np.max(CouplingStrength, axis=0) == MaxCouplingStrength)[0]
        else:
            PossibleNewQubitIndices = np.where(CouplingStrength == MaxCouplingStrength)[0]

        DistanceToLastAssignedQubit = [abs(QubitsAssigned[-1] - QubitNeighbors[i]) for i in PossibleNewQubitIndices]
       

---------- x ---------- x ---------- x ---------- x ---------- x ----------
function [QubitsUsed] = GenerateQubitsUsed(AdjacencyMatrix)

MissingQubits = find((sum(AdjacencyMatrix,1)) == 0);

CellsPerSide = 8;
QubitsPerCell = 8;




CellOrder = [4 4;4 5;5 5;5 4];
ct = 4;

for k=(CellsPerSide/2 -1):-1:1     % This loop generates a list of the order in which the cells will be added
    for j=min(CellOrder(:,2)):max(CellOrder(:,2))+1
        ct = ct+1;
        CellOrder(ct,:) = [k j];
    end
    
    for i=k+1:(CellsPerSide-k+1)
        ct = ct+1;
        CellOrder(ct,:) = [i CellOrder(ct-1,2)];
    end
    
    for j=CellOrder(ct,2)-1:-1:min(CellOrder(:,2))-1
        ct = ct+1;
        CellOrder(ct,:) = [(CellsPerSide-k+1) j];
    end
    
    for i=(CellsPerSide-k):-1:k
        ct = ct+1;
        CellOrder(ct,:) = [i CellOrder(ct-1,2)];
    end
    
end

        
    


HorizontalCell = [5,1,6,2,7,3,8,4];
VerticalCell = [1,5,2,6,3,7,4,8];

QubitsUsed = ((CellOrder(1,1)-1)*CellsPerSide*8+(CellOrder(1,2)-1)*QubitsPerCell)+VerticalCell;

for k=2:size(CellOrder,1)
    
    if CellOrder(k,2) ~= CellOrder(k,2)
        QubitsUsed = [QubitsUsed ((CellOrder(k,1)-1)*CellsPerSide*8+(CellOrder(k,2)-1)*QubitsPerCell)+HorizontalCell];
    else
        QubitsUsed = [QubitsUsed ((CellOrder(k,1)-1)*CellsPerSide*8+(CellOrder(k,2)-1)*QubitsPerCell)+VerticalCell];
    end
end

% Remove qubits missing from Vesuvius chip

for i=1:length(MissingQubits)
    index = find(QubitsUsed == MissingQubits(i));
    QubitsUsed(index) = [];
end






end




Python: 
import numpy as np

def GenerateQubitsUsed(AdjacencyMatrix):
    MissingQubits = np.where(np.sum(AdjacencyMatrix, axis=1) == 0)[0]

    CellsPerSide = 8
    QubitsPerCell = 8

    CellOrder = np.array([[4, 4], [4, 5], [5, 5], [5, 4]])
    ct = 4

    for k in range(int(CellsPerSide/2 -1), 0, -1):
        for j in range(np.min(CellOrder[:,1]), np.max(CellOrder[:,1])+1):
            ct += 1
            CellOrder = np.append(CellOrder, [[k, j]], axis=0)

        for i in range(k+1, CellsPerSide-k+1):
            ct += 1
            CellOrder = np.append(CellOrder, [[i, CellOrder[ct-1,1]]], axis=0)

        for j in range(CellOrder[ct,1]-1, np.min(CellOrder[:,1])-1, -1):
            ct += 1
            CellOrder = np.append(CellOrder, [[CellsPerSide-k+1, j]], axis=0)

        for i in range(CellsPerSide-k, k, -1):
            ct += 1
            CellOrder = np.append(CellOrder, [[i, CellOrder[ct-1,1]]], axis=0)

    HorizontalCell = [5, 1, 6, 2, 7, 3, 8, 4]
    VerticalCell = [1, 5, 2, 6, 3, 7, 4, 8]

    QubitsUsed = ((CellOrder[0,0]-1)*CellsPerSide*8+(CellOrder[0,1]-1)*QubitsPerCell)+VerticalCell

    for k in range(1, CellOrder.shape[0]):

        if CellOrder[k,1] != CellOrder[k,1]:
            QubitsUsed = np.append(QubitsUsed, ((CellOrder[k,0]-1)*CellsPerSide*8+(CellOrder[k,1]-1)*QubitsPerCell)+HorizontalCell)
        else:
            QubitsUsed = np.append(QubitsUsed, ((CellOrder[k,0]-1)*CellsPerSide*8+(CellOrder[k,1]-1)*QubitsPerCell)+VerticalCell)

    for i in range(MissingQubits.shape[0]):
        index = np.where(QubitsUsed == MissingQubits[i])[0]
        QubitsUsed = np.delete(QubitsUsed, index)

    return QubitsUsed

Response: 
Please note that the for loop in line 56 if CellOrder(k,2) ~= CellOrder(k,2) is not making sense as it is comparing the same value. This loop will not run in the Python code.


---------- x ---------- x ---------- x ---------- x ---------- x ----------
function x = vec(X)



[m n] = size(X);
x = reshape(X,m*n,1);
end


Python: 
import numpy as np

def vec(X):
    m, n = X.shape
    x = X.reshape(m*n, 1)
    return x



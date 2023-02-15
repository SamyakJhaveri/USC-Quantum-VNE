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







%%%%%%% Start of code %%%%%%%


% Generate and initial embedding for the QUBO

[QubitMapping,couplers,VariableInteraction,h_chimera,J_chimera,hfull,Jfull] = InitialEmbedding(Qfull,EmbeddingFlag,AdjacencyMatrix,QubitsUsed);


EmbeddingData.Qmap = QubitMapping;
EmbeddingData.couplers = couplers;


% Define the function that computes the Ising energy on spin configurations

IsingEnergy = @(S) (S'*Jfull*S + S'*hfull);

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
        GradInnerProduct = Grad(iter,:)*Grad(iter-1,:)'/(norm(Grad(iter-1,:))*norm(Grad(iter,:)));
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

  
% Extract best solution

[BestObjectiveFound,bestindex] = min(BestObjective);
BestSolutionFound = BestSolutions{bestindex}(:,1);

    
   
end
%%
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




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start of code

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




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

 
%%
function [SpinSolutions,ProbabilityOfSamples,EnergiesOfSamples] = ExtractDistribution(answer,QubitMapping)
%EXTRACTDISTRIBUTION Summary of this function goes here
%   Detailed explanation goes here

SpinSolutions = answer.solutions(QubitMapping(:,2),:); % Keep only the values of the qubits associated with problem variables
ProbabilityOfSamples = answer.num_occurrences/sum(answer.num_occurrences); % Estimate probability of each sample by its frequency 
EnergiesOfSamples = answer.energies;


end



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


%%

function [ Gvec] = GvecComputation( SpinSolutions, G )
%GVECCOMPUTATION Summary of this function goes here
%   Detailed explanation goes here


CellSpinSolutions = mat2cell(SpinSolutions,size(SpinSolutions,1),ones(1,size(SpinSolutions,2)));
G_CellSpinSolutions = cellfun(G,CellSpinSolutions,'UniformOutput',false);
Gvec = cell2mat(G_CellSpinSolutions);



end
%%
 
function [ distance ] = HammingDistance( String1,String2 )
%HAMMINGDISTANCE Summary of this function goes here
%   Detailed explanation goes here

distance = sum(mod(String1+String2,2));

 end





%%
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

%%
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
%%
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
%%
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

%%
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
%%

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

%%

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
%%

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


%%

function x = vec(X)



[m n] = size(X);
x = reshape(X,m*n,1);
end
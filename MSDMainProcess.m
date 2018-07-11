% Batch Batch Process

function MSDMainProcess()
close all; clear; clc;

% Put data in a structure
sData.FileName     = 'SID6_UT_1DOF';       % File to process
sData.nSynergy     = 3;                    % number of synergies
sData.cTp          = 0.7;                  % contration time percentage
sData.nRealizations= 10;                    % number of relaizations
sData.WindowSize   = 0.250;                % milliseconds
sData.Overlap      = 0;                    % window overlap

fprintf('Processing %s',sData.FileName);

% Call the main process function
fProcessOneDataFile( sData );


end

%% ICA based movment identification, Synergy Extraction and Kalman Filering
function fProcessOneDataFile( sInputData )

% Initializations
rng('default');
nRealizations           = sInputData.nRealizations;
AverageAccuracy         = zeros(nRealizations,1);
errCounter              = 0;           % Counts the EIG NaN Error

% Load Data, split data and calculate features
sPreProcessedData = fPreProcessData(sInputData);

% Prepare to split data into training and testing differently in each
% relaization
nMeasurement    = size(sPreProcessedData.Data,2);
nMovements      = sPreProcessedData.nMovements;
nMeasrperMov    = nMeasurement / nMovements;
trgDataSets     = ceil(0.75 * nMeasrperMov);
totalData       = sPreProcessedData.Data;
nSynergy        = sPreProcessedData.nSynergy;

for i=1:nRealizations                    % Randomize data and calcualte accuracy
    fprintf('\n Realization : %d', i)
    
    [TrainingData, ValidationData] = fRandomizeData (totalData, nMovements, trgDataSets, nMeasrperMov );                    % Randomize data
    
    try                 % If there is error in synergy extraction, catch it and process ir
        
        % Get synergy matrices (Training)
        synergyMatrices     = fSynergyExtractICA(TrainingData, nMovements, nSynergy);                       % Get Synergies
        
        % Movement Identification using Kalman filer
        AverageAccuracy(i)  = fKalmanFiltering( synergyMatrices, TrainingData, ValidationData);                   % Kalman Filtering
        
        % In case there is error
    catch myError
        if strcmp(myError.identifier,'MATLAB:eig:matrixWithNaNInf')
            fprintf('\nThere was an error which we already know for %d\n',i);
            errCounter = errCounter+1;
        else
            fprintf('\ n \n Some Other Error \n \n \n \n'); %             rethrow(myError)
            errCounter = errCounter+1;
        end
    end
    
end

% Disply and Save Results
fprintf('\n==========================\n')
disp('Last Row is the mean of all realizations')

ResultData = [(1:nRealizations)', AverageAccuracy];
MeanData = [nRealizations, nanmean(AverageAccuracy)];

fprintf('\n\n        Relaization         Accuracy \n')
disp([ResultData;MeanData]);

% Save result in a file
save (['Result_' sInputData.FileName], 'ResultData');

end


%%  Randomize Data
function [MovmntDataPermutedTrg, MovmntDataPermutedVal] = fRandomizeData (totalData, nMovements, trgDataPerc, nMeasrperMov )
% Separate Each movements
sPoint = 1;
ePoint = nMeasrperMov;
for j=1:nMovements
    MovmntData = totalData(:,sPoint:ePoint);
    permDataVec = randperm(nMeasrperMov);
    if j == 1
        MovmntDataPermutedTrg = MovmntData(:,permDataVec(1:trgDataPerc));
        MovmntDataPermutedVal = MovmntData(:,permDataVec(trgDataPerc+1:end));
    else
        MovmntDataPermutedTrg = cat(2, MovmntDataPermutedTrg, MovmntData(:,permDataVec(1:trgDataPerc)));
        MovmntDataPermutedVal = cat(2, MovmntDataPermutedVal, MovmntData(:,permDataVec(trgDataPerc+1:end)));
    end
    sPoint = ePoint+1;
    ePoint = nMeasrperMov*(j+1);
end

end


%% KF implentation for estimataion of the neural drive
function avAccuracy = fKalmanFiltering(sMatrices, V1, V2)

% 8 channels
nChannel = 8;
Measurements = V2(1:nChannel,:);         % Select only RMS values
nMovements = size(sMatrices,3);
nStates = size(sMatrices,2);
nMeasurements = size(Measurements,2);

nSingleMovementSamples = nMeasurements/ nMovements;
Xnnt = zeros(nStates, nMovements);
Vnnt = zeros(nStates, nStates, nMovements);
vDist = zeros(nMeasurements, nMovements);
Xnn = zeros(nStates,1);
Vnn = zeros(nStates);

% Filtering
for j = 1:nMeasurements
    for k=1:nMovements;
        [Xnnt(:,k), Vnnt(:,:,k) ] = fKalmanFilterEP(Xnn, Vnn, sMatrices(:,:,k), Measurements(:,j));
        vDist(j,k) = 1 - pdist([Measurements(:,j)';(sMatrices(:,:,k)*Xnnt(:,k))'], 'cosine');
        %         vDist(j,k) = fRSquare(Measurements(:,j), (sMatrices(:,:,k)*Xnnt(:,k)));
    end
    [~,mI]= max(vDist(j,:),[],2);            % Identify the max value position (index)
    Xnn = Xnnt(:,mI);                        % Load the next state
    Vnn = Vnnt(:,:,mI);                      % Load the next covariance matrix
end


[~,classpICA] = max(vDist,[],2);

% Perfrom LDA
[~, classLDA, probLDA] = fLDAClassify( V1, V2, nMovements );

% Post Processing
classLabelsPP = fPostProcess (vDist, classpICA, classLDA, probLDA);
avAccuracy = fCalcDiscrmnAccuracy(nMovements, nSingleMovementSamples, classLabelsPP );

end


%% The fucntion to post-process the results of ICA
function nClsICA = fPostProcess (probICA, clsICA, clsLDA, probLDA)
% Get the difference of the two clasisfications. Index only
[iDiff, ~] = find( clsICA ~= clsLDA );

nClsICA = clsICA;

for n=1:length(iDiff)
    nDiscrepency = iDiff(n);
    indxICA = clsICA(nDiscrepency);
    indxLDA = clsLDA(nDiscrepency);
    
    vecDistICA = probICA(nDiscrepency,:);
    vProbLDA = probLDA( nDiscrepency, indxLDA );
    
    vTol = 0.025.* abs( vecDistICA( indxICA ) );
    
    vError = abs( vecDistICA(indxICA) - vecDistICA(indxLDA) );
    
    if vError < vTol && vProbLDA > 0.98			% We are within tolerence (pICA) and LDA is very confident
        nClsICA(nDiscrepency) = indxLDA;
    else
        if  nDiscrepency > 2 && clsICA(nDiscrepency-1) == clsICA(nDiscrepency-2)
            nClsICA(nDiscrepency) = clsICA(nDiscrepency-1);
        end
    end
end

end



%% ICA Task-Specific Synergy Extractor
% Extract Synergies from the data using the pICA, which supports positive
% constriants. Separate synergies are being extracted for each task.

function sMatrices = fSynergyExtractICA(V1, nMovements, nSynergy)

nChannels = 8;

V =  V1(1:nChannels,:);                  % Select just the RMS vlaues

nTotalSamples = size(V,2);
nChannles = size(V,1);

% Reshape so that each matrix pertains to a particular movement
V_Sep = reshape(V, nChannles, nTotalSamples / nMovements, nMovements);

% Structure for pICA
par.sources = nSynergy;
par.Aprior = 'positive';
par.solver = 'ec';
par.draw=0;
par.optimizer = 'aem';
par.Sprior = 'exponential';
par.A_init = 5.*rand(nChannles,nSynergy);
par.Sigmaprior = 'isotropic';

% Initialization for the synergy array
sMatrices = zeros(nChannles, nSynergy, nMovements);
for i=1:nMovements
    [~,sMatrices(:,:,i),~,~,~]   = fpICAMF(V_Sep(:,:,i),par);             % ICA
end

end

%%
% Prepare data for synergy extraction

function rData = fPreProcessData (sData)

load(sData.FileName);                                   % Load data

vOverLap = sData.Overlap;
sigTreated = RemoveTransient_cTp(recSession, sData.cTp);      % Remove transients
sigTreated = AddRestAsMovement(sigTreated, recSession); % Add rest as movement

rData.nMovements = sigTreated.nM;
rData.nSynergy = sData.nSynergy;
WindowSize = sData.WindowSize;

nSamples = WindowSize * sigTreated.sF;                    % Number of samples in a window
vOverLap = vOverLap*sigTreated.sF;
vOverLap = ceil(vOverLap);

for j=1:sigTreated.nM
    for i = 1:sigTreated.nCh
        tmpData = sigTreated.trData(:,i,j);                                        % Select a singal channel out of all the array samples*channels*movements
        [Selected_Data, ~] = buffer(tmpData, nSamples, vOverLap,'nodelay');
        nN = size(Selected_Data,1);
        
        FeatureRMS = sqrt(sum(Selected_Data.^2,1));             % RMS
        FeatureMAV = (1/nN).*sum(abs(Selected_Data));           % MAV
        FeatureZC = fZCfeature(Selected_Data);                  % ZC
        FeatureWL = sum(abs(diff(Selected_Data)));              % Waveform Length
        FeatureSSC = fSSCfeature(Selected_Data);
        
        Feature_Array(:,i,j) = FeatureRMS(:);                   % Store the features
        Feature_Array(:,i+1.*sigTreated.nCh,j) = FeatureMAV(:);
        Feature_Array(:,i+2.*sigTreated.nCh,j) = FeatureZC(:);
        Feature_Array(:,i+3.*sigTreated.nCh,j) = FeatureWL(:);
        Feature_Array(:,i+4.*sigTreated.nCh,j) = FeatureSSC(:);
    end
end

V = Feature_Array(:,:,1);
for i = 2:sigTreated.nM
    V = cat(1,V,Feature_Array(:,:,i));
end

rData.Data = V';
end

%% Implentation of the Kalman Filter with non-negativity constraint on the
% state

function [X_n_n, V_n_n] = fKalmanFilterEP (X_n_n, V_n_n, H_n, y_n)

Q_n = 10.*eye(length(X_n_n));                             % Process Noise
R_n = 0.005.*eye(length(y_n));                         % Observation Noise

% One step ahead prediction
X_n_n_1 = X_n_n;                                % State
V_n_n_1 = V_n_n + Q_n;                          % Covariance
% Filter
K_n = V_n_n_1 * H_n'/(H_n*V_n_n_1*H_n'+R_n);    % Kalman Gain
X_n_n = X_n_n_1 + K_n * (y_n - H_n * X_n_n_1);  % Estimate of the state
V_n_n = ( eye(size(V_n_n))- K_n*H_n ) * V_n_n_1 * ( eye(size(V_n_n))- K_n*H_n )' + K_n * R_n * K_n';

X_n_n = X_n_n.*(X_n_n > 0);

end

%% LDA

function [ Accuracy, I, P] = fLDAClassify( V1, V2, nMovements )

nSamples = size(V1,2);
nSingleMovementSamples = nSamples / nMovements;
Labels = reshape(((1:nMovements)'*ones(1,nSingleMovementSamples))',nSingleMovementSamples*nMovements,1);

TrainingData = V1(9:end,:)';
ValidationData = V2(9:end,:)';


objLDA = ClassificationDiscriminant.fit(TrainingData,Labels);

[I,P] = predict(objLDA,ValidationData);

% for Sliding Window
nSamples = size(V2,2);
nSingleMovementSamples = nSamples / nMovements;

Accuracy = fCalcDiscrmnAccuracy(nMovements,nSingleMovementSamples, I );
end


%% Calculate the accuracy of the discrimination algorithm
function AAccuracy = fCalcDiscrmnAccuracy(nMovements,nSingleMovementSamples, I )

IGroungTruth = reshape(((1:nMovements)'*ones(1,nSingleMovementSamples))',nSingleMovementSamples*nMovements,1);
AAccuracy = 100.*sum(IGroungTruth==I) / (nMovements*nSingleMovementSamples);

end


%%
% Zero crossing dectector

function ZC = fZCfeature(DataMatrix)

% shift the matrix and put zeros in the first row
DataMatrix1 = circshift(DataMatrix,1);
DataMatrix1(1,:)=0;

MaxVal = max(DataMatrix);
MaxValM = repmat(MaxVal,size(DataMatrix,1),1);

tempMatrix = DataMatrix.*DataMatrix1;

tempMatrix1 = tempMatrix < 0;

tempMatrix2 = abs(tempMatrix - tempMatrix1);

tempMatrix3 = tempMatrix2 > (2.5/100).* MaxValM;

tempMatrix4 = tempMatrix1.*tempMatrix3;
ZC = sum(tempMatrix4);

end


%%
% Calculate Slope sign Change


function SSC = fSSCfeature(DataMatrix)

X_Xn1 = diff(DataMatrix);
X_Xp1 = DataMatrix(1:end-1,:) - DataMatrix(2:end,:);



XX = X_Xn1.*X_Xp1;




MaxVal = max(DataMatrix);
MaxValM = repmat(MaxVal,size(DataMatrix,1)-1,1);




SSC = sum(abs(XX) > (2.5/100).* MaxValM);

end




%%
% ---------------------------- Copyright Notice ---------------------------
% This file is part of BioPatRec © which is open and free software under 
% the GNU Lesser General Public License (LGPL). See the file "LICENSE" for 
% the full license governing this code and copyrights.
%
% BioPatRec was initially developed by Max J. Ortiz C. at Integrum AB and 
% Chalmers University of Technology. All authors’ contributions must be kept
% acknowledged below in the section "Updates % Contributors". 
%
% Would you like to contribute to science and sum efforts to improve 
% amputees’ quality of life? Join this project! or, send your comments to:
% maxo@chalmers.se.
%
% The entire copyright notice must be kept in this or any source file 
% linked to BioPatRec. This will ensure communication with all authors and
% acknowledge contributions here and in the project web page (optional).
%
% -------------------------- Function Description -------------------------
% funtion to add information about rest or no movement as an actuall
% movemen for training
%
% ------------------------- Updates & Contributors ------------------------
% [Contributors are welcome to add their email]
% 2011-xx-xx / Max Ortiz  / Creation
% 20xx-xx-xx / Author  / Comment on update


function sigTreated = AddRestAsMovement(sigTreated, recSession)

    sF      = recSession.sF;
    cT      = recSession.cT;
    rT      = recSession.rT;
    nR      = recSession.nR;
    nM      = recSession.nM;
    tdata   = recSession.tdata;
     
    % Collect the 50% to 75% of rest in between each contraction per each
    % movement
    for ex = 1 : nM
    tempdata =[];   
        for rep = 1 : nR
            % Samples of the exersice to be consider for training
            % (sF*cT*rep) Number of samples that takes a contraction
            % (sF*rT*rep) Number of samples that takes a relaxation
            is = fix((sF*cT*rep) + (sF*rT*.5) + (sF*rT*(rep-1)) + 1);
            fs = fix((sF*cT*rep) + (sF*rT*.75) + (sF*rT*(rep-1)));
            tempdata = [tempdata ; tdata(is:fs,:,ex)];
        end
        trData(:,:,ex) = tempdata;
    end
    
    % Gather the required amount of data for a movement
    % The rest data set is made with contributions from all movements rest
    % period
    totSamp = size(sigTreated.trData,1);
    sampXmov = fix(totSamp / sigTreated.nM);    
    sd = totSamp - sampXmov * sigTreated.nM;    %samples difference

%     if totSamp ~= sampXmov * sigTreated.nM;
%         disp(['"Rest" not fitted with ' num2str(sd) ' samples']);
%         errordlg(['"Rest" not fitted with ' num2str(sd) ' samples'],'Error');
%     end
    
    restData = [];
    %Using the first samples of each movement
    is = 1;
    fs = sampXmov;
    for ex = 1 : nM
        restData = [restData ; trData(is:fs,:,ex)];         
    end
    
    % If the rest data set wasn't completed from the information of all
    % rest periods, then it  willbe completed using the information from
    % the last rest period of the 1st movement
    if size(restData,1) ~= totSamp
        restData = [restData ; trData(end-sd+1:end,:,1)];
    end

    %Random selection of the sets to be use

%     trDL = size(trData,1);
%     while size(restData,1) ~= size(sigTreated.trData,1)
%         ex = fix(1 + (nM-1).*rand);
%         sOff = fix(0 + (trDL-sampXmov).*rand);
%         is = sOff + 1;
%         fs = sOff + sampXmov;
%         restData = [restData ; trData(is:fs,:,ex)];    
%     end
    
    sigTreated.nM = sigTreated.nM+1;
    sigTreated.mov(sigTreated.nM) = {'Rest'};
    sigTreated.trData(:,:,sigTreated.nM) = restData;
end

    
    %%
    
    
% ---------------------------- Copyright Notice ---------------------------
% This file is part of BioPatRec © which is open and free software under 
% the GNU Lesser General Public License (LGPL). See the file "LICENSE" for 
% the full license governing this code and copyrights.
%
% BioPatRec was initially developed by Max J. Ortiz C. at Integrum AB and 
% Chalmers University of Technology. All authors’ contributions must be kept
% acknowledged below in the section "Updates % Contributors". 
%
% Would you like to contribute to science and sum efforts to improve 
% amputees’ quality of life? Join this project! or, send your comments to:
% maxo@chalmers.se.
%
% The entire copyright notice must be kept in this or any source file 
% linked to BioPatRec. This will ensure communication with all authors and
% acknowledge contributions here and in the project web page (optional).
%
% -------------------------- Function Description -------------------------
% Function to compute Traning Data according to the contraction time
% percentage (cTp)
%
% ------------------------- Updates & Contributors ------------------------
% [Contributors are welcome to add their email]
% 20xx-xx-xx / Max Ortiz  / Creation
% 2011-07-19 / Max Ortiz  / Updated to consider cTp before and after
%                           contraction
% 20xx-xx-xx / Author  / Comment on update

function sigTreated = RemoveTransient_cTp(recSession, cTp)

    sF      = recSession.sF;
    cT      = recSession.cT;
    rT      = recSession.rT;
    nR      = recSession.nR;
    nM      = recSession.nM;
    tdata   = recSession.tdata;
    
    % New structured for the signal treated
    sigTreated      = recSession;
    sigTreated.cTp  = cTp;
    eRed            = (1-cTp)/2;    % effective reduction at the begining and at the end of contraction

    % Removed useless fields for following operations
    if isfield(sigTreated,'tdata')
        sigTreated = rmfield(sigTreated,'tdata');         
    end
    if isfield(sigTreated,'trdata')
        sigTreated = rmfield(sigTreated,'trdata');                 
    end
    
    for ex = 1 : nM
        tempdata =[];
        for rep = 1 : nR
            % Samples of the exersice to be consider for training
            % (sF*cT*(cTp-1)) Number of the samples that wont be consider for training
            % (sF*cT*rep) Number of samples that takes a contraction
            % (sF*rT*rep) Number of samples that takes a relaxation
            is = fix((sF*cT*(1-cTp-eRed)) + (sF*cT*(rep-1)) + (sF*rT*(rep-1)) + 1);
            fs = fix((sF*cT*(cTp+eRed)) + (sF*cT*(rep-1)) + (sF*rT*(rep-1)));
            tempdata = [tempdata ; tdata(is:fs,:,ex)];
        end
        trData(:,:,ex) = tempdata;
    end
    
    sigTreated.trData = trData;
end

%%



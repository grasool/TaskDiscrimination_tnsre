%% Plot EMG data

close all; clear; clc;

% Load a data file
load('SID1_UT_1DOF');

% How many movements
nMovements = recSession.nM;

% How many channels
nChannels = recSession.nCh;

% How many samples per channel
nSamples = size(recSession.tdata,1);

% Counter
Ctr = 1;

figure('units','normalized','outerposition',[0 0 1 1]);     % Maximize the figure

% time - x-axis for the plots
t = (1:nSamples)/recSession.sF;


for kk = 1:nChannels                                % Go to each channel
    for ii = 1:nMovements                           % Go to each movement

        EMGData = recSession.tdata(:,kk, ii);           % Get the data
        subplot(nChannels,nMovements, Ctr);             % Specify subplot
        plot(t, EMGData);                               % plot the data
        
        % Set some visual properties
        ax = gca;
        ax.YLim = [-0.5 0.5]; 
        ax.YTick=[];                                    % Turn off Y ticks
        
        
        if kk ~= nChannels
            ax.XTick=[];                                    % Turn off X Ticks for all but last row
        end
        
        if kk==1                                        % First row of plot only
            ax.Title.String = recSession.mov{ii};
        end
        
        if ii==1                                        % First Column of plots only
            ax.YLabel.String = kk;
        end
               
        Ctr = Ctr+1;
    end
end

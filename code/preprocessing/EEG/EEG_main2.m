clear;clc;
target_fs = 500; %Hz
raw_fs=512;
%% 1. prepare files
% 1.1 load data
data_path='E:\LeftRight_Hand_MI';
data_file = 's01.mat';
fid = [data_path, filesep, data_file];
load(fid);

fs = eeg.srate;
EEG_channels = 1:64;
EEG_data1 = eeg.movement_left; % there are multiple datasets in eeg, make sure use all of them. eg: eeg.movement_right
EEG_data1 = EEG_data1(EEG_channels, :);

EEG_data2 = eeg.movement_right; % there are multiple datasets in eeg, make sure use all of them. eg: eeg.movement_right
EEG_data2 = EEG_data2(EEG_channels, :);

EEG_data3 = eeg.imagery_left; % there are multiple datasets in eeg, make sure use all of them. eg: eeg.movement_right
EEG_data3 = EEG_data3(EEG_channels, :);

EEG_data4 = eeg.imagery_right; % there are multiple datasets in eeg, make sure use all of them. eg: eeg.movement_right
EEG_data4 = EEG_data4(EEG_channels, :);

EEG_data5 = eeg.rest; % there are multiple datasets in eeg, make sure use all of them. eg: eeg.movement_right
EEG_data5 = EEG_data5(EEG_channels, :);



% 1.2 prepare electrodes locations
load('biosemi_template.mat');
% check if the locations are the same to data description
topoplot([], locs, 'electrodes', 'ptslabels', 'plotdisk', 'on');

%% Use the method of ICA and IClabel


EEG_epochs1 = IC_label_artifact_removal( EEG_data1, fs, target_fs,locs,EEG_channels);
EEG_epochs2 = IC_label_artifact_removal( EEG_data2, fs, target_fs,locs,EEG_channels);
EEG_epochs3 = IC_label_artifact_removal( EEG_data3, fs, target_fs,locs,EEG_channels);
EEG_epochs4 = IC_label_artifact_removal( EEG_data4, fs, target_fs,locs,EEG_channels);
EEG_epochs5 = IC_label_artifact_removal( EEG_data5, fs, target_fs,locs,EEG_channels);

EEG_epochsall=[EEG_epochs1;EEG_epochs2;EEG_epochs3;EEG_epochs4;EEG_epochs5;]


%% 3. visual check
EEG_epochsall = visual_check(EEG_epochsall, target_fs);


%% save

output = 'C:\DL_denoising_data_cut\EEG_output';
save([output, filesep, 'EEG_epochs_s01','.mat'], 'EEG_epochsall');
%writeNPY(EEG_epochs, [output, filesep, 'EEG_epochs_Cz', char(film_list(film_id)),'.npy']);



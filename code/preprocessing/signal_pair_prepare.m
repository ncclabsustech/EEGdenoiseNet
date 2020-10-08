

data_path =  'C:\EEG_EEGN\EOG_all_4514.mat';%%%%%∑≈»Îc≈Ãº¥ø…

EEG_epochs_512hz = resample(EEG_epochs', 512, 500)';

%EOG_all_epochs = preprocessed_EOG;
%% Randomly scrambled data

randIndex = randperm(size(EEG_epochs_512hz,1));
EEG_epochs_512hz_random_4535 = EEG_epochs_512hz(randIndex,:);
EEG_epochs_512hz_random_5598 = [EEG_epochs_512hz_random_4535(1:1063,:); EEG_epochs_512hz_random_4535];
EEG_epochs_512hz = EEG_epochs_512hz(1:4514,:);
%EEG_4514_random_500hz = EEG_4514_random;
%% Store as .Mat & .npy file
fs = 256
output='C:\EEGdenoiseNet\data';
save([output, filesep, 'EMG_all_epochs.mat'], 'EMG_all_epochs','fs');
writeNPY(EEG_epochs_512hz, [output, filesep, 'EEG_all_epochs_512hz.npy']);
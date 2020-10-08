function [ EEG_epochs] = IC_label_artifact_removal( EEG_data, fs, target_fs,locs,EEG_channels)

% to use ICLabel, EEGlab is needed. In addition, you need to manully
% install the ICLabel plugin. To do this, typein eeglab in Matlab, select
% 'File' > 'Manage EEGlab extensions' > 'DataProcessing extensions', tick
% Instalall of ICLabel, then press Ok. ICLabel will start to download in
% back ground. You need to wait untill it is fully downloaded, and then you
% can use it for artifact removal



%% 2. rereference
% ICLabel needs average re-reference. The dataset is already in average
% re-freference

%% 3. filter and detrend
lp = 1;
hp = 200; % please not this freq should not be larger than half of fs
order = 6600;
powerline_freq = 60; % in this data, the powerline is 60Hz
EEG_data = filter_notch(EEG_data, fs, powerline_freq);
EEG_data = filter_data(EEG_data, fs, 1, 80, order);
EEG_data = detrend(EEG_data')';

%% 4. ICA decomposition using fast ICA
% 3.1 pca withening
[dewhitening, score, latent]=pca(EEG_data','Centered','off');
whitening=inv(dewhitening);
PC=whitening*EEG_data;
latent=100*latent/sum(latent);
n_pcs=sum(latent>0.01*mean(latent));
retained_variance=sum(latent(1:n_pcs));

[IC, mixing, unmixing]=fastica(PC(1:n_pcs, :),'approach','defl','g','tanh','maxNumIterations',500); % here you can visually check the ICs
IC=unmixing*PC(1:n_pcs,:);
mixing_matrix = dewhitening(:,1:n_pcs)*mixing;
unmixing_matrix = unmixing*whitening(1:n_pcs,:);

%% 4. Use ICLabel
% prepare EEGlab struct
data_len = size(EEG_data,2);
time_len = data_len/fs;
time_axis = linspace(0,time_len, data_len);
EEG_struct.times = time_axis;
EEG_struct.data = EEG_data;
EEG_struct.chanlocs = locs;
EEG_struct.srate = fs;
EEG_struct.trials = 1;
EEG_struct.pnts = data_len;
EEG_struct.icawinv = mixing_matrix;
EEG_struct.icaweights = unmixing_matrix;
EEG_struct.icaact = unmixing_matrix*EEG_data;
EEG_struct.icachansind = EEG_channels;
EEG_struct.ref = 'averef';
% classification
EEG_struct= iclabel(EEG_struct);
class_prob = EEG_struct.etc.ic_classification.ICLabel.classifications;
class_type = EEG_struct.etc.ic_classification.ICLabel.classes;
%% 7. reconstruct cleaned EEG and save to D
ic_num = size(class_prob,1);
for iter_ic = 1:ic_num
        [max_prob_val(1, iter_ic), class_index(1, iter_ic)] = max(class_prob(iter_ic, :));
end
class_label.brain_ic_index = sort(find(class_index == 1));
class_label.muscle_ic_index = sort(find(class_index == 2));
class_label.eye_ic_index = sort(find(class_index == 3));
class_label.heart_ic_index = sort(find(class_index == 4));
class_label.line_noise_ic_index = sort(find(class_index == 5));
class_label.channel_noise_ic_index = sort(find(class_index == 6));
class_label.other_noise_ic_index = sort(find(class_index == 7)); % what is other, should I remove it?
    
brain_ic_num = length(class_label.brain_ic_index);
if(brain_ic_num == 0)
        warning('No brain IC recovered! Please check')
end
    
bad_ic_index = unique(sort([class_label.muscle_ic_index, class_label.eye_ic_index, class_label.heart_ic_index, class_label.line_noise_ic_index, class_label.channel_noise_ic_index, class_label.other_noise_ic_index]));
    
EEG_data_processed = EEG_data - mixing_matrix(:, bad_ic_index)*IC(bad_ic_index,:);

EEG_data_processed  = double(EEG_data_processed );
EEG_data_processed = resample(EEG_data_processed', target_fs, fs);
EEG_data_processed = EEG_data_processed';
t_num = size(EEG_data_processed,2);
good_data_num = idivide(t_num,int32(1000));



EEG_epochs = [];
for iter_cut = 1:good_data_num
    
    start = iter_cut*1000-999;
    
    EEG_epochs = [EEG_epochs; EEG_data_processed(:, start:start+999)];
        
    
end

%% 4. remove bad epochs according to PSD template
epoch_num = size(EEG_epochs, 1);
corr_threshold = 0.8;
frequencies = 1:120;
template_PSD = exp(-1*frequencies/20);
template_PSDs = repmat(template_PSD', 1, epoch_num);

[PSDs, f] = pwelch(EEG_epochs', [], [], frequencies, target_fs);
PSD_corr = corr(PSDs, template_PSDs);
PSD_corr = PSD_corr(:,1);
EEG_epochs(PSD_corr < corr_threshold, :) = []; % remove those are not look like template
PSDs(:, PSD_corr < corr_threshold) = [];

high_freq = 40:120;
power_all = sum(PSDs, 1);
high_power = sum(PSDs(high_freq,:), 1);
ratio = high_power./power_all;
EEG_epochs(ratio > 0.2,: ) = []; % when there's too much high frequency, it might be contaminated by EMG
PSDs(:, ratio >0.2) = [];

%% 4. remove bad EMGs according to PSD
ratio_threshold = 70; % low_freq/high_freq < 1, the more higher frequency, the better the EMG signal
frequencies = 1:200;

low_freq = 1:30;
high_freq = 40:200;

[PSDs, f] = pwelch(EEG_epochs', [], [], frequencies, target_fs);
low_powers = mean(PSDs(low_freq,:), 1);
high_powers = mean(PSDs(high_freq,:), 1);
ratios = low_powers./high_powers;

EEG_epochs(ratios<ratio_threshold, :) = []; % remove those are not look like template



    
function [ EMG_epochs, target_fs ] = EMG_cut( EMG_signal, raw_fs, target_fs, raw_unit)
%EMG_CUT Summary of this function goes here
%   Detailed explanation goes here

    %% 1. configurations
    %target_fs = 500;
    preprocess_filter_range = [1, 120]; %Hz
    %preprocess_filter_range(2) = round(target_fs/2 -10);
    detection_filter_range_1 = [50, 120]; %Hz
    detection_filter_range_2 = 8; %Hz
    filter_order = 22528;

    cut_range = [-1,1]; %s output epochs range
    random_window_location = 'on';
    random_range = 1; %s shift the cut window to left or right less than 0.5s randomly
	
    scale_method = 1;
    normalized_threshold = 0.3;
    
    %scale_mehtod = 2;
    %normalized_threshold = 3;
    
    
    %% 2. preprocess
    % 2.1 filter
    preprocessed_EMG=filter_data(EMG_signal,raw_fs,preprocess_filter_range(1),preprocess_filter_range(2),filter_order);
    % 2.2 resample
    preprocessed_EMG = double(preprocessed_EMG);
    preprocessed_EMG = resample(preprocessed_EMG, target_fs, raw_fs);

    %% 3. prepare for detection
    detection_EMG = filter_data(preprocessed_EMG, target_fs, detection_filter_range_1(1), detection_filter_range_1(2),filter_order); % band pass
    detection_EMG = double(detrend(detection_EMG));
    if scale_method
       [pks, ~]= findpeaks(abs(detection_EMG));
        pks = sort(pks, 'descend');
        max_val = prctile(pks, 95);
        scaled_detection_EMG = detection_EMG./max_val;
    else
        scaled_detection_EMG = (detection_EMG - mean(detection_EMG))./std(detection_EMG);
    end
    %% 3. detect, cut and save
    % 3.1 threshold
    scaled_detection_EMG = abs(scaled_detection_EMG);
    scaled_detection_EMG = filter_data(scaled_detection_EMG, target_fs, [], detection_filter_range_2,filter_order); % low pass to get envelope
    scaled_detection_EMG = double(scaled_detection_EMG);
    scaled_detection_EMG(scaled_detection_EMG < normalized_threshold) = 0;
    % 3.2 detect
    [~, EMG_max_locs] = findpeaks(scaled_detection_EMG,'MinPeakDistance',500);
    EMG_cut_window = cut_range.*target_fs;
    EMG_cut_window(2) = EMG_cut_window(2) - 1;
    sample_num = length(detection_EMG);
    EMG_max_locs(EMG_max_locs <= 2*abs(EMG_cut_window(1))) = [];
    EMG_max_locs(EMG_max_locs >= (sample_num - 2*EMG_cut_window(2))) = [];
    
    
    epoch_num = length(EMG_max_locs);
    epoch_len = length(EMG_cut_window(1):EMG_cut_window(2));
    epoch_start = EMG_max_locs + EMG_cut_window(1);
    epoch_stop = EMG_max_locs + EMG_cut_window(2);
    % 3.3 randomly shift cut window if configured
    if(strcmp(random_window_location, 'on'))
        random_range_pt = random_range*target_fs;
        rand_shift = randi([-1*random_range_pt, random_range_pt], 1, epoch_num);
        epoch_start = epoch_start + rand_shift;
        epoch_stop = epoch_stop + rand_shift;
    end
    % cut epochs
    EMG_epochs = nan(epoch_num, epoch_len);
    for iter_epoch = 1:epoch_num
        EMG_epochs(iter_epoch, :) = preprocessed_EMG(epoch_start(iter_epoch):epoch_stop(iter_epoch));
    end
    %% 4. remove bad EMGs according to PSD
	ratio_threshold = 3.5; % low_freq/high_freq < 1, the more higher frequency, the better the EMG signal
    frequencies = 1:120;
    
    low_freq = 1:30;
    high_freq = 40:120;
    
    [PSDs, f] = pwelch(EMG_epochs', [], [], frequencies, target_fs);
    low_powers = mean(PSDs(low_freq,:), 1);
    high_powers = mean(PSDs(high_freq,:), 1);
    ratios = low_powers./high_powers;
    
    EMG_epochs(ratios>ratio_threshold, :) = []; % remove those are not look like template
    
    %% 5. convert to uV
    c = nan;
    switch raw_unit
        case 'uV'
            c = 1;
        case 'mV'
            c = 1000;
        case 'V'
            c = 1000000;
    end
    if(isnan(c))
        error('undefined input data unit!');
    else
        EMG_epochs = EMG_epochs.*c;
    end
end


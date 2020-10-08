function [ EOG_epochs, target_fs ] = EOG_cut( eog_signal, raw_fs, target_fs, raw_unit)
%EOG_CUT Summary of this function goes here
%   Detailed explanation goes here

    %% 1. configurations
    output_filtering = 'off'; % low pass filter for output data
    
    preprocess_filter_range = [0.3, 80]; %Hz
    detection_filter_range = [0.5, 8]; %Hz
    filter_order = 22528;
    output_filter_range = 20; % lowpass cutoff f=12 Hz
    cut_range = [-1,1]; %s output epochs range
    random_window_location = 'on';
    random_range = 0.8; %s shift the cut window to left or right less than 0.5s randomly
	notch_frequency = 50;
    scale_method = 1;
    normalized_threshold = 3;
    
    %scale_mehtod = 2;
    %normalized_threshold = 3;
    
    
    %% 2. preprocess
    % 2.1 filter
    preprocessed_eog = filter_notch(eog_signal, raw_fs, notch_frequency); 
    preprocessed_eog=filter_data(preprocessed_eog,raw_fs,preprocess_filter_range(1),preprocess_filter_range(2),filter_order);
    % 2.2 resample
    preprocessed_eog = double(preprocessed_eog);
    preprocessed_eog = resample(preprocessed_eog', target_fs, raw_fs)';
    preprocessed_eog = double(detrend(preprocessed_eog));
    % 2.3 filter for output data if condigured
    if(strcmp(output_filtering, 'on'))
        output_eog = filter_data(eog_signal, target_fs, [], output_filter_range,filter_order);
        output_eog = double(output_eog);
    else
        output_eog = preprocessed_eog;
    end

    %% 3. prepare for detection
    detection_eog = filter_data(preprocessed_eog, target_fs, detection_filter_range(1), detection_filter_range(2),filter_order);
    detection_eog = double(detrend(detection_eog));
    if scale_method
       [pks, ~]= findpeaks(abs(detection_eog));
        pks = sort(pks, 'descend');%按升序对 pks 进行排序。
        max_val = prctile(pks, 95);%第百分之95的数
        scaled_detection_eog = detection_eog./max_val;
    else
        scaled_detection_eog = (detection_eog - mean(detection_eog))./std(detection_eog);
    end
    %% 3. detect, cut and save
    % 3.1 threshold
    scaled_detection_eog = abs(scaled_detection_eog);
    scaled_detection_eog(scaled_detection_eog < normalized_threshold) = 0;
    % 3.2 detect
    [~, eog_max_locs] = findpeaks(scaled_detection_eog,'MinPeakDistance',500);% ,'MinPeakHeight',80);
    eog_cut_window = cut_range.*target_fs;
    eog_cut_window(2) = eog_cut_window(2) - 1;%-500到499
    sample_num = length(detection_eog);
    eog_max_locs(eog_max_locs <= 2*abs(eog_cut_window(1))) = []; %小于1000的位置不要
    eog_max_locs(eog_max_locs >= (sample_num - 2*eog_cut_window(2))) = [];
    
    
    epoch_num = length(eog_max_locs);
    epoch_len = length(eog_cut_window(1):eog_cut_window(2));
    epoch_start = eog_max_locs + eog_cut_window(1);
    epoch_stop = eog_max_locs + eog_cut_window(2);
    % 3.3 randomly shift cut window if configured
    if(strcmp(random_window_location, 'on'))
        random_range_pt = random_range*target_fs;
        rand_shift = randi([-1*random_range_pt, random_range_pt], 1, epoch_num);
        epoch_start = epoch_start + rand_shift;
        epoch_stop = epoch_stop + rand_shift;
    end
    % cut epochs
    EOG_epochs = nan(epoch_num, epoch_len);
    for iter_epoch = 1:epoch_num
        EOG_epochs(iter_epoch, :) = output_eog(epoch_start(iter_epoch):epoch_stop(iter_epoch));
    end
    
    %% 4. convert to uV
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
        EOG_epochs = EOG_epochs.*c;
    end
end


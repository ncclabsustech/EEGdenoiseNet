function [ EEG_epochs, target_fs ] = EEG_cut( EEG_signal, fp1_fp2_channels, raw_fs, target_fs, raw_unit, notch_frequency)
%EEG_CUT Summary of this function goes here
%   Detailed explanation goes here
    

    %% 1. configurations
    preprocess_filter_range = [0.3, 120]; %Hz
    cut_duration = 2; %s output epochs range
    cut_length = cut_duration*target_fs;
    filter_order = 22528;
    scale_method = 1;
    threshold = 0.2;
    
    %% 2. preprocess
    % 2.1 filter
    preprocessed_EEG = filter_notch(EEG_signal, raw_fs, notch_frequency); 
    preprocessed_EEG=filter_data(preprocessed_EEG,raw_fs,preprocess_filter_range(1),preprocess_filter_range(2), filter_order);
    % 2.2 resample
    preprocessed_EEG = double(preprocessed_EEG);
    preprocessed_EEG = resample(preprocessed_EEG', target_fs, raw_fs);
    preprocessed_EEG = preprocessed_EEG';
    preprocessed_EEG = double(detrend(preprocessed_EEG));
    
    %% 3. find the good parts according to fp1 and fp2
    fp1_fp2_signal = abs(preprocessed_EEG(fp1_fp2_channels,:));
    fp1_fp2_signal = filter_data(fp1_fp2_signal, target_fs, [], 5, filter_order);
    fp1_fp2_signal = double(mean(fp1_fp2_signal,1));
    if scale_method
       [pks, ~]= findpeaks(fp1_fp2_signal);
        pks = sort(pks, 'descend');
        max_val = prctile(pks, 95);
        fp1_fp2_signal = fp1_fp2_signal./max_val;
    else
        fp1_fp2_signal = (fp1_fp2_signal - mean(detection_eog))./std(detection_eog);
    end
    good_data_mask = fp1_fp2_signal;
    good_data_mask(good_data_mask >= threshold) = 0;
    good_data_mask(good_data_mask ~= 0) = 1;
    diff_good_data_mask = diff(good_data_mask);
    good_data_start_points = find(diff_good_data_mask == 1) + 1;
    good_data_stop_points = find(diff_good_data_mask == -1);
    
    good_data_stop_points(good_data_stop_points <= good_data_start_points(1)) = [];
    good_data_start_points(good_data_start_points >= good_data_stop_points(end)) = [];
    len = min([length(good_data_start_points), length(good_data_stop_points)]);
    good_data_start_points = good_data_start_points(1:len);
    good_data_stop_points = good_data_stop_points(1:len);
    good_data_len = good_data_stop_points - good_data_start_points;
    good_data_start_points(good_data_len < cut_length) = [];
    good_data_stop_points(good_data_len < cut_length) = [];
    good_data_len(good_data_len < cut_length) = [];
    good_data_num = length(good_data_start_points);
    %% 3. cut EEG
    
    EEG_epochs = [];
    for iter_cut = 1:good_data_num
        tmp_good_len = good_data_len(iter_cut);
        cut_num = floor(tmp_good_len/cut_length);
        residual = tmp_good_len - cut_num*cut_length;
        if(residual > 2)
            shift = floor(residual/2);
        else
            shift = 0;
        end
        
        start = good_data_start_points(iter_cut) + shift;
        for iter_sub_cut = 1:cut_num
            EEG_epochs = [EEG_epochs; preprocessed_EEG(3:7, start:start+cut_length-1)];
            start = start+cut_length;
        end
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
        EEG_epochs = EEG_epochs.*c;
    end
            
    
end


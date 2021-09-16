function [ EOG_segments, target_fs ] = EOG_cut( eog_signal, raw_fs, target_fs, raw_unit)
%EOG_CUT Summary of this function goes here
%   Detailed explanation goes here

    %% 1. configurations
   
    
    preprocess_filter_range = [0.3, 80]; %Hz
    output_filter_range = [0.3, 12];
    notch_frequency = 50;
    filter_order = 22528;
   
    
    cut_range = [-1,1]; %s output epochs range
	
    r_square_thresh = 0.6;
    
    
    %% 2. preprocess
    % 2.1 filter
    eog_for_detection = filter_notch(eog_signal, raw_fs, notch_frequency); 
    eog_for_detection=filter_data(eog_for_detection,raw_fs,preprocess_filter_range(1),preprocess_filter_range(2),filter_order);
    % 2.2 resample
    eog_for_detection = double(eog_for_detection);
    eog_for_detection = resample(eog_for_detection', target_fs, raw_fs)';
    eog_for_detection = double(detrend(eog_for_detection));
    % 2.3 filter for output data if condigured
    output_eog = filter_data(eog_for_detection, target_fs, [], output_filter_range,filter_order);
	output_eog = double(output_eog);
    

    %% 3. cut segments
    segment_len = (cut_range(2) - cut_range(1))*target_fs;
    sample_num = length(eog_for_detection);
    
    segment_num = floor(sample_num/segment_len);
    
    
    cut_locs = randi([target_fs+1, sample_num-target_fs], 1, segment_num);
    
    
    segments_for_detection = nan(segment_num, segment_len);
    EOG_segments = nan(segment_num, segment_len);
    
    g = fittype({'1/x','1'});
    frq_interval=[2 79];%frq_interval=[2 48];
    nfft = 1024;
    df = [1 100];
    for iter_segment = 1:segment_num
        start_range = cut_locs(iter_segment) + cut_range(1)*target_fs;
        stop_range = cut_locs(iter_segment) + cut_range(2)*target_fs;
        range = start_range+1:stop_range;
        
        segments_for_detection(iter_segment, :) = eog_for_detection(range);
        EOG_segments(iter_segment, :) = output_eog(range);
        [freq, sp] = my_PSD(EOG_segments(iter_segment, :),nfft,target_fs,@hanning,50,df,'off');
        vec = find(freq > frq_interval(1) & freq < frq_interval(2));
        [x,f] = fit(freq(vec),sp(vec),g);
        r_square(iter_segment) = f.rsquare;
    end
    
    %% 4. preliminary exclusion based on 1/f fit
    segment_to_remove = find(r_square<=r_square_thresh);
    EOG_segments(segment_to_remove, :) = [];
    
    
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
        EOG_segments = EOG_segments.*c;
    end
end

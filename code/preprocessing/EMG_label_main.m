target_fs = 512; %Hz
%% 1. configuration
%% 这里填写数据路径，需要修改
data_path =  'C:\DL_denoising_data_cut\Mimetic_Interfaces-Facial_Surface_EMG_Dataset_2015\Data';
%%

subject_list = 1:15;
condition_list = {'A', 'B'};
raw_fs = 2048;
raw_data_unit = 'mV'; % 'uV', 'mV', 'V'


%% 2. start processing
subject_num = length(subject_list);
condition_num = length(condition_list);
fid_template = [data_path, filesep, 'data%02d%s.mat'];

epoch_all = [];
for iter_subject = subject_list(1):subject_list(subject_num)
    for iter_condition = 1:condition_num
        % load data
        fid = [data_path, filesep, 'data', num2str(iter_subject, '%02d'), condition_list{iter_condition}, '.mat']
        load(fid);
        EMGs = D.Eraw;
        [sample_num, channel_num] =size(EMGs);
        for iter_channel = 1:channel_num
            tmp_EMG_signal = EMGs(:,iter_channel)';
            %% 2. cut file
            [tmp_EMG_epochs, new_fs] = EMG_cut(tmp_EMG_signal, raw_fs, target_fs, raw_data_unit);
            epoch_all = [epoch_all; tmp_EMG_epochs];
        end
    end
end
%% 3. visual check
EMG_epochs = visual_check(epoch_all, target_fs);

fs = new_fs;
output = 'C:\DL_denoising_data_cut\EMG_output';
if(~exist(output, 'dir'))
    mkdir(output);
end

save([output, filesep, 'EMG_epochs_new.mat'], 'EMG_epochs', 'fs');
writeNPY(EMG_epochs, [output, filesep, 'EMG_epochs_new.npy']);
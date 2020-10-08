%clear; close all;
target_fs = 500; %Hz

data_path = 'E:\new_eog_eeg';

subject_list = 1:20;

raw_data_unit = 'uV'; % 'uV', 'mV', 'V';
subject_num = length(subject_list);
%condition_list = {'Trial01', 'Trial02', 'Trial03'};
condition_num = 4;
raw_fs = 250;
film_list = {'S01.mat', 'S02.mat'};
film_num = length(film_list);
signal = data{1, 2}.X;
eog=signal(:,31)-signal(:,30);
eog=double(eog');
p = plot(eog(50000:100000))
p(1).LineWidth = 0.5;


%% 1. load data and cut
vertical_EOG_epochs = [];
% for iter_subject = 1:film_num
%     file_id= [data_path, filesep, char(film_list(iter_subject))];
%     load(file_id);   
%         %subject_data = EOG_eyblink.(['Sub', num2str(subject_list(iter_subject), '%S02d')]);
%         for iter_condition = 1:condition_num
            %condition_data = subject_data.(condition_list{iter_condition});
            
            EEG_EOG = EEG.data;
            v_EOG = EEG_EOG(24,:)-(EEG_EOG(23,:)+EEG_EOG(25,:))*0.5;
            h_EOG = EEG_EOG(23,:)-EEG_EOG(25,:);
            eog_signal =   v_EOG  ;
            [tmp_vertical_EOG_epochs, new_fs] = EOG_cut(eog_signal, raw_fs, target_fs, raw_data_unit);
            
            vertical_EOG_epochs = [vertical_EOG_epochs; tmp_vertical_EOG_epochs];
%         end
% end
%% 3. visual check
vertical_EOG_epochs = visual_check(vertical_EOG_epochs, target_fs);

fs = new_fs;

output = 'C:\DL_denoising_data_cut\EOG_output\bci';
if(~exist(output, 'dir'))
    mkdir(output);
end
save([output, filesep, 'bci_h_EOG_epochs_9E.mat'], 'vertical_EOG_epochs', 'fs');
%writeNPY(vertical_EOG_epochs, [output, filesep, 'bci_h_EOG_epochs_5E.npy']);

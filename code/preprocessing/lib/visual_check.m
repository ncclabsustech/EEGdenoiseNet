function [ good_epochs ] = visual_check( epochs, fs)
%VISUAL_CHECK Summary of this function goes here
%   Detailed explanation goes here
    
     %% Initiations
    [epoch_num, epoch_len ]= size(epochs);
    time_start = 0;
    time_end = epoch_len/fs;
    time_axis = linspace(time_start, time_end, epoch_len);

    frequencies = 1:120;
    global plot_data;
    global my_handles;
    
    plot_data.epochs = epochs;
    plot_data.time_axis = time_axis;
    plot_data.epoch_num = epoch_num;
    plot_data.bad_epochs_list = [];
    plot_data.bad_epoch_num = 0;
    plot_data.epoch_index = 1;
    plot_data.epoch_info = sprintf('Epoch %d of %d, labeled as %s', plot_data.epoch_index, plot_data.epoch_num, is_epoch_good);
    
    %% calculate psd for all epochs
    [plot_data.PSDs, plot_data.frequency_axis] = pwelch(epochs', [], [], frequencies, fs);
    plot_data.PSDs = plot_data.PSDs';
    
    %% put ui components
    my_handles.main_window = figure;
    my_handles.main_window.Color = [1,1,1];
    figure_width = 1000;
    figure_height = 600;
    my_handles.main_window.InnerPosition=[0,0,figure_width, figure_height];
    
    margin_width = 5;
    ui_size_width = 50;
    ui_size_height = 30;
    position = [0,0, ui_size_width,ui_size_height];
    my_handles.left_button = uicontrol('Style', 'pushbutton', 'String', '<-', 'FontSize', 14, 'FontWeight', 'Bold', 'Position', position, 'BackgroundColor', [0.8078, 0.3725, 0.3725],'ForegroundColor', [1,1,1],'Callback', @cbc_left_btn);
    
    position(1) = position(1) + ui_size_width + margin_width;
    my_handles.text_edit = uicontrol('Style', 'edit','String', '1', 'FontSize', 14, 'Position', position);
    
    position(1) = position(1) + ui_size_width + margin_width;
    my_handles.go_button = uicontrol('Style', 'pushbutton','String', 'Go', 'FontSize', 14, 'FontWeight', 'Bold', 'Position', position, 'BackgroundColor', [0.8078, 0.3725, 0.3725],'ForegroundColor', [1,1,1],'Callback', @cbc_go_btn);
    
    position(1) = position(1) + ui_size_width + margin_width;
    my_handles.right_button = uicontrol('Style', 'pushbutton','String', '->', 'FontSize', 14, 'FontWeight', 'Bold', 'Position', position, 'BackgroundColor', [0.8078, 0.3725, 0.3725],'ForegroundColor', [1,1,1],'Callback', @cbc_right_btn);
    
    position(1) = position(1) + 4*ui_size_height + margin_width;
    position(3) = 4*ui_size_width;
    my_handles.label_as_bad_button = uicontrol('Style', 'pushbutton','String', 'Label as bad', 'FontSize', 14, 'FontWeight', 'Bold', 'Position', position, 'BackgroundColor', [0.8078, 0.3725, 0.3725],'ForegroundColor', [1,1,1],'Callback', @cbc_label_as_bad_btn);
    
    position(1) = position(1) + 8*ui_size_height + margin_width;
    position(3) = 4*ui_size_width;
    my_handles.finish_button = uicontrol('Style', 'pushbutton','String', 'Finish Label', 'FontSize', 14, 'FontWeight', 'Bold', 'Position', position, 'BackgroundColor', [0.8078, 0.3725, 0.3725],'ForegroundColor', [1,1,1],'Callback', @cbc_finish_label_btn);
    
    
    %% refresh plot
    refresh_plot();
    
    %% wait for return
    uiwait(my_handles.main_window);
    
    %% return updated epochs
    good_epochs = epochs;
    good_epochs(plot_data.bad_epochs_list,:) = [];
    disp('finished');
end




%% private functions
%% callbacks
function cbc_left_btn(hObject, event, handles)
        global plot_data;
        if(plot_data.epoch_index > 1)
            plot_data.epoch_index = plot_data.epoch_index - 1;
        end
        plot_data.epoch_info = sprintf('Epoch %d of %d, labeled as %s', plot_data.epoch_index, plot_data.epoch_num, is_epoch_good);
        refresh_plot();
        
end
function cbc_right_btn(hObject, event, handles)
        global plot_data;
        if(plot_data.epoch_index < plot_data.epoch_num)
            plot_data.epoch_index = plot_data.epoch_index + 1;
        end
        plot_data.epoch_info = sprintf('Epoch %d of %d, labeled as %s', plot_data.epoch_index, plot_data.epoch_num, is_epoch_good);
        refresh_plot();
end
function cbc_go_btn(hObject, event, handles)
        global plot_data;
        global my_handles;
        go_to_index = str2num(my_handles.text_edit.String);
        if( go_to_index > plot_data.epoch_num | go_to_index < 1)
            go_to_index = 1;
        end
        plot_data.epoch_index = go_to_index;
        plot_data.epoch_info = sprintf('Epoch %d of %d, labeled as %s', plot_data.epoch_index, plot_data.epoch_num, is_epoch_good);
        refresh_plot();
end
function cbc_label_as_bad_btn(hObject, event, handles)
    global plot_data;
    plot_data.bad_epochs_list = [plot_data.bad_epochs_list plot_data.epoch_index];
    plot_data.bad_epochs_list = unique(plot_data.bad_epochs_list);
    plot_data.bad_epoch_num = length(plot_data.bad_epochs_list);
    plot_data.epoch_info = sprintf('Epoch %d of %d, labeled as %s', plot_data.epoch_index, plot_data.epoch_num, is_epoch_good);
    refresh_plot();
    
    disp(['a bad epoch was just labeld: epoch ', num2str(plot_data.epoch_index)]);
end

function cbc_finish_label_btn(hObject, event, handles)
    global my_handles
    close(my_handles.main_window);
end

%% internal functions
    function [] = refresh_plot()
        %% plot in time field
        global plot_data;
        global my_handles;
        
        time_axis = plot_data.time_axis;
        epoch = plot_data.epochs(plot_data.epoch_index,:);
        subplot(1,2,1);
        plot(time_axis, epoch);
        xlabel('Time (s)');
        ylabel(['epoch ', num2str(plot_data.epoch_index)]);
        title('Epoch in time field');
        my_handles.text_edit.String = num2str(plot_data.epoch_index);
        
        %% plot PSD
        subplot(1,2,2);
        frequency_axis = plot_data.frequency_axis;
        PSD = plot_data.PSDs(plot_data.epoch_index, :);
        area(frequency_axis, PSD, 'EdgeColor', [0.1,0.6,0.1], 'FaceColor', [0.1,0.6,0.1], 'FaceAlpha', 0.4);
        xlabel('Frequency (Hz)');
        title('Epoch in frequency field');
        
        %sgtitle(plot_data.epoch_info);
    end
    
    function [tag] = is_epoch_good()
        global plot_data;
        bad_flag = ismember(plot_data.epoch_index, plot_data.bad_epochs_list);
        if(bad_flag)
            tag = 'bad';
        else
            tag = 'good';
        end
        
    end
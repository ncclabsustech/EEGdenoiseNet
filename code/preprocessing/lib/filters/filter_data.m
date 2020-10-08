function filtered_data=filter_data(sensor_data,Fs,hp,lp, filter_order)

EEG.data=single(sensor_data);
EEG.pnts = size(EEG.data,2);
EEG.srate=Fs;
EEG.nbchan= size(EEG.data,1);
EEG.trials=1;
EEG.xmin=0;
EEG.xmax=max(EEG.data(:));
EEG.times= 1:size(EEG.data,2);
EEG.event=[];
newEEG=filter_data_sub_function(EEG,hp,lp,filter_order);
filtered_data=newEEG.data;
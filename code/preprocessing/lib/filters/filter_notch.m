
%%
function data_out=filter_notch(data_in,Fs, notch_frequency)

% Making sure the number of rows are bigger, because the filtering
% operation happens columnwise(time domain)
if size(data_in,2)>size(data_in,1)
    data_out=data_in';
else
    data_out=data_in;
end

%These are the stop band limits
power1 = [notch_frequency-0.05, notch_frequency+0.05];
power2=[notch_frequency*2-0.05, notch_frequency*2+0.05];


% for 50 Hz noise

if Fs>100 %Added to check that sampling rate is high enough for filtering MB 2018-01-04

    %We design a 1st order cheby filter with 10 db attenuation in stop band
    [b,a]=cheby2(1,10,power1*2/Fs,'stop');
    data_out=filter(b,a,data_out);
    
    %We flip the data and do the filtering again, to produce zero phase
    %distortion
    data_out=flipud(data_out);
    data_out=filter(b,a,data_out);
    data_out=flipud(data_out);
    
    
    % - Additional attenuation
    [b,a]=cheby2(1,10,power1*2/Fs,'stop');
    data_out=filter(b,a,data_out);
    data_out=flipud(data_out);
    data_out=filter(b,a,data_out);
    data_out=flipud(data_out);

end

% for 100 Hz noise

if Fs>200 %Added to check that sampling rate is high enough for filtering MB 2018-01-04
    
    %We design a 1st order cheby filter with 10 db attenuation in stop band
    [b,a]=cheby2(1,10,power2*2/Fs,'stop');
    data_out=filter(b,a,data_out);
    
    %We flip the data and do the filtering again, to produce zero phase
    %distortion
    data_out=flipud(data_out);
    data_out=filter(b,a,data_out);
    data_out=flipud(data_out);
    
    
    % - Additional attenuation
    [b,a]=cheby2(1,10,power1*2/Fs,'stop');
    data_out=filter(b,a,data_out);
    data_out=flipud(data_out);
    data_out=filter(b,a,data_out);
    data_out=flipud(data_out);
    
end

%%%%%%%%%%



if size(data_in,2)>size(data_in,1)
    data_out=data_out';
end
%%
% Revision history:
%{
2014-04-13 
    v0.1 Updated the file based on initial versions from Dante
(Revision author : Sri).
   

%}
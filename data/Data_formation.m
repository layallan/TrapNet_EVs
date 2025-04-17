clc
clear
% This program is trying to load data from different dirction in the
% computer.
% Gather them together and make them into the same formation is the rule.
% 'data.mat' is the result of this program, which consist all the data from
% three types of proteins. To switch the data we want to use, change the
% data file in the certain direction can work well.
%% path of different types of proteins
disp('[On] Porgram')
path_MDA = 'dataset/train/MDA/';
path_MCF7A = 'dataset/train/MCF7A/';
path_MCF10A = 'dataset/train/MCF10A/';

%% get the data from different protein files
disp('[Start] Loading data');
length_stand = zeros(1,3);
[data_MDA,length_stand(1)] = F_DataFromDir(path_MDA);
[data_MCF7A,length_stand(2)] = F_DataFromDir(path_MCF7A);
[data_MCF10A,length_stand(3)] = F_DataFromDir(path_MCF10A);
disp('[Done] Loading data');
%% data length standardlization and decrease sampling rate
disp('[Start] Data standarlization')
% down sampling parameters setting
% ********************************
% ********************************
target_sampling_rate = 10; %500;
original_sampling_rate = 1e5;
shrink_rate = original_sampling_rate/target_sampling_rate;
length_sample_original = target_sampling_rate*shrink_rate;
% ********************************
% ********************************

% data length standardlization processing
% **************************************
% **************************************
num_frames_perchannel_MDA = floor(size(data_MDA,2)/length_sample_original);
num_frames_perchannel_MCF7A = floor(size(data_MCF7A,2)/length_sample_original);
num_frames_perchannel_MCF10A = floor(size(data_MCF10A,2)/length_sample_original);

length_stand_MDA = num_frames_perchannel_MDA * length_sample_original;
length_stand_MCF7A = num_frames_perchannel_MCF7A * length_sample_original;
length_stand_MCF10A = num_frames_perchannel_MCF10A * length_sample_original;
data_MDA = data_MDA(:,1:length_stand_MDA);
data_MCF7A = data_MCF7A(:,1:length_stand_MCF7A);
data_MCF10A = data_MCF10A(:,1:length_stand_MCF10A);
% **************************************
% **************************************

% channel cancelling and framing the data
% ***************************************
% ***************************************
data_MDA = F_ChannelCancel(data_MDA);
data_MCF7A = F_ChannelCancel(data_MCF7A);
data_MCF10A = F_ChannelCancel(data_MCF10A);

data = {10*(data_MDA-mean(data_MDA))+2.*mean(data_MDA),...
    10*(data_MCF7A-mean(data_MCF7A))+2.*mean(data_MCF7A),...
    10*(data_MCF10A-mean(data_MCF10A))+2.*mean(data_MCF10A)};
data{1} = [data{1};ones(1,size(data{1},2)).*1];
data{2} = [data{2};ones(1,size(data{2},2)).*2];
data{3} = [data{3};ones(1,size(data{3},2)).*3];
data_new = {};
for i = 1:length(data)
    num_frames = size(data{i},2)/length_sample_original;
    for j = 1:num_frames
        data_new{end+1} = data{i}(:,(j-1)*length_sample_original+1: ...
            j*length_sample_original);
    end
end
disp('[Done] Data standarlization')
% decrease sample rating
% **********************
% **********************
disp('[Start] Down sampling')
for j = 1:length(data_new)
    temp_data = data_new{j};
    temp_data_trans = zeros(2,target_sampling_rate);
    for i = 1:target_sampling_rate
        temp_data_trans(1,i) = mean(temp_data(1,(i-1)*shrink_rate+1:i*shrink_rate));
        temp_data_trans(2,:) = temp_data(2,1);
        data_new{j} = temp_data_trans;
    end
end
disp('[Done] Down sampling')
% **********************
% **********************
disp('[Start] Data saving')
save data_10.mat data_new
num_samples_per_fram = target_sampling_rate;
save num_samples_10.mat num_samples_per_fram;
disp('[Done] Data saving')
disp('[Off] Program')


function extract_data_gripmove
% EXTRACT_DATA_GRIPMOVE: loads ROS and neural data selects some important
% data fields, and then combines them. Saves data structs with all
% timestamps aligned to Plexon time.
%
% Saves for each session two structs:
%    task_data:
%
%    neural_data:
%
% Required preprocessing before running this script:
%    bag2mat <the bag files>
%    plx2mat <the plx files>

%% input parameters

% Bag file names.
% Bag file names.
bag_names = {...
    %'0059_2016-01-18-12-48-52',...
    %'0059_2016-01-18-13-02-45',...
    %'0059_2016-01-18-13-19-29',...
    %'0059_2016-01-18-13-25-07',...
    %'0059_2016-01-18-13-41-19'...
    %'0059_2016-05-25-15-41-44',...
    %'0059_2016-05-25-15-58-17',...  
    %'504_2016-05-25-14-46-46',...
    %'504_2016-05-25-15-02-58',...
    %'0059_2016-05-26-12-17-53',...
    %'0059_2016-05-26-12-38-56',...  
    %'504_2016-05-26-11-25-03',...
    '504_2016-05-26-11-45-52',...
    
    %'0059_2016-02-26-16-28-27',...
    %'0059_2016-02-26-16-38-13',...
    %'504_2016-01-11-13-56-44',...
    %'504_2016-01-11-14-10-01',...
    %'504_2016-07-18-12-19-08',...
    %'0059_2016-07-06-13-44-24',...
    %'0059_2017-02-01-13-48-35',...
    %'0059_2017-02-09-12-52-17',...
    %'0059_2017-02-09-13-18-40',...
    %'0059_2017-02-09-13-46-37',...
    %'504_2017-02-14-12-09-21',...
    %'504_2017-02-14-12-35-41',...
    %'504_2017-02-14-13-01-34',...
    };

% Corresponding cameras bag file names
camera_bag_names = {...
    %'0059_cameras_2016-01-18-12-48-52',...
    %'0059_cameras_2016-01-18-13-02-45',...
    %'0059_cameras_2016-01-18-13-19-29',...
    %'0059_cameras_2016-01-18-13-25-07',...
    %'0059_cameras_2016-01-18-13-41-19'...
    %'0059_cameras_2016-05-25-15-41-43',...
    %'0059_cameras_2016-05-25-15-58-17',...
    %'504_cameras_2016-05-25-14-46-46',...
    %'504_cameras_2016-05-25-15-02-58',...
    %'0059_cameras_2016-05-26-12-17-53',...
    %'0059_cameras_2016-05-26-12-38-56',...
    %'504_cameras_2016-05-26-11-25-03',...
    '504_cameras_2016-05-26-11-45-52',...
    %'0059_cameras_2016-02-26-16-28-27',...
    %'0059_cameras_2016-02-26-16-38-13',...
    %'504_cameras_2016-01-11-14-10-01',...
    %'504_cameras_2016-07-18-12-19-08',...
    %'0059_cameras_2016-07-06-13-44-24',...
    %'0059_cameras_2017-02-01-13-48-35',...
    %'0059_cameras_2017-02-09-12-52-17',...
    %'0059_cameras_2017-02-09-13-18-39',...
    %'0059_cameras_2017-02-09-13-46-36',...
    %'504_cameras_2017-02-14-12-35-41',...
    %'504_cameras_2017-02-14-12-35-41',...
    %'504_cameras_2017-02-14-13-01-34',...
    };

% Neural recording file names. The following is a cell of cells, where each
% inner cell must have the same number of arguments:
neural_data_files = {...
    %{'MAP1_0059_01182016001-01','MAP2_0059_01182016001-01','MAP3_0059_01182016001-01'},...
    %{'MAP1_0059_01182016002-01','MAP2_0059_01182016002-01','MAP3_0059_01182016002-01'},...
    %{'MAP1_0059_01182016003-01','MAP2_0059_01182016003-01','MAP3_0059_01182016003-01'},...
    %{'MAP1_0059_01182016004-01','MAP2_0059_01182016004-01','MAP3_0059_01182016004-01'},...
    %{'MAP1_0059_01182016005-01','MAP2_0059_01182016005-01','MAP3_0059_01182016005-01'}...
    %{'MAP1_0059_05252016002-01','MAP2_0059_05252016002-01','MAP3_0059_05252016002-01'}...
    %{'MAP1_0059_05252016003-01','MAP2_0059_05252016003-01','MAP3_0059_05252016003-01'}...
    %{'MAP1_504_05262016001-01','MAP2_504_05262016001-01','MAP3_504_05262016001-01'}...
    {'MAP1_504_05262016002-01','MAP2_504_05262016002-01','MAP3_504_05262016001-01'}...
    %{'MAP1_0059_05262016001-02','MAP2_0059_05262016001-01','MAP3_0059_05262016001-01'}...
    %{'MAP1_0059_05262016002-01','MAP2_0059_05262016002-01','MAP3_0059_05262016002-01'}...
    %{'MAP1_0059_02262016003-01','MAP2_0059_02262016003-01','MAP3_0059_02262016003-01'}...
    %{'MAP1_0059_02262016004-01','MAP2_0059_02262016004-01','MAP3_0059_02262016004-01'}...
    %{'MAP1_504_01112016001-01','MAP2_504_01112016001-01','MAP3_504_01112016001-01'}...
    %{'MAP1_504_01112016002-01','MAP2_504_01112016002-01','MAP3_504_01112016002-01'}...
    %{'MAP1_504_07182016002-VeryCrudeSort','MAP2_504_07182016002-VeryCrudeSort','MAP3_504_07182016002-VeryCrudeSort'}...
    %{'MAP1_0059_07062016001-AutoSort','MAP2_0059_07062016001-AutoSort','MAP3_0059_07062016001-AutoSort'}
    %{'MAP1_0059_02012017005-01','MAP2_0059_02012017005-01','MAP3_0059_02012017005-01'}...
    %{'MAP1_0059_02092017001-01','MAP2_0059_02092017001-01','MAP3_0059_02092017001-01'}...
    %{'MAP1_0059_02092017002-01','MAP2_0059_02092017002-01','MAP3_0059_02092017002-01'}...
    %{'MAP1_0059_02092017004-01','MAP2_0059_02092017004-01','MAP3_0059_02092017004-01'}...    
    %{'MAP1_504_02142017001-01','MAP2_504_02142017001-01','MAP3_504_02142017001-01'}
    %{'MAP1_504_02142017002-01','MAP2_504_02142017002-01','MAP3_504_02142017002-01'}
    %{'MAP1_504_02142017003-01','MAP2_504_02142017003-01','MAP3_504_02142017003-01'}    
    };

recording_systems = {...
    {'map1', 'map2', 'map3'},...
    %{'map1', 'map2', 'map3'},...
    %{'map1', 'map2', 'map3'},...
    %{'map1', 'map2', 'map3'},...
    %{'map1', 'map2', 'map3'}...
    };

% strobe logging versions:
%    - log_after_sending:  before 8-19-2015
%    - log_before_sending:  after 8-19-2015. more accurate.
strobe_logging_version = {...
    'log_before_sending',...
    %'log_before_sending',...
    %'log_before_sending',...
    %'log_before_sending',...
    %'log_before_sending'...
    };

% brain regions (this doesn't change every day)
brain_regions = {'m1', 's1', 'pmd', 'pmv'};

% map for brain region --> {recording system, channel subset}
%   (this doesn't change every day)
brain_region_chan_map = containers.Map;
brain_region_chan_map('pmd') = {'map1', 1:96};
brain_region_chan_map('m1') = {'map2', 1:96};
brain_region_chan_map('s1') = {'map3', 1:96};
brain_region_chan_map('pmv') = {{'map1',97:128}, {'map2', 97:128}, {'map3', 97:128} };


% alignment based on strobe event
%
%   method 'linear_lsq': linear offset + drift correction
%        neuraltime = (1+drift) * tasktime + offset
%        trained with least squares.
%
%   method 'poly_lsq': polynomial 2nd order
%        neuraltime = polyval(p2, tasktime)
%        trained with least squares.
%
%   method 'linear_huber': same as linear_lsq, but trained
%                          Huber loss

time_correction_method = 'poly_lsq';

%% checking
disp('checking input');

nsess = length(bag_names);

assert(length(camera_bag_names) == nsess);
assert(length(neural_data_files) == nsess);
assert(length(recording_systems) == nsess);
assert(length(strobe_logging_version) == nsess);
for isess = 1:nsess
    assert(length(neural_data_files{isess}) ...
        == length(recording_systems{isess}));
end

disp('tests passed!');

%% gripmove data extraction in a matlab-y way
% The goal here is to only extract specific ROS data into
% big simple matrices i.e. no struct array horseshit.
%

for isess = 1:nsess
    fprintf('\nloading %s\n', bag_names{isess});
    load(bag_names{isess}); coalesceChunks;
    
    topic_idx = @(topic)cellfun(@(x)strcmp(x, topic), ...
        ros_data.topics.name);
    
    clear data;
    clear task_data;
    
    %% scalars, parameters
    % param dumper output
    [task_data.param_struct, task_data.param_map, ~, ...
        task_data.param_names_char ] = extract_param_dumper_output(ros_data);
    
    % start time
    task_data.t_begin = ros_data.bag.time_begin;
    task_data.t_end = ros_data.bag.time_end;
    
    %%% streaming waves:
    % intended_pose
    idx = topic_idx('/intended_pose_wam');
    s = ros_data.topics.data{idx};
    nmsg = length(s);
    task_data.intended_pose.position = [s.position];
    task_data.intended_pose.orientation = repmat(...
        ros_data.topics.proto{idx}.orientation, 1, nmsg);
    task_data.intended_pose.timestamps = [s.timestamp] - task_data.t_begin;
    
    % TODO: hand jointstates
    
    % grip force
    if task_data.param_struct.auto_grasp
        idx = topic_idx('/virt_force_sensor');
        disp('detected auto_grasp=true');
    else
        idx = topic_idx( '/force_sensor');
    end
    s = ros_data.topics.data{idx};
    task_data.force_sensor.force_gf = [s.force_gf];
    task_data.force_sensor.receipt_ts = [s.timestamp] - task_data.t_begin;
    
    force_sensor_has_header_ts = isfield(s(1).header.stamp, 'sec');
    if force_sensor_has_header_ts
        task_data.force_sensor.header_ts = cellfun(@(x)header_ts(x), {s.header})...
            - task_data.t_begin;
    else
        warning('Could not find header timestamps in force sensor messages.');
    end
    
    %%%%%%%%%%%%%%%%%%%%%
    %% topics to extract: 
    %/punishment_delivered (? or did we stop recording)
    %/punishment_num
    %/reward_delivered
    %/reward_num
    %try: /catch_trial_pub (introduced later)

    idx = topic_idx( '/punishment_num');
    s = ros_data.topics.data{idx};
    task_data.punishment_num.val = [s.val];
    task_data.punishment_num.ts = [s.timestamp] - task_data.t_begin;
    
    idx = topic_idx( '/reward_num');
    s = ros_data.topics.data{idx};
    task_data.reward_num.val = [s.val];
    task_data.reward_num.ts = [s.timestamp] - task_data.t_begin;
    
    if ~(isempty(find(topic_idx('/catch_trial_pub')))) 
        disp('trying catch_trials')
        idx = topic_idx( '/catch_trial_pub');
        s = ros_data.topics.data{idx};
        task_data.catch_trial_pub.val = [s.val];
        task_data.catch_trial_pub.ts = [s.timestamp] - task_data.t_begin;
    end
        
    
    %% discrete events:
    
    % strobe events
    switch strobe_logging_version{isess}
        case 'log_before_sending'
            strobe_patt = 'sending strobe';
            offset = 0;
        case 'log_after_sending'
            strobe_patt = 'sent strobe';
            offset = -500e-6;
        otherwise
            strobe_patt = 'sending strobe';
            offset = 0;
    end
    
    
    idx = topic_idx('/rosout');
    
    rosout = ros_data.topics.data{idx};
    
    %         nmsg = length(rosout);
    %         for i = 1:nmsg
    %             if strcmp(rosout(i).file, 'core.py')
    %             fprintf('%-20s [%9.6f]: %s\n', rosout(i).name, ...
    %                 rosout(i).timestamp - task_data.t_begin, rosout(i).msg);
    %             end
    %         end
    
    find_pattern = @(y,z)cellfun(@(x)~isempty(x), regexp(y, z));
    
    % ROS strobe/transition events
    trans_msgs = rosout(find_pattern({rosout.msg}, 'Transitioned to'));
    strobe_msgs = rosout(find_pattern({rosout.msg}, strobe_patt));
    
    cell_strobenums = regexp({strobe_msgs.msg}, '[0-9]+', 'match');
    strobe_nums = cellfun(@(x)str2num(x{1}), cell_strobenums);
    strobe_receipt_ts = [strobe_msgs.timestamp] - task_data.t_begin;
    strobe_header_ts = cellfun(@(x)header_ts(x), {strobe_msgs.header}) ...
        - task_data.t_begin + offset;
    
    %     trans_scene_names = regexp({trans_msgs.msg},...
    %         '(?<=Transitioned to )[A-Za-z_]+(?=_scene)', 'match');
    %     trans_nums = cellfun(@(x)task_data.param_map(...
    %         sprintf('/gripmove_task/scene_codes/%s_scene', x{1})), ...
    %         trans_scene_names);
    %     trans_receipt_ts = [trans_msgs.timestamp] - task_data.t_begin;
    %     trans_header_ts = cellfun(@(x)header_ts(x), {trans_msgs.header}) ...
    %         - task_data.t_begin;
    %     [istrobe, itrans] = align_inds_timestamps(...
    %         strobe_header_ts, trans_header_ts, time_correction_method);
    %
    %     assert(all(trans_nums(itrans) == strobe_nums(istrobe)));
    %     a = cellfun(@(x)x{1}, trans_scene_names,...
    %         'UniformOutput', 0);
    %     task_data.strobe.scene_names = a(itrans);
    
    scene_code_patt = '(?<=\/gripmove_task\/scene_codes\/)[A-Za-z_]+(?=_scene)';
    b = task_data.param_map.keys();
    a = find_pattern(b, scene_code_patt);
    c = b(a);
    num2scene = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    for i = 1:length(c)
        num = task_data.param_map(c{i});
        num2scene(num) = regexp(c{i}, scene_code_patt, 'match');
    end
    scene_names_from_strobe = {};
    for i = 1:length(strobe_nums)
        scene_names_from_strobe{i} = num2scene(strobe_nums(i));
    end
    
    task_data.strobe.receipt_ts = strobe_receipt_ts;
    task_data.strobe.header_ts = strobe_header_ts;
    task_data.strobe.nums = strobe_nums;
    task_data.strobe.scene_names = scene_names_from_strobe;
    
    % reward events
    reward_patt = '(?<=start reward delivery \()[0-9.]+(?=s\))';
    reward_msgs = rosout(find_pattern({rosout.msg}, reward_patt));
    
    task_data.reward.ontime = cellfun(@(x)str2num(x{1}),...
        regexp({reward_msgs.msg}, reward_patt, 'match'));
    task_data.reward.header_ts = cellfun(@(x)header_ts(x), ...
        {reward_msgs.header}) - task_data.t_begin;
    task_data.reward.receipt_ts = [reward_msgs.timestamp] - task_data.t_begin;
    
    
    
    
    %% video
    fprintf('loading %s\n', camera_bag_names{isess});
    clear ros_data;
    load(camera_bag_names{isess}); coalesceChunks;
    
    % all compressed video
    task_data.cam1_data = recorded_images(ros_data, ...
        '/cam1/camera/image_raw/compressed');
    task_data.cam2_data = recorded_images(ros_data, ...
        '/cam2/camera/image_raw/compressed');
    
    % save data from this file to an intermediate gripmove_1
    s = whos('task_data');
    sizemb = s.bytes/(1e6);
    fprintf('data subset size: %0.1fMB\n', sizemb);
    
    %% neural data, align ROS data to plx
    
    % brain region info
    neural_data.brain_regions = brain_regions;
    neural_data.brain_region_chan_map = brain_region_chan_map;
    
    nfiles = length(neural_data_files{isess});
    
    % search plx files for strobe event
    clear neural_data;
    neural_data.Strobed  = [];
    neural_data.recording_systems = recording_systems{isess};
    
    for i = 1:nfiles
        c = neural_data_files{isess};
        s = load(c{i}, 'data');
        plx_data = s.data;
        
        if isempty(neural_data.Strobed)
            neural_data.Strobed = plx_data.Strobed;
        end
        
        neural_data.spikeTimes{i} = plx_data.spikeTimes;
        
        %%%%commented out below because too large to save v7, which ->
        %%%%problems with opening it with sio.loadmat
        %if isfield(plx_data, 'ad')
            %neural_data.ad{i} = plx_data.ad;
            %neural_data.t_start_ad(i) = plx_data.t_start_ad;
        %end
        %neural_data.metadata{i} = plx_data.metadata;
        %neural_data.ad_sampling_rate = plx_data.metadata.SlowADSamplingRate;
    end
    
    assert(~isempty(neural_data.Strobed));
    
    % find index subset based on timestamp alignment
    [istr_task, istr_neur] = ...
        align_inds_timestamps(task_data.strobe.header_ts, ...
        neural_data.Strobed(:,1)', time_correction_method);
    
    if ~all(task_data.strobe.nums(istr_task) ...
            == neural_data.Strobed(istr_neur,2)') % indicates alignment failure
        
        %error(['Strobes: Simple index shift alignment failed. Could resort ' ...
        %      'to Needleman-Wunch Algorithm to attempt aligning ' ...
        %      'timestamps. But, cannot recover all events. To proceed ' ...
        %      'anyway, comment out this error msg command.']);
        
        [poffset, amatch, bmatch, matchidx] = align_timestamps(task_data.strobe.header_ts, neural_data.Strobed(:,1)', 1e-3);
        
        [ia, ib] = ...
            align_inds_timestamps(amatch, ...
            bmatch, time_correction_method);
        
        istr_task = matchidx(ia,1);
        istr_neur = matchidx(ib,2);
                
    end
    
    if length(istr_neur) < size(neural_data.Strobed,1)
        disp('Had to truncate Plexon strobed words to line up with ROS strobe messages.');
    end
    
    %assert(all(task_data.strobe.nums(istr_task) ...
    %    == neural_data.Strobed(istr_neur,2)'));
    
    % truncate strobes in both task and neural data to subset
    task_data.strobe.receipt_ts = task_data.strobe.receipt_ts(istr_task);
    task_data.strobe.header_ts = task_data.strobe.header_ts(istr_task);
    task_data.strobe.nums = task_data.strobe.nums(istr_task);
    task_data.strobe.scene_names = task_data.strobe.scene_names(istr_task);
    
    neural_data.Strobed = neural_data.Strobed(istr_neur,:);
    
    fprintf('time correction:\n');
    
    % should now have perfect 1 to 1 correspondence
    ts_task = task_data.strobe.header_ts';
    ts_neur = neural_data.Strobed(:, 1);
    
    [correct_tasktime, timecorrect_params] = ...
        compute_time_correction(ts_task,...
        ts_neur, time_correction_method);
    
    ts_task_corrected = correct_tasktime(ts_task);
    
    spread = std(ts_task_corrected - ts_neur);
    fprintf('corrected spread: %0.2f ms\n', 1000*spread);
    
    plot(ts_task, ts_task_corrected - ts_neur, 'o');
    xlabel('Time (s)');
    ylabel('Task Time - Neural Time (s)');
    
    
    % actually do the time correcting
    task_data.intended_pose.timestamps = ...
        correct_tasktime(task_data.intended_pose.timestamps);
    task_data.force_sensor.receipt_ts = ...
        correct_tasktime(task_data.force_sensor.receipt_ts);
    if force_sensor_has_header_ts
        task_data.force_sensor.header_ts = ...
            correct_tasktime(task_data.force_sensor.header_ts);
    end
    task_data.reward.receipt_ts = ...
        correct_tasktime(task_data.reward.receipt_ts);
    task_data.reward.header_ts = ...
        correct_tasktime(task_data.reward.header_ts);
    task_data.strobe.receipt_ts = ...
        correct_tasktime(task_data.strobe.receipt_ts);
    task_data.strobe.header_ts = ...
        correct_tasktime(task_data.strobe.header_ts);
    task_data.cam1_data.timestamps = ...
        correct_tasktime(task_data.cam1_data.timestamps);
    task_data.cam2_data.timestamps = ...
        correct_tasktime(task_data.cam2_data.timestamps);
    
    task_data.timecorrect_params = timecorrect_params;
    
    
    %JH addition: correct time params for reward_pub, punishment_pub, and
    %catch_trial_pub
    
    task_data.punishment_num.ts = correct_tasktime(task_data.punishment_num.ts);
    task_data.reward_num.ts = correct_tasktime(task_data.reward_num.ts);
    
    if ~(isempty(find(topic_idx('/catch_trial_pub')))) 
        task_data.catch_trial_pub.ts = correct_tasktime(task_data.catch_trial_pub.ts);
    end
        
    
    
    
    %% save the data
    s = whos('neural_data');
    sizemb = sizemb + s.bytes/(1e6);
    fprintf('total size = %0.3f MB\n', sizemb);
    
    neural_gb = s.bytes/(1e9);
    s = whos('task_data');
    task_gb = s.bytes/(1e9);
    if neural_gb > 2 || task_gb > 2
        usev73 = 1;
    else
        usev73 = 0;
    end
    
    fname = ['Extracted_', bag_names{isess}];
    fprintf('saving %s.mat\n',fname);
    
    if usev73
        save(fname, 'task_data', 'neural_data', '-v7');
    else
        save(fname, 'task_data', 'neural_data');
    end
    
end


end


function [t] = header_ts(header)
sec = header.stamp.sec;
nsec =  header.stamp.nsec;
t = double(sec) + 1e-9*double(nsec);
end
function extract_ts_val_test(filename)
%pull out timestamps and reward/punishment values by trial
%uses data from Extracted_NHPID_date_time.mat

%filelist:

load (filename)
%load('Extracted_0059_2015-10-19-16-25-20.mat');

%%%% strobe words %%%%
% 0 = reset
% 1 = disp_rp
% 2 = reaching
% 3 = grasping
% 4 = transport
% 5 = releasing
% 6 = success
% 7 = penalty
% 8 = shattering

%pulling relevant strobe information
reset_scene = neural_data.Strobed(neural_data.Strobed(:, 2) == 0);
reset_scene(:,2) = 0; %to match dimensions of other arrays
reset_scene(:, 3) = 0;

cue_presentation = neural_data.Strobed(neural_data.Strobed(:, 2) == 1);
cue_presentation(:, 2) = 0; %to match dimensions of other arrays
cue_presentation(:, 3) = 1;

reach_scene = neural_data.Strobed(neural_data.Strobed(:,2) == 2);
reach_scene(:,2) = 0;
reach_scene(:,3) = 2;

grasp_scene = neural_data.Strobed(neural_data.Strobed(:,2) == 3);
grasp_scene(:,2) = 0;
grasp_scene(:,3) = 3;

transport_scene = neural_data.Strobed(neural_data.Strobed(:,2) == 4);
transport_scene(:,2) = 0;
transport_scene(:,3) = 4;

release_scene = neural_data.Strobed(neural_data.Strobed(:,2) == 5);
release_scene(:,2) = 0;
release_scene(:,3) = 5;

reward_delivery = neural_data.Strobed(neural_data.Strobed(:, 2) == 6);
reward_delivery(:, 2) = 0; %to match dimensions of other arrays
reward_delivery(:, 3) = 6;

penalty_scene = neural_data.Strobed(neural_data.Strobed(:, 2) == 7);
penalty_scene(:, 2) = 0; %to match dimensions of other arrays
penalty_scene(:, 3) = 7;

shattering_scene = neural_data.Strobed(neural_data.Strobed(:, 2) == 8);
shattering_scene(:, 2) = 0; %to match dimensions of other arrays
shattering_scene(:, 3) = 8;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%key of all data: (column 3)
%reward_num = 10;
%punishment_num = 11;
%catch_trial = 12;
%reset_scene = 0;
%cue_presentation = 1;
%reward_delivery = 6;
%penalty_scene = 7;
%shattering_scene = 8;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

reward_num=[];
reward_num(:, 1) = task_data.reward_num.ts;
reward_num(:, 2) = task_data.reward_num.val;
reward_num(:, 3) = 10;

punishment_num = [];
punishment_num(:, 1) = task_data.punishment_num.ts;
punishment_num(:, 2) = task_data.punishment_num.val;
punishment_num(:, 3) = 11;

catch_trial = [];

if sum(strcmp(fieldnames(task_data), 'catch_trial_pub')) == 1
    catch_trial(:, 1) = task_data.catch_trial_pub.ts;
    catch_trial(:, 2) = task_data.catch_trial_pub.val;
    catch_trial(:, 3) = 12;
end

all = cat(1, reward_num, punishment_num, reset_scene, cue_presentation, reach_scene, grasp_scene, transport_scene, release_scene, reward_delivery, penalty_scene, shattering_scene);

if ~(isempty(catch_trial))
    disp('catch_trials exist')
    all = cat(1, all, catch_trial);
else
    disp('no catch_trials')
end

sorted_all = sortrows(all, 1);

trial_ct = 0;
i = 1;
resets = find(sorted_all(:, 3) == 0);
trial_breakdown=zeros(length(resets), 7);

success_ct = 0;
failure_ct = 0;

for i = 1: length(resets) -1
    
    trial_start = resets(i);
    trial_end = resets(i+1) - 1;
    trial_ct = trial_ct + 1;
    
    %This column is to give the reset time at the end of this trial for
    %comparison purposes (when reward or punishment would have been
    %delivered)
    trial_breakdown(trial_ct, 7) = sorted_all(resets(i+1) , 1);
    
    for j = trial_start + 1 : trial_end
        
        if sorted_all(j, 3) == 10
            %reward_num
            reward_val = sorted_all(j, 2);
            trial_breakdown(trial_ct, 1) = reward_val;

        elseif sorted_all(j, 3) == 11
            %punishment_num
            punishment_val = sorted_all(j, 2);
            trial_breakdown(trial_ct, 2) = punishment_val;

        elseif sorted_all(j, 3) == 12
            %catch_trial
            catch_val = sorted_all(j, 2);
            trial_breakdown(trial_ct, 3) = catch_val;

        elseif sorted_all(j, 3) == 1
            %cue displayed
            cue_ts = sorted_all(j, 1);
            trial_breakdown(trial_ct, 4) = cue_ts;
            
        elseif sorted_all(j,3) == 2
            %reach
            reach_ts = sorted_all(j,1);
            trial_breakdown(trial_ct,8) = reach_ts;
            
        elseif sorted_all(j,3) == 3
            %grasp
            grasp_ts = sorted_all(j,1);
            trial_breakdown(trial_ct,9) = grasp_ts;
            
        elseif sorted_all(j,3) == 4
            %transport
            transport_ts = sorted_all(j,1);
            trial_breakdown(trial_ct,10) = transport_ts;
            
        elseif sorted_all(j,3) == 5
            %release
            release_ts = sorted_all(j,1);
            trial_breakdown(trial_ct,11) = release_ts;
            
        elseif sorted_all(j, 3) ==6
            %reward delivery
            reward_delivery_ts = sorted_all(j, 1);
            trial_breakdown(trial_ct, 5) = reward_delivery_ts;
            success_ct = success_ct + 1;
             
        elseif sorted_all(j, 3) ==7
            %penalty scene
            penalty_ts = sorted_all(j,1);
            trial_breakdown(trial_ct, 6) = penalty_ts;
            failure_ct = failure_ct + 1;
             
            
        elseif sorted_all(j, 3) ==8
            %shattering scene
            %penalty and shattering both counted as penalty here
            shattering_penalty_ts = sorted_all(j, 1);
            trial_breakdown(trial_ct, 6) = shattering_penalty_ts;
            failure_ct = failure_ct + 1;
                 
        end
    end 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
%take care of 3/-3 blocks

if any((trial_breakdown(:,1)) == 3)
    disp('3/-3 block')
    trial_breakdown(trial_breakdown(:,1) == 3) = 1;
    for i =1:length(trial_breakdown)
        if trial_breakdown(i,2) == 3
            trial_breakdown(i,2) = 1;
        end
    end
    
    %trial_breakdown(trial_breakdown(:,2) == 3) = 1;
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%

all_r = trial_breakdown(trial_breakdown(:, 1) == 1, :);
all_nr = trial_breakdown(trial_breakdown(:, 1) == 0, :);

all_p = trial_breakdown(trial_breakdown(:, 2) == 1, :);
all_np = trial_breakdown(trial_breakdown(:, 2) == 0, :);

%r_s = reward cue, success. r_f = reward cue, fail, nr = no reward, etc
%TODO adjust for 2, 3 levels of r/p

%reward cue, success (rewarding)
r_s = trial_breakdown((trial_breakdown(:, 1) == 1) & ~(trial_breakdown(:, 5) == 0) & (trial_breakdown(:, 3) == 0), :);
r_s_cue_ts = r_s(:, 4)';
r_s_rdelivery_ts = r_s(:, 5)';

%reward cue, fail (not rewarding)
r_f = trial_breakdown((trial_breakdown(:, 1) == 1) & ~(trial_breakdown(:, 6) == 0) & (trial_breakdown(:, 3) == 0), :);
r_f_cue_ts = r_f(:, 4)';
r_f_nextreset = r_f(:, 7)';

%no reward cue, success (not rewarding)
nr_s= trial_breakdown((trial_breakdown(:, 1) == 0) & ~(trial_breakdown(:, 5) == 0) & (trial_breakdown(:, 3) == 0), :);
nr_s_cue_ts = nr_s(:, 4)';
nr_s_nextreset = nr_s(:, 7)';

%no reward cue, fail (not rewarding)
nr_f = trial_breakdown((trial_breakdown(:, 1) == 0) & ~(trial_breakdown(:, 6) == 0) & (trial_breakdown(:, 3) == 0), :);
nr_f_cue_ts = nr_f(:,4)';
nr_f_nextreset = nr_f(:, 7)';

%punishment cue, success (no punishment)
p_s = trial_breakdown((trial_breakdown(:, 2) == 1) & ~(trial_breakdown(:, 5) == 0) & (trial_breakdown(:, 3) == 0), :);
p_s_cue_ts = p_s(:, 4)';
p_s_nextreset = p_s(:, 7)';

%punishment cue, fail (punishment)
p_f = trial_breakdown((trial_breakdown(:, 2) == 1) & ~(trial_breakdown(:, 6) == 0) & (trial_breakdown(:, 3) == 0), :);
p_f_cue_ts = p_f(:, 4)';
p_f_pdelivery_ts = p_f(:, 6)';

%no punishmnt cue, success (no punishment)
np_s = trial_breakdown((trial_breakdown(:, 2) == 0) & ~(trial_breakdown(:, 5) == 0) & (trial_breakdown(:, 3) == 0), :);
np_s_cue_ts = np_s(:, 4)';
np_s_nextreset = np_s(:, 7)';

%no punishment cue, fail (no punishment)
np_f = trial_breakdown((trial_breakdown(:, 2) == 0) & ~(trial_breakdown(:, 6) == 0) & (trial_breakdown(:, 3) == 0), :);
np_f_cue_ts = np_f(:, 4)';
np_f_nextreset = np_f(:, 7)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%rewarding catch trial (supposed to be rewarding, not)
%also look at rewarding catch trials that fail? (vv)
r_s_catch = trial_breakdown((trial_breakdown(:,2) == 1) & ~(trial_breakdown(:, 5) ==0) & (trial_breakdown(:, 3) == 1), :);
r_s_catch_cue_ts = r_s_catch(:, 4)';
r_s_catch_nextreset = r_s_catch(:, 7)';

%punishment catch trial (supposed to be punishment, not)
p_f_catch = trial_breakdown((trial_breakdown(:,2) == 1) & ~(trial_breakdown(:, 5) ==0) & (trial_breakdown(:, 3) == 2), :);
p_f_catch_cue_ts = p_f_catch(:, 4)';
p_f_catch_nextreset = p_f_catch(:, 7)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this data also included above
%% both cues, success (rewarding) 
rp_s = trial_breakdown((trial_breakdown(:,1) == 1) & (trial_breakdown(:,2) == 1) & ~(trial_breakdown(:,5) == 0) & (trial_breakdown(:, 3) ==0), :);
rp_s_cue_ts = rp_s(:, 4)';
rp_s_rdelivery_ts = rp_s(:, 5)';

%both cues, fail (punishment)
rp_f = trial_breakdown((trial_breakdown(:,1) == 1) & (trial_breakdown(:,2) == 1) & ~(trial_breakdown(:,6) == 0) & (trial_breakdown(:, 3) ==0), :);
rp_f_cue_ts = rp_f(:, 4)';
rp_f_pdelivery_ts = rp_f(:, 6)';

%neither cue, success (not rewarding or punishing)
nrnp_s = trial_breakdown((trial_breakdown(:,1) == 0) & (trial_breakdown(:,2) == 0) & ~(trial_breakdown(:,5) == 0) & (trial_breakdown(:, 3) ==0), :);
nrnp_s_cue_ts = nrnp_s(:, 4)';
nrnp_s_nextreset = nrnp_s(:, 7)';

%neither cue, fail (not rewarding or punishing)
nrnp_f = trial_breakdown((trial_breakdown(:,1) == 0) & (trial_breakdown(:,2) == 0) & ~(trial_breakdown(:,6) == 0) & (trial_breakdown(:, 3) ==0), :);
nrnp_f_cue_ts = nrnp_f(:, 4)';
nrnp_f_nextreset = nrnp_f(:, 7)';

%reward cue only (no punishment cue), successful (rewarding)
r_only_s = trial_breakdown((trial_breakdown(:, 1) == 1) & (trial_breakdown(:,2) ==0) & ~(trial_breakdown(:, 5) == 0) & (trial_breakdown(:, 3) == 0), :);
r_only_s_cue_ts = r_only_s(:, 4)';
r_only_s_rdelivery_ts = r_only_s(:, 5)';

%reward cue only (no punishment cue), unsuccessful (non-rewarding)
r_only_f = trial_breakdown((trial_breakdown(:, 1) == 1) & (trial_breakdown(:,2) ==0) & ~(trial_breakdown(:, 6) == 0) & (trial_breakdown(:, 3) == 0), :);
r_only_f_cue_ts = r_only_f(:, 4)';
r_only_f_nextreset = r_only_f(:, 7)';

%punishment cue only (no reward cue), successful (non-punishing)
p_only_s = trial_breakdown((trial_breakdown(:, 1) == 0) & (trial_breakdown(:,2) ==1) & ~(trial_breakdown(:, 5) == 0) & (trial_breakdown(:, 3) == 0), :);
p_only_s_cue_ts = p_only_s(:, 4)';
p_only_s_nextreset = p_only_s(:, 7)';

%punishment cue only (no reward cue), unsuccessful (punishing)
p_only_f = trial_breakdown((trial_breakdown(:, 1) == 0) & (trial_breakdown(:,2) ==1) & ~(trial_breakdown(:, 6) == 0) & (trial_breakdown(:, 3) == 0), :);
p_only_f_cue_ts = p_only_f(:, 4)';
p_only_f_pdelivery_ts = p_only_f(:, 6)';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%extract performance metrics %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

perc_success = success_ct / trial_ct
perc_failure = failure_ct / trial_ct

%reach time = grasp - reach
reach_times = trial_breakdown(:,9) - trial_breakdown(:,8); 
reach_times = reach_times(reach_times > 0); %negative values where failed during this scene, so this removes them
reach_avg = mean(reach_times)
reach_std = std2(reach_times);

%grasp time = transport - grasp
grasp_times = trial_breakdown(:,10) - trial_breakdown(:,9); 
grasp_times = grasp_times(grasp_times > 0);
grasp_avg = mean(grasp_times)
grasp_std = std2(grasp_times);

%transport time = release - transport
transport_times = trial_breakdown(:,11) - trial_breakdown(:,10);
transport_times = transport_times(transport_times > 0);
transport_avg = mean(transport_times)
transport_std = std2(transport_times);

%release time = success - release
release_times = trial_breakdown(:,5) - trial_breakdown(:,11); 
release_times = release_times(release_times > 0);
release_avg = mean(release_times)
release_std = std2(release_times);

total_times = trial_breakdown(:,5) - trial_breakdown(:,8);
total_times = total_times(total_times > 0);
total_time_avg = mean(total_times)
total_time_std = std2(total_times);


%%%%%%%%%%%%
% save %%
 
clear  catch_trial catch_val cue_presentation cue_ts i j penalty_scene penalty_ts punishment_num punishment_val reset_scene resets reward_delivery reward_delivery_ts reward_num reward_val shattering_penalty_ts shattering_scene task_data trial_end trial_start all neural_data

string= sprintf('%s_timestamps.mat', filename(1:end-4));

string2 = sprintf('%s_performance_metrics.mat', filename(1:end-4));

save(string2,'perc_success','perc_failure','reach_avg','reach_std','grasp_avg','grasp_std','transport_avg','transport_std','release_avg','release_std','total_time_avg','total_time_std')
clear perc_success perc_failure reach_avg reach_std grasp_avg grasp_std transport_avg transport_std release_avg release_std total_time_avg total_time_std

save(string, '-v7');

clear all

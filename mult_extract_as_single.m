function extract_ts_val_test(filename)
%pull out timestamps and reward/punishment values by trial
%uses data from Extracted_NHPID_date_time.mat


load (filename)

%load('Extracted_0059_2017-02-09-13-46-37.mat');

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

ordered_data = neural_data.Strobed;

temp = ordered_data(ordered_data(:,2) == 0);

%because doesn't start with a reset
resets = zeros(length(temp) +1,1);
resets(2:end) = temp;

trial_breakdown = zeros(length(resets),11);
trial_breakdown(:,1) = resets;

reset_ct = 1;
for i = 1 : length(ordered_data)
    if ordered_data(i,2) == 1
        %disp_rp (cue scene)
        trial_breakdown(reset_ct,2) = ordered_data(i,1);
    elseif ordered_data(i,2) == 6
        %success
        trial_breakdown(reset_ct,3) = ordered_data(i,1);
    elseif ordered_data(i,2) == 7
        %failure
        trial_breakdown(reset_ct,4) = ordered_data(i,1);
    elseif ordered_data(i,2) == 0
        %next reset
        trial_breakdown(reset_ct,9) = ordered_data(i,1);
        reset_ct = reset_ct+ 1;
    end   
end


reward_ts = task_data.reward_num.ts';
reward_val = task_data.reward_num.val';

punishment_ts = task_data.punishment_num.ts';
punishment_val = task_data.punishment_num.val';

if ~(length(reward_ts) == length(punishment_ts))
   if ((reward_ts(1) - punishment_ts(1)) > 1) &((length(punishment_ts) - length(reward_ts )) == 1)
       temp = punishment_ts;
       punishment_ts = [];
       punishment_ts = temp(2:end);
       temp2 = punishment_val;
       punishment_val = [];
       punishment_val = temp2(2:end);    
   elseif ((reward_ts(1) - punishment_ts(1)) < -1) & ((length(reward_ts) - length(punishment_ts)) == 1)
       temp = reward_ts;
       reward_ts = [];
       reward_ts = temp(2:end);
       temp2 = reward_val;
       reward_val = [];
       reward_val = temp2(2:end);
   else
      disp('reward num and punishment num alignment error, not fixed')
   end 
end
    
trial_breakdown((2:length(reward_ts)+1),5) = reward_ts;
trial_breakdown((2:length(reward_val)+1),6) = reward_val;
trial_breakdown((2:length(punishment_ts)+1),7) = punishment_ts;
trial_breakdown((2:length(punishment_val)+1),8) = punishment_val;

if trial_breakdown(2,5) > trial_breakdown(2,9) && trial_breakdown(2,7) > trial_breakdown(2,9)
    trial_breakdown(3:length(reward_ts)+2,5) = trial_breakdown(2:length(reward_ts)+1,5);
    trial_breakdown(3:length(reward_val)+2,6) = trial_breakdown(2:length(reward_val)+1,6);
    trial_breakdown(3:length(punishment_ts)+2,7) = trial_breakdown(2:length(punishment_ts)+1,7);
    trial_breakdown(3:length(punishment_val)+2,8) = trial_breakdown(2:length(punishment_val)+1,8);
    trial_breakdown(2,5) = 0;
    trial_breakdown(2,6) = 0;
    trial_breakdown(2,7) = 0;
    trial_breakdown(2,8) = 0;
end


%sanity check
for i = 1:length(trial_breakdown)
   for j = 1:8
       if (trial_breakdown(i,j) > trial_breakdown(i,9)) && ~(j == 6) && ~(j == 8)
           fprintf('error indexing trial %i of %i\n',i,length(trial_breakdown))
       end
   end
end


%trial breakdown columns:
%1      2           3           4           5           6           7          8            9             10                 11
%reset  disp_rp     succ_scene  fail_scene time_r_pub   rew_num    time_p_pub  pun_num   nextreset       catch_trial_num     catch_trial_type


if sum(strcmp(fieldnames(task_data), 'catch_trial_pub')) == 1
    catch_trial = [];
    catch_trial(:,1) = task_data.catch_trial_pub.ts';
    catch_trial(:,2) = task_data.catch_trial_pub.val';
    j = 1;
    catch_indices = zeros(length(catch_trial),2);
    for i=1:length(trial_breakdown)
        if (catch_trial(j,1) > trial_breakdown(i,1)) & (catch_trial(j,1) < trial_breakdown(i,9))
            trial_breakdown(i,10) = catch_trial(j,1);
            trial_breakdown(i,11) = catch_trial(j,2);
            catch_indices(j,1) = i;
            catch_indices(j,2) = catch_trial(j,2);
            j = j + 1;
            if j > length(catch_indices)
                break
            end
        end
    end
    if catch_indices(end,1) == 0
       disp('potential error indexing catch_trials') 
    end
end


if ~(isempty(catch_trial))
    disp('catch_trials exist')
else
    disp('no catch_trials')
end


%%%%%%%%%%%%%%%%%%%%%%%%%
all_r = trial_breakdown(trial_breakdown(:, 6) > 0, :);
all_nr = trial_breakdown(trial_breakdown(:, 6) == 0, :);

all_p = trial_breakdown(trial_breakdown(:, 8) > 0, :);
all_np = trial_breakdown(trial_breakdown(:, 8) == 0, :);

%r_s = reward cue, success. r_f = reward cue, fail, nr = no reward, etc




% %ALL: reward cue, success (rewarding)
% r_s = trial_breakdown((trial_breakdown(:, 4) >= 1) & ~(trial_breakdown(:,7)==0) & (trial_breakdown(:, 8) == 0) & (trial_breakdown(:, 9) == 0), :);
% r_s_cue_ts = r_s(:, 2)';
% r_s_rdelivery_ts = r_s(:, 7)';
% 
% %ALL: reward cue, fail (not rewarding)
% r_f = trial_breakdown((trial_breakdown(:, 4) >= 1) & (trial_breakdown(:,7)==0) & ~(trial_breakdown(:, 8) == 0) & (trial_breakdown(:, 9) == 0), :);
% r_f_cue_ts = r_f(:, 2)';
% r_f_nextreset = r_f(:, 10)';
% 
% %ALL: no reward cue, success (not rewarding)
% nr_s= trial_breakdown((trial_breakdown(:, 4) == 0) & ~(trial_breakdown(:,7)==0) & (trial_breakdown(:, 8) == 0) & (trial_breakdown(:, 9) == 0), :);
% nr_s_cue_ts = nr_s(:, 2)';
% nr_s_nextreset = nr_s(:, 10)';
% 
% %ALL: no reward cue, fail (not rewarding)
% nr_f = trial_breakdown((trial_breakdown(:, 4) == 0) & (trial_breakdown(:,7)==0) & ~(trial_breakdown(:, 8) == 0) & (trial_breakdown(:, 9) == 0), :);
% nr_f_cue_ts = nr_f(:,2)';
% nr_f_nextreset = nr_f(:, 10)';
% 
% %ALL: punishment cue, success (no punishment)
% p_s = trial_breakdown(~(trial_breakdown(:, 6) == 0) & ~(trial_breakdown(:,7)==0) & (trial_breakdown(:, 8) == 0) & (trial_breakdown(:, 9) == 0), :);
% p_s_cue_ts = p_s(:, 2)';
% p_s_nextreset = p_s(:, 10)';
% 
% %ALL: punishment cue, fail (punishment)
% p_f = trial_breakdown(~(trial_breakdown(:, 6) == 0) & (trial_breakdown(:,7)==0) & ~(trial_breakdown(:, 8) == 0) & (trial_breakdown(:, 9) == 0), :);
% p_f_cue_ts = p_f(:, 2)';
% p_f_pdelivery_ts = p_f(:, 8)';
% 
% %ALL: no punishmnt cue, success (no punishment)
% np_s = trial_breakdown((trial_breakdown(:, 6) == 0) & ~(trial_breakdown(:,7)==0) & (trial_breakdown(:, 8) == 0) & (trial_breakdown(:, 9) == 0), :);
% np_s_cue_ts = np_s(:, 4)';
% np_s_nextreset = np_s(:, 10)';
% 
% %ALL: no punishment cue, fail (no punishment)
% np_f = trial_breakdown((trial_breakdown(:, 6) == 0) & (trial_breakdown(:,7)==0) & ~(trial_breakdown(:, 8) == 0) & (trial_breakdown(:, 9) == 0), :);
% np_f_cue_ts = np_f(:, 4)';
% np_f_nextreset = np_f(:, 10)';
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this data also included above
%% both cues, success (rewarding) 
rp_s = trial_breakdown(~(trial_breakdown(:,6) == 0) & ~(trial_breakdown(:,8) == 0) & ~(trial_breakdown(:,3) == 0) & (trial_breakdown(:,4) == 0) & (trial_breakdown(:,10) == 0), :);    %TODO rest- not catch trial col, others?    %(trial_breakdown(:,8)==0) & (trial_breakdown(:, 9) ==0), :);
rp_s_cue_ts = rp_s(:, 2)';
rp_s_rdelivery_ts = rp_s(:, 3)';
rp_s_rnum = rp_s(:,6)';
rp_s_pnum = rp_s(:,8)';

%both cues, fail (punishment)
rp_f = trial_breakdown(~(trial_breakdown(:,6) == 0) & ~(trial_breakdown(:,8) == 0) & (trial_breakdown(:,3) == 0) & ~(trial_breakdown(:,4) == 0) & (trial_breakdown(:,10) == 0), :);  
rp_f_cue_ts = rp_f(:, 2)';
rp_f_pdelivery_ts = rp_f(:, 4)';
rp_f_rnum = rp_f(:,6)';
rp_f_pnum = rp_f(:,8)';

%neither cue, success (not rewarding or punishing)
nrnp_s = trial_breakdown((trial_breakdown(:,6) == 0) & (trial_breakdown(:,8) == 0) & ~(trial_breakdown(:,3) == 0) & (trial_breakdown(:,4) == 0) & (trial_breakdown(:,10) == 0), :);  
nrnp_s_cue_ts = nrnp_s(:, 2)';
nrnp_s_nextreset = nrnp_s(:, 9)';
nrnp_s_rnum = nrnp_s(:,6)';
nrnp_s_pnum = nrnp_s(:,8)';

%neither cue, fail (not rewarding or punishing)
nrnp_f = trial_breakdown((trial_breakdown(:,6) == 0) & (trial_breakdown(:,8) == 0) & (trial_breakdown(:,3) == 0) & ~(trial_breakdown(:,4) == 0) & (trial_breakdown(:,10) == 0), :);  
nrnp_f_cue_ts = nrnp_f(:, 2)';
nrnp_f_nextreset = nrnp_f(:, 9)';
nrnp_f_rnum = nrnp_f(:,6)';
nrnp_f_pnum = nrnp_f(:,8)';

%reward cue only (no punishment cue), successful (rewarding)
r_only_s = trial_breakdown(~(trial_breakdown(:,6) == 0) & (trial_breakdown(:,8) == 0) & ~(trial_breakdown(:,3) == 0) & (trial_breakdown(:,4) == 0) & (trial_breakdown(:,10) == 0), :);  
r_only_s_cue_ts = r_only_s(:, 2)';
r_only_s_rdelivery_ts = r_only_s(:, 3)';
r_only_s_rnum = r_only_s(:,6)';
r_only_s_pnum = r_only_s(:,8)';

%reward cue only (no punishment cue), unsuccessful (non-rewarding)
r_only_f = trial_breakdown(~(trial_breakdown(:,6) == 0) & (trial_breakdown(:,8) == 0) & (trial_breakdown(:,3) == 0) & ~(trial_breakdown(:,4) == 0) & (trial_breakdown(:,10) == 0), :);  
r_only_f_cue_ts = r_only_f(:, 2)';
r_only_f_nextreset = r_only_f(:, 9)';
r_only_f_rnum = r_only_f(:,6)';
r_only_f_pnum = r_only_f(:,8)';

%punishment cue only (no reward cue), successful (non-punishing)
p_only_s = trial_breakdown((trial_breakdown(:,6) == 0) & ~(trial_breakdown(:,8) == 0) & ~(trial_breakdown(:,3) == 0) & (trial_breakdown(:,4) == 0) & (trial_breakdown(:,10) == 0), :);  
p_only_s_cue_ts = p_only_s(:, 2)';
p_only_s_nextreset = p_only_s(:, 9)';
p_only_s_rnum = p_only_s(:,6)';
p_only_s_pnum = p_only_s(:,8)';

%punishment cue only (no reward cue), unsuccessful (punishing)
p_only_f = trial_breakdown((trial_breakdown(:,6) == 0) & ~(trial_breakdown(:,8) == 0) & (trial_breakdown(:,3) == 0) & ~(trial_breakdown(:,4) == 0) & (trial_breakdown(:,10) == 0), :);  
p_only_f_cue_ts = p_only_f(:, 2)';
p_only_f_pdelivery_ts = p_only_f(:, 4)';
p_only_f_rnum = p_only_f(:,6)';
p_only_f_pnum = p_only_f(:,8)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%rewarding catch trial (supposed to be rewarding, not)
%also look at rewarding catch trials that fail? (vv)
r_s_catch = trial_breakdown(~(trial_breakdown(:,10) == 0) & (trial_breakdown(:,11) == 1),:); 
r_s_catch_cue_ts = r_s_catch(:, 2)';
r_s_catch_nextreset = r_s_catch(:, 10)';
rp_s_rnum = r_s_catch(:,6)';
rp_s_pnum = r_s_catch(:,8)';

%punishment catch trial (supposed to be punishment, not)
p_f_catch = trial_breakdown(~(trial_breakdown(:,10) == 0) & (trial_breakdown(:,11) == 2),:); 
p_f_catch_cue_ts = p_f_catch(:, 2)';
p_f_catch_nextreset = p_f_catch(:, 10)';
rp_s_rnum = p_f_catch(:,6)';
rp_s_pnum = p_f_catch(:,8)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%extract performance metrics %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

success_dimensions = size(rp_s) + size(nrnp_s) + size(r_only_s) + size(p_only_s);
failure_dimensions = size(rp_f) + size(nrnp_f) + size(r_only_f) + size(p_only_f);
trial_dimensions = size(trial_breakdown);


perc_success = success_dimensions(1)/trial_dimensions(1)
perc_failure = failure_dimensions(1)/trial_dimensions(1)


% %performance metrics
% %reach time = grasp - reach
% %reach_times = trial_breakdown(:,9) - trial_breakdown(:,8); 
% %reach_times = reach_times(reach_times > 0); %negative values where failed during this scene, so this removes them
% %reach_avg = mean(reach_times)
% %reach_std = std2(reach_times);
% 
% %grasp time = transport - grasp
% %grasp_times = trial_breakdown(:,10) - trial_breakdown(:,9); 
% %grasp_times = grasp_times(grasp_times > 0);
% %grasp_avg = mean(grasp_times)
% %grasp_std = std2(grasp_times);
% 
% %transport time = release - transport
% %transport_times = trial_breakdown(:,11) - trial_breakdown(:,10);
% %transport_times = transport_times(transport_times > 0);
% %transport_avg = mean(transport_times)
% %transport_std = std2(transport_times);
% 
% %release time = success - release
% %release_times = trial_breakdown(:,5) - trial_breakdown(:,11); 
% %release_times = release_times(release_times > 0);
% %release_avg = mean(release_times)
% %release_std = std2(release_times);
% 
% %total_times = trial_breakdown(:,5) - trial_breakdown(:,8);
% %total_times = total_times(total_times > 0);
% %total_time_avg = mean(total_times)
% %total_time_std = std2(total_times);
% 
% 


%%%%%%%%%%%

gripforce = [task_data.force_sensor.receipt_ts',task_data.force_sensor.force_gf'];


%%%%%%%%%%%%
% save %%
 
%clear  catch_trial catch_val cue_presentation cue_ts i j penalty_scene penalty_ts punishment_num punishment_val reset_scene resets reward_delivery reward_delivery_ts reward_num reward_val shattering_penalty_ts shattering_scene task_data trial_end trial_start all neural_data

string= sprintf('%s_timestamps.mat', filename(1:end-4));

%string2 = sprintf('%s_performance_metrics.mat', filename(1:end-4));

%save(string2,'perc_success','perc_failure','reach_avg','reach_std','grasp_avg','grasp_std','transport_avg','transport_std','release_avg','release_std','total_time_avg','total_time_std')
%clear perc_success perc_failure reach_avg reach_std grasp_avg grasp_std transport_avg transport_std release_avg release_std total_time_avg total_time_std

save(string, '-v7');

clear all







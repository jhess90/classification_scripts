load('master_hist.mat');

bfr_cue = M1_hist_dict.bfr_cue;
aft_cue = M1_hist_dict.aft_cue;
bfr_result = M1_hist_dict.bfr_result;
aft_result = M1_hist_dict.aft_result;

%TODO should be arrays of 1/0, some 2 though

[trials,units,bfr_bins] = size(bfr_cue);
[trials2,units2,aft_bins] = size(aft_cue);


bc_struct = struct([]);
ac_struct = struct([]);
br_struct = struct([]);
ar_struct = struct([]);
c_struct = struct([]);
r_struct = struct([]);
a_struct = struct([]);

for i=1:trials

    bc_struct(i).trialId = i;
    bc_struct(i).spikes = squeeze(bfr_cue(i,:,:));

    ac_struct(i).trialId = i;
    ac_struct(i).spikes = squeeze(aft_cue(i,:,:));

    br_struct(i).trialId = i;
    br_struct(i).spikes = squeeze(bfr_result(i,:,:));

    ar_struct(i).trialId = i;
    ar_struct(i).spikes = squeeze(aft_result(i,:,:));

    c_struct(i).trialId = i;
    c_struct(i).spikes = cat(2,squeeze(bfr_cue(i,:,:)),squeeze(aft_cue(i,:,:)));
 
    r_struct(i).trialId = i;
    r_struct(i).spikes = cat(2,squeeze(bfr_result(i,:,:)),squeeze(aft_result(i,:,:)));
    
    a_struct(i).trialId = i;
    a_struct(i).spikes = cat(2,squeeze(bfr_cue(i,:,:)),squeeze(aft_cue(i,:,:)),squeeze(bfr_result(i,:,:)),squeeze(aft_result(i,:,:)));
 
end

rewarding = find(condensed(:,4) > 0 & condensed(:,6) == 1);
punishing = find(condensed(:,5) > 0 & condensed(:,6) == -1);


save('cue.mat','c_struct')
save('result.mat','r_struct')
save('all.mat','a_struct')
save('trial_type.mat','rewarding','punishing')
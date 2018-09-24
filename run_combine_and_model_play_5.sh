#!/bin/sh

#
cd /home/jack/Dropbox/model_nl/rp_cue/0059/
mv Extracted* r_p_pop_response/
cd r_p_pop_response/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_lin_diff_pop_response.py
mv Extracted* ../play_avg_noz/
cd ../play_avg_noz/
python /home/jack/workspace/classification_scripts/combine_and_model_play_avgs_noz.py
mv Extracted* ../


cd /home/jack/Dropbox/model_nl/rp_cue/504/
mv Extracted* r_p_pop_response/
cd r_p_pop_response/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_lin_diff_pop_response.py
mv Extracted* ../

#
cd /home/jack/Dropbox/model_nl/alt_cue/0059/
mv Extracted* r_p_pop_response/
cd r_p_pop_response/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_lin_diff_pop_response.py
mv Extracted* ../play/
cd ../play/
python /home/jack/workspace/classification_scripts/combine_and_model_play.py
mv Extracted* ../play_r_p_only/
cd ../play_r_p_only/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_only.py
mv Extracted* ../play_avg/
cd ../play_avg/
python /home/jack/workspace/classification_scripts/combine_and_model_play_avgs.py
mv Extracted* ../

cd /home/jack/Dropbox/model_nl/alt_cue/504
mv Extracted* r_p_pop_response/
cd r_p_pop_response/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_lin_diff_pop_response.py
mv Extracted* ../

#
cd /home/jack/Dropbox/model_nl/single_cue/0059/
mv Extracted* r_p_pop_response/
cd r_p_pop_response/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_lin_diff_pop_response.py
mv Extracted* ../


cd /home/jack/Dropbox/model_nl/single_cue/504/
mv Extracted* r_p_pop_response/
cd r_p_pop_response/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_lin_diff_pop_response.py
mv Extracted* ../

#
cd /home/jack/Dropbox/model_nl/single_uncued/0059/
mv Extracted* r_p_pop_response/
cd r_p_pop_response/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_lin_diff_pop_response.py
mv Extracted* ../


cd /home/jack/Dropbox/model_nl/single_uncued/504/
mv Extracted* r_p_pop_response/
cd r_p_pop_response/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_lin_diff_pop_response.py
mv Extracted* ../

#
cd /home/jack/Dropbox/model_nl/uncued/0059/
mv Extracted* r_p_pop_response/
cd r_p_pop_response/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_lin_diff_pop_response.py
mv Extracted* ../


cd /home/jack/Dropbox/model_nl/uncued/504/
mv Extracted* r_p_pop_response/
cd r_p_pop_response/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_lin_diff_pop_response.py
mv Extracted* ../

#
cd /home/jack/Dropbox/model_nl/zero_or_three/0059/
mv Extracted* r_p_pop_response/
cd r_p_pop_response/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_lin_diff_pop_response.py
mv Extracted* ../


cd /home/jack/Dropbox/model_nl/zero_or_three/504/
mv Extracted* r_p_pop_response/
cd r_p_pop_response/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_lin_diff_pop_response.py
mv Extracted* ../





exec bash

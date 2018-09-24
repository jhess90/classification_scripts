#!/bin/sh

cd /home/jack/Dropbox/model_nl/rp_cue/0059/

mkdir play_r_p_only play_pop_avg
mv Extracted* play_r_p_only/
cd play_r_p_only/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_only.py
mv Extracted* ../play_pop_avg/
cd ../play_pop_avg/
python /home/jack/workspace/classification_scripts/combine_and_model_play_pop_response.py
mv Extracted* ../


cd /home/jack/Dropbox/model_nl/rp_cue/504/
mkdir play_r_p_only play_pop_avg
mv Extracted* play_r_p_only/
cd play_r_p_only/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_only.py
mv Extracted* ../play_pop_avg/
cd ../play_pop_avg/
python /home/jack/workspace/classification_scripts/combine_and_model_play_pop_response.py
mv Extracted* ../

#
cd /home/jack/Dropbox/model_nl/alt_cue/0059/
mkdir play_r_p_only play_pop_avg
mv Extracted* play_r_p_only/
cd play_r_p_only/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_only.py
mv Extracted* ../play_pop_avg/
cd ../play_pop_avg/
python /home/jack/workspace/classification_scripts/combine_and_model_play_pop_response.py
mv Extracted* ../

cd /home/jack/Dropbox/model_nl/alt_cue/504/
mkdir play_r_p_only play_pop_avg
mv Extracted* play_r_p_only/
cd play_r_p_only/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_only.py
mv Extracted* ../play_pop_avg/
cd ../play_pop_avg/
python /home/jack/workspace/classification_scripts/combine_and_model_play_pop_response.py
mv Extracted* ../

#
cd /home/jack/Dropbox/model_nl/single_cue/0059/
mkdir play_r_p_only play_pop_avg
mv Extracted* play_r_p_only/
cd play_r_p_only/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_only.py
mv Extracted* ../play_pop_avg/
cd ../play_pop_avg/
python /home/jack/workspace/classification_scripts/combine_and_model_play_pop_response.py
mv Extracted* ../


cd /home/jack/Dropbox/model_nl/single_cue/504/
mkdir play_r_p_only play_pop_avg
mv Extracted* play_r_p_only/
cd play_r_p_only/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_only.py
mv Extracted* ../play_pop_avg/
cd ../play_pop_avg/
python /home/jack/workspace/classification_scripts/combine_and_model_play_pop_response.py
mv Extracted* ../

#
cd /home/jack/Dropbox/model_nl/single_uncued/0059/
mkdir play_r_p_only play_pop_avg
mv Extracted* play_r_p_only/
cd play_r_p_only/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_only.py
mv Extracted* ../play_pop_avg/
cd ../play_pop_avg/
python /home/jack/workspace/classification_scripts/combine_and_model_play_pop_response.py
mv Extracted* ../


cd /home/jack/Dropbox/model_nl/single_uncued/504/
mkdir play_r_p_only play_pop_avg
mv Extracted* play_r_p_only/
cd play_r_p_only/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_only.py
mv Extracted* ../play_pop_avg/
cd ../play_pop_avg/
python /home/jack/workspace/classification_scripts/combine_and_model_play_pop_response.py
mv Extracted* ../

#
cd /home/jack/Dropbox/model_nl/uncued/0059/
mkdir play_r_p_only play_pop_avg
mv Extracted* play_r_p_only/
cd play_r_p_only/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_only.py
mv Extracted* ../play_pop_avg/
cd ../play_pop_avg/
python /home/jack/workspace/classification_scripts/combine_and_model_play_pop_response.py
mv Extracted* ../


cd /home/jack/Dropbox/model_nl/uncued/504/
mkdir play_r_p_only play_pop_avg
mv Extracted* play_r_p_only/
cd play_r_p_only/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_only.py
mv Extracted* ../play_pop_avg/
cd ../play_pop_avg/
python /home/jack/workspace/classification_scripts/combine_and_model_play_pop_response.py
mv Extracted* ../

#
cd /home/jack/Dropbox/model_nl/zero_or_three/0059/
mkdir play_r_p_only play_pop_avg
mv Extracted* play_r_p_only/
cd play_r_p_only/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_only.py
mv Extracted* ../play_pop_avg/
cd ../play_pop_avg/
python /home/jack/workspace/classification_scripts/combine_and_model_play_pop_response.py
mv Extracted* ../


cd /home/jack/Dropbox/model_nl/zero_or_three/504/
mkdir play_r_p_only play_pop_avg
mv Extracted* play_r_p_only/
cd play_r_p_only/
python /home/jack/workspace/classification_scripts/combine_and_model_play_r_p_only.py
mv Extracted* ../play_pop_avg/
cd ../play_pop_avg/
python /home/jack/workspace/classification_scripts/combine_and_model_play_pop_response.py
mv Extracted* ../




exec bash

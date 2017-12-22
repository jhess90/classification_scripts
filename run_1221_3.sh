#!/bin/sh

cd /home/jack/Dropbox/dpca/new/all_alt_cue/z_gauss/temp/

cd 0_3_10_1/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_10_2/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_10_3/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_13_1/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_13_2/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_13_3/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_14_1/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_14_2/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_14_3/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_27_1/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_27_2/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_28_1/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_28_2/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_28_3/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_9_1/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_9_2/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/
cd ../0_3_9_3/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../0059/


cd ../5_3_10_1/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../504/
cd ../5_3_10_2/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../504/
cd ../5_3_10_3/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../504/
cd ../5_3_13_1/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../504/
cd ../5_3_13_2/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../504/
cd ../5_3_13_3/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../504/
cd ../5_3_14_1/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../504/
cd ../5_3_14_2/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../504/
cd ../5_3_14_3/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../504/
cd ../5_3_28_2/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../504/
cd ../5_3_28_3/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../504/
cd ../5_3_9_1/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../504/
cd ../5_3_9_2/
python /home/jack/workspace/classification_scripts/model_3d_both.py
mv master* ../../504/

cd ../../0059/
mkdir with_d no_d
mv master*.npy with_d/
cd with_d/
python /home/jack/workspace/classification_scripts/dpca_setup_rp.py
mv master*.npy ../no_d/
cd ../no_d/
python /home/jack/workspace/classification_scripts/dpca_setup_binaryrp_nod.py
mv master*.npy ../

cd ../../504/
mkdir with_d no_d
mv master*.npy with_d/
cd with_d/
python /home/jack/workspace/classification_scripts/dpca_setup_rp.py
mv master*.npy ../no_d/
cd ../no_d/
python /home/jack/workspace/classification_scripts/dpca_setup_binaryrp_nod.py
mv master*.npy ../

mv /home/jack/Dropbox/dpca/new/all_alt_cue/z_gauss/temp/ /home/jack/Dropbox/new/all_alt_cue/

exec bash





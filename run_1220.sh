#!/bin/sh

cd /home/jack/Dropbox/dpca/new/

#cd cued/
#mkdir zscore z_gauss z_2after
#cd zscore
#mkdir 0059 504
#cd ../z_gauss/
#mkdir 0059 504
#cd ../

#cd zscore/
#mv /home/jack/Dropbox/dpca/new/cued/gaus30_bin10/temp/ .

#cd temp/
#cd 0_8_1/
#python /home/jack/workspace/classification_scripts/model_3d_zscore.py
cd /home/jack/Dropbox/dpca/new/cued/zscore/temp/0_8_1/
mv master* ../../0059/
0cd ../0_8_2/
python /home/jack/workspace/classification_scripts/model_3d_zscore.py
mv master* ../../0059/
cd ../0_9_1/
python /home/jack/workspace/classification_scripts/model_3d_zscore.py
mv master* ../../0059/
cd ../0_9_2/
python /home/jack/workspace/classification_scripts/model_3d_zscore.py
mv master* ../../0059/

cd ../5_8_1/
python /home/jack/workspace/classification_scripts/model_3d_zscore.py
mv master* ../../504/
cd ../5_8_2/
python /home/jack/workspace/classification_scripts/model_3d_zscore.py
mv master* ../../504/
cd ../5_9_1/
python /home/jack/workspace/classification_scripts/model_3d_zscore.py
mv master* ../../504/
cd ../5_9_2/
python /home/jack/workspace/classification_scripts/model_3d_zscore.py
mv master* ../../504/
cd ../5_14_1/
python /home/jack/workspace/classification_scripts/model_3d_zscore.py
mv master* ../../504/
cd ../5_14_2/
python /home/jack/workspace/classification_scripts/model_3d_zscore.py
mv master* ../../504/
cd ../5_14_3/
python /home/jack/workspace/classification_scripts/model_3d_zscore.py
mv master* ../../504/

cd ../../0059/
mkdir with_d no_d
cd with_d/
python /home/jack/workspace/classification_scripts/dpca_setup_rp.py
cd ../no_d/
python /home/jack/workspace/classification_scripts/dpca_setup_binaryrp_nod.py

cd ../../504/
mkdir with_d no_d
cd with_d/
python /home/jack/workspace/classification_scripts/dpca_setup_rp.py
cd ../no_d/
python /home/jack/workspace/classification_scripts/dpca_setup_binaryrp_nod.py


#####move npy to with_d, then no_d, then back before running



exec bash

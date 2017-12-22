#!/bin/sh

cd /home/jack/Dropbox/dpca/new/cued/z_2after/
mv /home/jack/Dropbox/dpca/new/cued/z_gauss/temp/ .

cd temp/
cd 0_8_1/
python /home/jack/workspace/classification_scripts/model_3d_zscore_2after.py
mv master* ../../0059/
cd ../0_8_2/
python /home/jack/workspace/classification_scripts/model_3d_zscore_2after.py
mv master* ../../0059/
cd ../0_9_1/
python /home/jack/workspace/classification_scripts/model_3d_zscore_2after.py
mv master* ../../0059/
cd ../0_9_2/
python /home/jack/workspace/classification_scripts/model_3d_zscore_2after.py
mv master* ../../0059/

cd ../5_8_1/
python /home/jack/workspace/classification_scripts/model_3d_zscore_2after.py
mv master* ../../504/
cd ../5_8_2/
python /home/jack/workspace/classification_scripts/model_3d_zscore_2after.py
mv master* ../../504/
cd ../5_9_1/
python /home/jack/workspace/classification_scripts/model_3d_zscore_2after.py
mv master* ../../504/
cd ../5_9_2/
python /home/jack/workspace/classification_scripts/model_3d_zscore_2after.py
mv master* ../../504/
cd ../5_14_1/
python /home/jack/workspace/classification_scripts/model_3d_zscore_2after.py
mv master* ../../504/
cd ../5_14_2/
python /home/jack/workspace/classification_scripts/model_3d_zscore_2after.py
mv master* ../../504/
cd ../5_14_3/
python /home/jack/workspace/classification_scripts/model_3d_zscore_2after.py
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


exec bash

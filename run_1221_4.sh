#!/bin/sh



cd /home/jack/Dropbox/dpca/new/all_alt_cue/z_2after/0059/
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





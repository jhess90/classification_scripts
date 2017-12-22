#!/bin/sh

/home/jack/Dropbox/dpca/new/all_alt_cue/zscore/504/with_d


cd /home/jack/Dropbox/dpca/new/all_alt_cue/zscore/504/
mv master*.npy with_d/
cd with_d/
python /home/jack/workspace/classification_scripts/dpca_setup_rp.py
mv master*.npy ../no_d/
cd ../no_d/
python /home/jack/workspace/classification_scripts/dpca_setup_binaryrp_nod.py
mv master*.npy ../

cd ../../504/
mv master*.npy with_d/
cd with_d/
python /home/jack/workspace/classification_scripts/dpca_setup_rp.py
mv master*.npy ../no_d/
cd ../no_d/
python /home/jack/workspace/classification_scripts/dpca_setup_binaryrp_nod.py
mv master*.npy ../


exec bash





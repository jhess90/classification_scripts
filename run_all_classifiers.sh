#!/bin/sh

#cd /home/jack/Dropbox/model_nl/rp_cue/0059/
#mkdir classifier_xgb/
#mv Extr* classifier_xgb/
#cd classifier_xgb/
#python /home/jack/workspace/classification_scripts/classifier_all_xgboost_crossval.py
#mv Extr* ../

#cd /home/jack/Dropbox/model_nl/rp_cue/504/
#mkdir classifier_xgb/
#mv Extr* classifier_xgb/
#cd classifier_xgb/
#python /home/jack/workspace/classification_scripts/classifier_all_xgboost_crossval.py
#mv Extr* ../

#
cd /home/jack/Dropbox/model_nl/alt_cue/0059/
#mkdir classifier_xgb/
#mv Extr* classifier_xgb/
cd classifier_xgb/
python /home/jack/workspace/classification_scripts/classifier_all_xgboost_crossval.py
mv Extr* ../

cd /home/jack/Dropbox/model_nl/alt_cue/504/
#mkdir classifier_xgb/
mv Extr* classifier_xgb/
cd classifier_xgb/
python /home/jack/workspace/classification_scripts/classifier_all_xgboost_crossval.py
mv Extr* ../

#
cd /home/jack/Dropbox/model_nl/single_cue/0059/
#mkdir classifier_xgb/
mv Extr* classifier_xgb/
cd classifier_xgb/
python /home/jack/workspace/classification_scripts/classifier_all_xgboost_crossval.py
mv Extr* ../

cd /home/jack/Dropbox/model_nl/single_cue/504/
mkdir classifier_xgb/
mv Extr* classifier_xgb/
cd classifier_xgb/
python /home/jack/workspace/classification_scripts/classifier_all_xgboost_crossval.py
mv Extr* ../

#
cd /home/jack/Dropbox/model_nl/single_uncued/0059/
mkdir classifier_xgb/
mv Extr* classifier_xgb/
cd classifier_xgb/
python /home/jack/workspace/classification_scripts/classifier_all_xgboost_crossval.py
mv Extr* ../

cd /home/jack/Dropbox/model_nl/single_uncued/504/
mkdir classifier_xgb/
mv Extr* classifier_xgb/
cd classifier_xgb/
python /home/jack/workspace/classification_scripts/classifier_all_xgboost_crossval.py
mv Extr* ../

#
cd /home/jack/Dropbox/model_nl/uncued/0059/
mkdir classifier_xgb/
mv Extr* classifier_xgb/
cd classifier_xgb/
python /home/jack/workspace/classification_scripts/classifier_all_xgboost_crossval.py
mv Extr* ../

cd /home/jack/Dropbox/model_nl/uncued/504/
mkdir classifier_xgb/
mv Extr* classifier_xgb/
cd classifier_xgb/
python /home/jack/workspace/classification_scripts/classifier_all_xgboost_crossval.py
mv Extr* ../

#
cd /home/jack/Dropbox/model_nl/zero_or_three/0059/
mkdir classifier_xgb/
mv Extr* classifier_xgb/
cd classifier_xgb/
python /home/jack/workspace/classification_scripts/classifier_all_xgboost_crossval.py
mv Extr* ../

cd /home/jack/Dropbox/model_nl/zero_or_three/504/
mkdir classifier_xgb/
mv Extr* classifier_xgb/
cd classifier_xgb/
python /home/jack/workspace/classification_scripts/classifier_all_xgboost_crossval.py
mv Extr* ../



exec bash

#!/bin/sh

cd /home/jack/Dropbox/model_nl/rp_cue/0059/
mv Extracted* play2/

cd play2/
python /home/jack/workspace/classification_scripts/combine_and_model_play2.py
#mv model_save.npy model_save_0059_2.npy
#cp model_save* ../../test_all
mv Extracted* ../play3

cd ../play3/
python /home/jack/workspace/classification_scripts/combine_and_model_play3.py
#mv model_save.npy model_save_0059_3.npy
#cp model_save* ../../test_all
mv Extracted* ../


#cd /home/jack/Dropbox/model_nl/rp_cue/504/
#mv Extracted* play/

#cd play/
#python /home/jack/workspace/classification_scripts/combine_and_model_play.py
#mv model_save.npy model_save_504_1.npy
#cp model_save* ../../test_all
#mv Extracted* ../play2

#cd ../play2/
#python /home/jack/workspace/classification_scripts/combine_and_model_play2.py
#mv model_save.npy model_save_504_2.npy
#cp model_save* ../../test_all
#mv Extracted* ../play3

#cd ../play3/
#python /home/jack/workspace/classification_scripts/combine_and_model_play3.py
#mv model_save.npy model_save_504_3.npy
#cp model_save* ../../test_all
#mv Extracted* ../



exec bash

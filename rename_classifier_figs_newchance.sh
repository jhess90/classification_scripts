#!/bin/sh

#make fig repository folder

cd /home/jack/Dropbox/model_nl/all_uncued/0059/
for f in chance*.png; do cp -- "$f" "/home/jack/Dropbox/fig_collection/classification_figs/new_chance/figs_S/all_uncued_$f"; done

cd /home/jack/Dropbox/model_nl/all_uncued/504/
for f in chance*.png; do cp -- "$f" "/home/jack/Dropbox/fig_collection/classification_figs/new_chance/figs_P/all_uncued_$f"; done

##

cd /home/jack/Dropbox/model_nl/rp_cue/0059/classifier_xgb_v2/
for f in chance*.png; do cp -- "$f" "/home/jack/Dropbox/fig_collection/classification_figs/new_chance/figs_S/rp_cue_$f"; done

cd /home/jack/Dropbox/model_nl/alt_cue/0059/classifier_xgb_v2/
for f in chance*.png; do cp -- "$f" "/home/jack/Dropbox/fig_collection/classification_figs/new_chance/figs_S/alt_cue_$f"; done

cd /home/jack/Dropbox/model_nl/single_cue/0059/classifier_xgb_v2/
for f in chance*.png; do cp -- "$f" "/home/jack/Dropbox/fig_collection/classification_figs/new_chance/figs_S/single_cue_$f"; done

cd /home/jack/Dropbox/model_nl/single_uncued/0059/classifier_xgb_v2/
for f in chance*.png; do cp -- "$f" "/home/jack/Dropbox/fig_collection/classification_figs/new_chance/figs_S/single_uncued_$f"; done

cd /home/jack/Dropbox/model_nl/uncued/0059/classifier_xgb_v2/
for f in chance*.png; do cp -- "$f" "/home/jack/Dropbox/fig_collection/classification_figs/new_chance/figs_S/uncued_$f"; done

cd /home/jack/Dropbox/model_nl/zero_or_three/0059/classifier_xgb_v2/
for f in chance*.png; do cp -- "$f" "/home/jack/Dropbox/fig_collection/classification_figs/new_chance/figs_S/zero_or_three_$f"; done


#######
cd /home/jack/Dropbox/model_nl/rp_cue/504/classifier_xgb_v2/
for f in chance*.png; do cp -- "$f" "/home/jack/Dropbox/fig_collection/classification_figs/new_chance/figs_P/rp_cue_$f"; done

cd /home/jack/Dropbox/model_nl/alt_cue/504/classifier_xgb_v2/
for f in chance*.png; do cp -- "$f" "/home/jack/Dropbox/fig_collection/classification_figs/new_chance/figs_P/alt_cue_$f"; done

cd /home/jack/Dropbox/model_nl/single_cue/504/classifier_xgb_v2/
for f in chance*.png; do cp -- "$f" "/home/jack/Dropbox/fig_collection/classification_figs/new_chance/figs_P/single_cue_$f"; done

cd /home/jack/Dropbox/model_nl/single_uncued/504/classifier_xgb_v2/
for f in chance*.png; do cp -- "$f" "/home/jack/Dropbox/fig_collection/classification_figs/new_chance/figs_P/single_uncued_$f"; done

cd /home/jack/Dropbox/model_nl/uncued/504/classifier_xgb_v2/
for f in chance*.png; do cp -- "$f" "/home/jack/Dropbox/fig_collection/classification_figs/new_chance/figs_P/uncued_$f"; done

cd /home/jack/Dropbox/model_nl/zero_or_three/504/classifier_xgb_v2/
for f in chance*.png; do cp -- "$f" "/home/jack/Dropbox/fig_collection/classification_figs/new_chance/figs_P/zero_or_three_$f"; done








exec bash

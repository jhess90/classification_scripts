#!/bin/sh

#cd /home/jack/Dropbox/dpca/new/all_alt_cue/0059/nod/
#pwd
#python /home/jack/classification_scripts/dpca_setup_binaryrp_nod.py
#python /home/jack/classification_scripts/analyze_dpca_nod.py
#Rscript /home/jack/classification_scripts/dpca_plot_analysis_nod.R

cd /home/jack/Dropbox/dpca/new/all_alt_cue/0059/mv/
pwd
#python /home/jack/classification_scripts/dpca_setup_alt_multi_m.py
#python /home/jack/classification_scripts/analyze_dpca_m.py
#Rscript /home/jack/classification_scripts/dpca_plot_analysis_m.R

#python /home/jack/classification_scripts/dpca_setup_alt_multi_v.py
python /home/jack/classification_scripts/analyze_dpca_v.py
Rscript /home/jack/classification_scripts/dpca_plot_analysis_v.R

cd /home/jack/Dropbox/dpca/new/all_alt_cue/504/mv/
pwd
#python /home/jack/classification_scripts/dpca_setup_alt_multi_m.py
#python /home/jack/classification_scripts/analyze_dpca_m.py
#Rscript /home/jack/classification_scripts/dpca_plot_analysis_m.R

python /home/jack/classification_scripts/dpca_setup_alt_multi_v.py
python /home/jack/classification_scripts/analyze_dpca_v.py
Rscript /home/jack/classification_scripts/dpca_plot_analysis_v.R



#cd /home/jack/Dropbox/dpca/new/cued/gaus30_bin10/0059/mv/
#pwd
#python /home/jack/classification_scripts/dpca_setup_multi_m.py
#python /home/jack/classification_scripts/analyze_dpca_m.py
#Rscript /home/jack/classification_scripts/dpca_plot_analysis_m.R

#python /home/jack/classification_scripts/dpca_setup_multi_v.py
#python /home/jack/classification_scripts/analyze_dpca_v.py
#Rscript /home/jack/classification_scripts/dpca_plot_analysis_v.R

cd /home/jack/Dropbox/dpca/new/cued/gaus30_bin10/504/mv/
pwd
python /home/jack/classification_scripts/dpca_setup_multi_m.py
python /home/jack/classification_scripts/analyze_dpca_m.py
Rscript /home/jack/classification_scripts/dpca_plot_analysis_m.R

python /home/jack/classification_scripts/dpca_setup_multi_v.py
python /home/jack/classification_scripts/analyze_dpca_v.py
Rscript /home/jack/classification_scripts/dpca_plot_analysis_v.R

exec bash



#!/bin/sh

#cd /Users/johnhessburg/dropbox/model/multi_rp/temp/0_8_1/test_corr/

####TODO
cd /home/jack/Dropbox/proposal_figs/corr_analysis/


#mkdir 5_8_1 5_8_2 5_9_1 5_9_2 5_14_1 5_14_2 5_14_3 0_8_1 0_8_2 0_9_1 0_9_2
#cd 5_8_1
#mkdir z_only g_only both_2 both_10 both_50
#cd ../5_8_2
#mkdir z_only g_only both_2 both_10 both_50
#cd ../5_9_1
#mkdir z_only g_only both_2 both_10 both_50
#cd ../5_9_2
#mkdir z_only g_only both_2 both_10 both_50
#cd ../5_14_1
#mkdir z_only g_only both_2 both_10 both_50
#cd ../5_14_2
#mkdir z_only g_only both_2 both_10 both_50
#cd ../5_14_3
#mkdir z_only g_only both_2 both_10 both_50
#
#cd ../0_8_1
#mkdir z_only g_only both_2 both_10 both_50
#cd ../0_8_2
#mkdir z_only g_only both_2 both_10 both_50
#cd ../0_9_1
#mkdir z_only g_only both_2 both_10 both_50
#cd ../0_9_2
#mkdir z_only g_only both_2 both_10 both_50

#cd ../
mv temp/5_8_1/Extracted* 5_8_1/z_only/
cd 5_8_1/z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../../
#mv temp/5_8_1/z_only/Extracted* 5_8_1/g_only/
cd  5_8_1/g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_2/
cd ../../
#mv temp/5_8_1/Extracted* 5_8_1/both_2/
cd 5_8_1/both_2/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_2.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../../
#mv temp/5_8_1/Extracted* 5_8_1/both_10/
cd 5_8_1/both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_50/
cd ../../
#mv temp/5_8_1/Extracted* 5_8_1/both_50/
cd 5_8_1/both_50/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_50.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* /home/jack/Dropbox/proposal_figs/corr_analysis/temp/5_8_1/
cd ../../

cd ../
mv temp/5_8_2/Extracted* 5_8_2/z_only/
cd 5_8_2/z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../../
#mv temp/5_8_2/Extracted* 5_8_2/g_only/
cd  5_8_2/g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_2/
cd ../../
#mv temp/5_8_2/Extracted* 5_8_2/both_2/
cd 5_8_2/both_2/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_2.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
cd ../../
#mv temp/5_8_2/Extracted* 5_8_2/both_10/
cd 5_8_2/both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_50/
cd ../../
#mv temp/5_8_2/Extracted* 5_8_2/both_50/
cd 5_8_2/both_50/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_50.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* /home/jack/Dropbox/proposal_figs/corr_analysis/temp/5_8_2/
cd ../../

cd ../
mv temp/5_9_1/Extracted* 5_9_1/z_only/
cd 5_9_1/z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../../
#mv temp/5_9_1/Extracted* 5_9_1/g_only/
cd  5_9_1/g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_2/
cd ../../
#mv temp/5_9_1/Extracted* 5_9_1/both_2/
cd 5_9_1/both_2/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_2.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../../
#mv temp/5_9_1/Extracted* 5_9_1/both_10/
cd 5_9_1/both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_50/
cd ../../
#mv temp/5_9_1/Extracted* 5_9_1/both_50/
cd 5_9_1/both_50/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_50.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.
mv Extracted* /home/jack/Dropbox/proposal_figs/corr_analysis/temp/5_9_1/
cd ../../

cd ../
mv temp/5_9_1/Extracted* 5_9_1/z_only/
cd 5_9_2/z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../../
#mv temp/5_9_2/Extracted* 5_9_2/g_only/
cd  5_9_2/g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_2/
cd ../../
#mv temp/5_9_2/Extracted* 5_9_2/both_2/
cd 5_9_2/both_2/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_2.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../../
#mv temp/5_9_2/Extracted* 5_9_2/both_10/
cd 5_9_2/both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_50/
cd ../../
#mv temp/5_9_2/Extracted* 5_9_2/both_50/
cd 5_9_2/both_50/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_50.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* /home/jack/Dropbox/proposal_figs/corr_analysis/temp/5_9_2/
cd ../../

cd ../
mv temp/5_14_1/Extracted* 5_14_1/z_only/
cd 5_14_1/z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../../
#mv temp/5_14_1/Extracted* 5_14_1/g_only/
cd  5_14_1/g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_2/
cd ../../
#mv temp/5_14_1/Extracted* 5_14_1/both_2/
cd 5_14_1/both_2/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_2.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../../
#mv temp/5_14_1/Extracted* 5_14_1/both_10/
cd 5_14_1/both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_50/
cd ../../
#mv temp/5_14_1/Extracted* 5_14_1/both_50/
cd 5_14_1/both_50/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_50.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* /home/jack/Dropbox/proposal_figs/corr_analysis/temp/5_14_1/
cd ../../

cd ../
mv temp/5_14_2/Extracted* 5_14_2/z_only/
cd 5_14_2/z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../../
#mv temp/5_14_2/Extracted* 5_14_2/g_only/
cd  5_14_2/g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_2/
cd ../../
#mv temp/5_14_2/Extracted* 5_14_2/both_2/
cd 5_14_2/both_2/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_2.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../../
#mv temp/5_14_2/Extracted* 5_14_2/both_10/
cd 5_14_2/both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_50/
cd ../../
#mv temp/5_14_2/Extracted* 5_14_2/both_50/
cd 5_14_2/both_50/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_50.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* /home/jack/Dropbox/proposal_figs/corr_analysis/temp/5_14_2/
cd ../../

cd ../
mv temp/5_14_3/Extracted* 5_14_3/z_only/
cd 5_14_3/z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../../
#mv temp/5_14_3/Extracted* 5_14_3/g_only/
cd  5_14_3/g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R\
mv Extracted* ../both_2/
cd ../../
#mv temp/5_14_3/Extracted* 5_14_3/both_2/
cd 5_14_3/both_2/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_2.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../../
#mv temp/5_14_3/Extracted* 5_14_3/both_10/
cd 5_14_3/both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_50
cd ../../
#mv temp/5_14_3/Extracted* 5_14_3/both_50/
cd 5_14_3/both_50/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_50.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* /home/jack/Dropbox/proposal_figs/corr_analysis/temp/5_14_3/
cd ../../

cd ../
mv temp/0_8_1/Extracted* 0_8_1/z_only/
cd 0_8_1/z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../../
#mv temp/0_8_1/Extracted* 0_8_1/g_only/
cd  0_8_1/g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_2/
cd ../../
#mv temp/0_8_1/Extracted* 0_8_1/both_2/
cd 0_8_1/both_2/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_2.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../../
#mv temp/0_8_1/Extracted* 0_8_1/both_10/
cd 0_8_1/both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_50/
cd ../../
#mv temp/0_8_1/Extracted* 0_8_1/both_50/
cd 0_8_1/both_50/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_50.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* /home/jack/Dropbox/proposal_figs/corr_analysis/temp/0_8_1/
cd ../../

cd ../
mv temp/0_8_2/Extracted* 0_8_2/z_only/
cd 0_8_2/z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../../
#mv temp/0_8_2/Extracted* 0_8_2/g_only/
cd  0_8_2/g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_2
cd ../../
#mv temp/0_8_2/Extracted* 0_8_2/both_2/
cd 0_8_2/both_2/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_2.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../../
#mv temp/0_8_2/Extracted* 0_8_2/both_10/
cd 0_8_2/both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_50/
cd ../../
#mv temp/0_8_2/Extracted* 0_8_2/both_50/
cd 0_8_2/both_50/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_50.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* /home/jack/Dropbox/proposal_figs/corr_analysis/temp/0_8_2/
cd ../../

cd ../
mv temp/0_9_1/Extracted* 0_9_1/z_only/
cd 0_9_1/z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../../
#mv temp/0_9_1/Extracted* 0_9_1/g_only/
cd  0_9_1/g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_2/
cd ../../
#mv temp/0_9_1/Extracted* 0_9_1/both_2/
cd 0_9_1/both_2/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_2.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../../
#mv temp/0_9_1/Extracted* 0_9_1/both_10/
cd 0_9_1/both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_50/
cd ../../
#mv temp/0_9_1/Extracted* 0_9_1/both_50/
cd 0_9_1/both_50/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_50.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* /home/jack/Dropbox/proposal_figs/corr_analysis/temp/0_9_1/
cd ../../

cd ../
mv temp/0_9_2/Extracted* 0_9_2/z_only/
cd 0_9_2/z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../../
#mv temp/0_9_2/Extracted* 0_9_2/g_only/
cd  0_9_2/g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_2/
cd ../../
#mv temp/0_9_2/Extracted* 0_9_2/both_2/
cd 0_9_2/both_2/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_2.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../../
#mv temp/0_9_2/Extracted* 0_9_2/both_10/
cd 0_9_2/both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_50/
cd ../../
#mv temp/0_9_2/Extracted* 0_9_2/both_50/
cd 0_9_2/both_50/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_50.py
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* /home/jack/Dropbox/proposal_figs/corr_analysis/temp/0_9_2/
cd ../../






exec bash

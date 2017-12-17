#!/bin/sh

cd /home/jack/Dropbox/corr_analysis/corr_temp/

cd /0_3_10_1/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_10_2/
mkdir z_only g_only both_10 
cd ../0_3_10_3/
mkdir z_only g_only both_10 
cd ../0_3_13_1/
mkdir z_only g_only both_10 
cd ../0_3_13_2/
mkdir z_only g_only both_10 
cd ../0_3_13_3/
mkdir z_only g_only both_10 
cd ../0_3_14_1/
mkdir z_only g_only both_10 
cd ../0_3_14_2/
mkdir z_only g_only both_10 
cd ../0_3_14_3/
mkdir z_only g_only both_10 
cd ../0_3_27_1/
mkdir z_only g_only both_10 
cd ../0_3_27_2/
mkdir z_only g_only both_10 
cd ../0_3_28_1/
mkdir z_only g_only both_10 
cd ../0_3_28_2/
mkdir z_only g_only both_10 
cd ../0_3_28_3/
mkdir z_only g_only both_10 
cd ../0_3_10_2/
mkdir z_only g_only both_10 
cd ../0_3_9_1/
mkdir z_only g_only both_10 
cd ../0_3_9_2/
mkdir z_only g_only both_10 
cd ../0_3_9_3/
mkdir z_only g_only both_10 
cd ../5_3_10_1/
mkdir z_only g_only both_10 
cd ../5_3_10_2/
mkdir z_only g_only both_10 
cd ../5_3_10_3/
mkdir z_only g_only both_10 
cd ../5_3_13_1/
mkdir z_only g_only both_10 
cd ../5_3_13_2/
mkdir z_only g_only both_10 
cd ../5_3_13_3/
mkdir z_only g_only both_10 
cd ../5_3_14_1/
mkdir z_only g_only both_10 
cd ../5_3_14_2/
mkdir z_only g_only both_10 
cd ../5_3_14_3/
mkdir z_only g_only both_10 
cd ../5_3_28_2/
mkdir z_only g_only both_10 
cd ../5_3_28_3/
mkdir z_only g_only both_10 
cd ../5_3_9_1/
mkdir z_only g_only both_10 
cd ../5_3_9_2/
mkdir z_only g_only both_10 
cd ../



#cd /home/jack/Dropbox/proposal_figs/corr_analysis/
#mv temp/5_8_1/Extracted* 5_8_1/z_only/
#cd 5_8_1/_only/
#pwd
#python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
#Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
#mv Extracted* ../g_only/
#cd ../../
#mv temp/5_8_1/z_only/Extracted* 5_8_1/g_only/
#cd  5_8_1/g_only/
#pwd
#python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
#Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
#mv Extracted* ../both_2/
#cd ../../
#mv temp/5_8_1/Extracted* 5_8_1/both_2/
#cd 5_8_1/both_2/
#pwd
#python /home/jack/workspace/classification_scripts/corr_analysis_both_2.py
#Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
#mv Extracted* ../both_10/
#cd ../../
#mv temp/5_8_1/Extracted* 5_8_1/both_10/
#cd 5_8_1/both_10/
#pwd
#python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
#Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
#mv Extracted* ../both_50/
#cd ../../
##mv temp/5_8_1/Extracted* 5_8_1/both_50/
#cd 5_8_1/both_50/
#pwd
#python /home/jack/workspace/classification_scripts/corr_analysis_both_50.py
#Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
#mv Extracted* /home/jack/Dropbox/proposal_figs/corr_analysis/temp/5_8_1/
#cd ../../

#cd ../
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

#cd ../
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

#cd ../
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

#cd ../
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

#cd ../
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

#cd ../
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

#cd ../
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

#cd ../
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

#cd ../
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

#cd ../
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

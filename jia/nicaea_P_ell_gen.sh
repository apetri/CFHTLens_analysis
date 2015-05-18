#bash file to generate nicaea power spectrum

cd /Users/jia/Documents/code/nicaea_2.4/Demo
for i in {0..10785}
do 
echo ------------------------------${i}-------------------------------
cp /Users/jia/CFHTLenS/emulator/nicaea_params/cosmo$i.par ./cosmo.par
./lensingdemo
cp P_kappa /Users/jia/CFHTLenS/emulator/nicaea_params/P_kappa$i
done
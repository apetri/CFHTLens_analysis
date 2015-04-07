#!bash

f=/work/02977/jialiu/KSsim
cd $f
for i in  `ls $f`
do echo $i
tar -cf /scratch/02977/jialiu/ranch_archive/KSsim/$i.tar $i
done

f=/work/02977/jialiu/KSsim_noiseless
cd $f
for i in  `ls $f`
do echo $i
tar -cf /scratch/02977/jialiu/ranch_archive/KSsim_noiseless/$i.tar $i
done

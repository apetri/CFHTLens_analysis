#!bash

f=/work/02977/jialiu/KSsim
cd $f
for i in  `ls $f`
do echo $i
echo tar -cvf $i.tar $i
done

f=/work/02977/jialiu/KSsim_noiseless
cd $f
for i in  `ls $f`
do echo $i
echo tar -cvf $i.tar $i
done

for i in  `ls /home3/02918/apetri/Catalogues/emulator_galpos_zcut0213`
do echo $i
VAL=`sfind /home3/02918/apetri/Catalogues/emulator_galpos_zcut0213/$i -offline | wc -l`
echo ${VAL}
if [ ${VAL} -gt 0 ];
then
stage -rw /home3/02918/apetri/Catalogues/emulator_galpos_zcut0213/$i
fi
done


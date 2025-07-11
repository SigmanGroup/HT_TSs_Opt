#!/bin/bash
file=$1
module load gcc/8.5.0
module load openmpi/4.1.1
module load orca/5.0.3

WORKDIR=$PWD
mkdir $WORKDIR/${file%".xyz"}_input
cp $WORKDIR/${file} $WORKDIR/${file%".xyz"}_input/
cd $WORKDIR/${file%".xyz"}_input/

o4wb3c --struc ${file} --charge 0 --uhf 1 

cp wb97x3c.inp $WORKDIR/${file%".xyz"}.inp 
cd $WORKDIR
rm -rf $WORKDIR/${file%".xyz"}_input

sed -i 's/wB97X-D4/wB97X-D4 miniprint nopop/g' ${file%".xyz"}.inp
sed -i 's/nprocs   4/nprocs   16/g' ${file%".xyz"}.inp
sed -i '/%pal/i %cpcm\n  epsilon 37.8\n  refrac 1.4384\nend\n' ${file%".xyz"}.inp

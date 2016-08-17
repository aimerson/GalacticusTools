#! /bin/tcsh

set QUEUE = "longq"
set RUNS = "2-17"

if ($#argv > 0) then
    set OUTDIR = $argv[1]
    echo "OUTPUT DIRECTORY = $OUTDIR"
else
    echo "*** ERROR! No Galacticus output directory specified!"
    exit
endif

set logdir = /nobackupNFS/sunglass/amerson/logs/galacticus/
mkdir -p $logdir

qsub -V -q $QUEUE -o $logdir -j oe -J $RUNS -v OUTDIR=$OUTDIR ./postprocess_output.csh

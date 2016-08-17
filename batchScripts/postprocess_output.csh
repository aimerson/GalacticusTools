#! /bin/tcsh
#PBS -S /bin/tcsh

echo "POSTPROCESSING GALACTICUS OUTPUT"

if( $?PBS_O_WORKDIR ) then
    echo "PBS WORKDIR = $PBS_O_WORKDIR"
    cd $PBS_O_WORKDIR    
endif
if( $?PBS_JOBID ) then
    echo "JOB_NAME = $PBS_JOBNAME"
    echo "QUEUE = $PBS_QUEUE"
endif
if( $?PBS_ARRAY_INDEX ) then
    echo "JOB_ID = $PBS_JOBID"    
endif

# Get path of file to process
if ($#argv > 0) then
    set OFILE = $argv[1]
else
    if( $?PBS_ARRAY_INDEX ) then
	set OFILE = ${OUTDIR}/galacticus_${PBS_ARRAY_INDEX}.hdf5
    else
        if( $?OUTFILE ) then
	    set OFILE = ${OUTFILE}
	endif
    endif
endif
if ($?OFILE ) then
    echo "OUTFILE = $OFILE"
else
    echo "*** ERROR! No Galacticus output file specified! Use either command line argument or environment variable 'OUTFILE'"
    exit
endif
echo "Post-processing Galacticus output file..."

echo "Creating copy of HDF5 file..."
set RAWfile = `echo $OFILE | sed -e "s/.hdf5/.RAW.hdf5/g"`
cp $OFILE $RAWfile

# Run post-processing scripts...

set perlScript =  ${GALACTICUS_ROOT_V094}/scripts/analysis/computeBasicTotals.pl
$perlScript $OFILE

#set perlScript =  ${GALACTICUS_ROOT_V094}/scripts/analysis/dustExtinguish.pl
#$perlScript $OFILE

set perlScript =  ${GALACTICUS_ROOT_V094}/scripts/analysis/addEmLines.pl
$perlScript $OFILE

set perlScript =  ${GALACTICUS_ROOT_V094}/scripts/analysis/addMagnitudes.pl
$perlScript $OFILE

#set perlScript =  ${GALACTICUS_ROOT_V094}/scripts/analysis/addEquivalentWidths.pl
#$perlScript $OFILE

echo "POST-PROCESSING COMPLETE"


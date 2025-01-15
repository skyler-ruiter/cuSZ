#! /usr/bin/bash

# get user input for error type and bound from command line args
args=("$@")
if [ $# -ne 2 ]; then
  echo "Usage: ./run_tests.sh <error_type> <error_bound>"
  exit 1
fi

# set error type and bound
ERROR_TYPE=${args[0]}
ERROR_BOUND=${args[1]}
echo "Error type: ${ERROR_TYPE} | Error bound: ${ERROR_BOUND}"

# set home directory for machine
CUSZ=${HOME}/sz_compression/cusz-dev
CUSZ_BIN=${CUSZ}/build/cusz

# load environment
source ${CUSZ}/profiling/scripts/env.sh

# delete any previous output files and throw away output
source ${CUSZ}/profiling/scripts/delete_comp.sh > /dev/null

# change EXAALT files to end in .f32 instead of .dat2
for datum in "${EXAALT_DIR}"/*.dat2; do
  if [ -f "$datum" ]; then
    echo "Renaming ${datum} to ${datum}.f32"
    mv ${datum} ${datum}.f32
  else
    echo "No .dat2 files found in ${EXAALT_DIR}"
  fi
done

# set data directory
DATA_DIR=${CUSZ}/profiling/data

# make an output directory
mkdir -p ${CUSZ}/profiling/output
OUTPUT_DIR=${CUSZ}/profiling/output

CESM_DIMS=3600x1800
EXAALT_DIMS=2869440
HURR_DIMS=500x500x100
HACC_M_DIMS=280953867

CESM_DIR=${DATA_DIR}/CESM
EXAALT_DIR=${DATA_DIR}/EXAALT
HURR_DIR=${DATA_DIR}/HURR
HACC_M_DIR=${DATA_DIR}/HACC_M

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DATA=(CESM EXAALT HURR HACC_M)

# run on each dataset
for data in "${DATA[@]}"; do
  echo "Running cusz on ${data} data"
  echo

  dims_var="${data}_DIMS"
  dims=${!dims_var}
  echo "${dims_var} = ${dims}"
  echo

  data_dir="${data}_DIR"

  # for each file in the dataset
  for datum in ${!data_dir}/*.f32; do

    ############################

    if [ -f "$datum" ]; then

      # make an output directory (/CESM, /EXAALT, /HURR, /HACC_M)
      mkdir -p ${OUTPUT_DIR}/${data}

      # run cusz 5 times with compression and decompression
      for i in {1..5}; do
        # make an output file
        OUTPUT=${OUTPUT_DIR}/${data}/$(basename ${datum}).cusz

        output_rep=${OUTPUT_DIR}/${data}/$(basename ${datum})_rep${i}.txt

        # run cusz with compression and report time and compression ratio
        CMD="${CUSZ_BIN} -t f32 -m ${ERROR_TYPE} -e ${ERROR_BOUND} -i ${datum} -l ${dims} -z --report time,cr"
        echo ${CMD}

        eval ${CMD} > ${output_rep}

        # run cusz with decompression and report time
        CMD_DECOMP="${CUSZ_BIN} -i ${datum}.cusza -x --report time --compare ${datum} > ${OUTPUT_DIR}/${data}/$(basename ${datum})_decomp_rep${i}.txt"
        eval ${CMD_DECOMP}
      done

    else
      echo "No .f32 files found in ${data}"
      echo ${!data}
    fi

    ############################

  done # file
done # dataset

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # Run cleaner
python3 ${CUSZ}/profiling/scripts/output_cleaner.py

# # run kernel data collector
python3 ${CUSZ}/profiling/scripts/kernel_data_collector.py

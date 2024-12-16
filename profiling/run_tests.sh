#! /usr/bin/bash

# set home directory for machine
# CUSZ=${HOME}/sz_compression/cusz-dev
CUSZ=${HOME}/cusz-dev

# load environment
source ${CUSZ}/profiling/env.sh

CESM=1800x3600
CESM_DIMS=3600x1800
EXAALT=2869440
EXAALT_DIMS=2869440
HURR=100x500x500
HURR_DIMS=500x500x100
HACC_M=280953867
HACC_M_DIMS=280953867
DATA_DIR=${CUSZ}/data

# make an output directory
mkdir -p ${CUSZ}/profiling/output
OUTPUT_DIR=${CUSZ}/profiling/output

CESM_DIR=${DATA_DIR}/${CESM}
EXAALT_DIR=${DATA_DIR}/${EXAALT}
HURR_DIR=${DATA_DIR}/${HURR}
HACC_M_DIR=${DATA_DIR}/${HACC_M}

DATA=(CESM EXAALT HURR HACC_M)

CUSZ_BIN=${CUSZ}/build/cusz

# chance EXAALT files to end in .f32 instead of .dat2
for datum in "${EXAALT_DIR}"/*.dat2; do
  if [ -f "$datum" ]; then
    echo "Renaming ${datum} to ${datum}.f32"
    mv ${datum} ${datum}.f32
  else
    echo "No .dat2 files found in ${EXAALT_DIR}"
  fi
done

# delete any previous output files and throw away output
source ${CUSZ}/profiling/delete_comp.sh > /dev/null

# run on each dataset
for data in "${DATA[@]}"; do
  echo "Running cusz on ${data} data"

  # for each file in the dataset
  for datum in ${!data}/*.f32; do

    ############################

    if [ -f "$datum" ]; then
      echo "Running cusz on ${datum}"

      # make an output directory (/CESM, /EXAALT, /HURR, /HACC_M)
      mkdir -p ${OUTPUT_DIR}/${data}

      # run cusz 5 times with compression and decompression
      for i in {1..5}; do
        # make an output file
        OUTPUT=${OUTPUT_DIR}/${data}/$(basename ${datum}).cusz

        # run cusz with compression and report time and compression ratio
        CMD="${CUSZ_BIN} -t f32 -m r2r -e 1e-4 -i ${datum} -l ${${!data}_DIMS} -z --report time,cr > ${OUTPUT_DIR}/${data}/$(basename ${datum})_rep${i}.txt"
        eval ${CMD}

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

# Run cleaner
python3 ${CUSZ}/profiling/output_cleaner.py

# run kernel data collector
python3 ${CUSZ}/profiling/kernel_data_collector.py

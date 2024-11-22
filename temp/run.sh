#!/bin/bash

#repository
CUSZ=${HOME}/sz_compression/cusz-dev
DATA_DIR=${CUSZ}/data

CESM=1800x3600
EXAALT=2869440
HURR=100x500x500
HACC_M=280953867

# make an output directory
mkdir -p ${CUSZ}/temp/output
OUTPUT_DIR=${CUSZ}/temp/output

CESM_DIR=${DATA_DIR}/${CESM}
EXAALT_DIR=${DATA_DIR}/${EXAALT}
HURR_DIR=${DATA_DIR}/${HURR}
HACC_M_DIR=${DATA_DIR}/${HACC_M}

CUSZ_BIN=${CUSZ}/build/cusz

# run cusz 

echo "Running cusz on CESM data"

for datum in "${CESM_DIR}"/*.f32; do
  if [ -f "$datum" ]; then
    echo "Running cusz on ${datum}"
    mkdir -p ${OUTPUT_DIR}/CESM
    for i in {1..5}; do
      OUTPUT=${OUTPUT_DIR}/CESM/$(basename ${datum}).cusz
      CMD="${CUSZ_BIN} -t f32 -m r2r -e 1e-4 -i ${datum} -l 3600x1800 -z --report time,cr > ${OUTPUT_DIR}/CESM/$(basename ${datum})_rep${i}.txt"
      eval ${CMD}

      CMD_DECOMP="${CUSZ_BIN} -i ${datum}.cusza -x --report time > ${OUTPUT_DIR}/CESM/$(basename ${datum})_decomp_rep${i}.txt"
      eval ${CMD_DECOMP}
    done
  else
    echo "No .f32 files found in ${CESM_DIR}"
  fi
done

python3 ${CUSZ}/temp/data.py ${OUTPUT_DIR}/CESM

echo "Running cusz on EXAALT data"

for datum in "${EXAALT_DIR}"/*.dat2; do
  if [ -f "$datum" ]; then
    echo "Running cusz on ${datum}"
    mkdir -p ${OUTPUT_DIR}/EXAALT
    for i in {1..5}; do
      OUTPUT=${OUTPUT_DIR}/EXAALT/$(basename ${datum}).cusz
      CMD="${CUSZ_BIN} -t f32 -m r2r -e 1e-4 -i ${datum} -l 2869440 -z --report time,cr > ${OUTPUT_DIR}/EXAALT/$(basename ${datum})_rep${i}.txt"
      eval ${CMD}

      CMD_DECOMP="${CUSZ_BIN} -i ${datum}.cusza -x --report time > ${OUTPUT_DIR}/EXAALT/$(basename ${datum})_decomp_rep${i}.txt"
      eval ${CMD_DECOMP}
    done
  else
    echo "No .f32 files found in ${EXAALT_DIR}"
  fi
done

python3 ${CUSZ}/temp/data.py ${OUTPUT_DIR}/EXAALT

echo "Running cusz on HURR data"

for datum in "${HURR_DIR}"/*.f32; do
  if [ -f "$datum" ]; then
    echo "Running cusz on ${datum}"
    mkdir -p ${OUTPUT_DIR}/HURR
    for i in {1..5}; do
      OUTPUT=${OUTPUT_DIR}/HURR/$(basename ${datum}).cusz
      CMD="${CUSZ_BIN} -t f32 -m r2r -e 1e-4 -i ${datum} -l 500x500x100 -z --report time,cr > ${OUTPUT_DIR}/HURR/$(basename ${datum})_rep${i}.txt"
      eval ${CMD}

      CMD_DECOMP="${CUSZ_BIN} -i ${datum}.cusza -x --report time > ${OUTPUT_DIR}/HURR/$(basename ${datum})_decomp_rep${i}.txt"
      eval ${CMD_DECOMP}
    done
  else
    echo "No .f32 files found in ${HURR_DIR}"
  fi
done

python3 ${CUSZ}/temp/data.py ${OUTPUT_DIR}/HURR

echo "Running cusz on HACC_M data"

for datum in "${HACC_M_DIR}"/*.f32; do
  if [ -f "$datum" ]; then
    echo "Running cusz on ${datum}"
    mkdir -p ${OUTPUT_DIR}/HACC_M
    for i in {1..5}; do
      OUTPUT=${OUTPUT_DIR}/HACC_M/$(basename ${datum}).cusz
      CMD="${CUSZ_BIN} -t f32 -m r2r -e 1e-4 -i ${datum} -l 2869440 -z --report time,cr > ${OUTPUT_DIR}/HACC_M/$(basename ${datum})_rep${i}.txt"
      eval ${CMD}

      CMD_DECOMP="${CUSZ_BIN} -i ${datum}.cusza -x --report time > ${OUTPUT_DIR}/HACC_M/$(basename ${datum})_decomp_rep${i}.txt"
      eval ${CMD_DECOMP}
    done
  else
    echo "No .f32 files found in ${HACC_M_DIR}"
  fi
done

python3 ${CUSZ}/temp/data.py ${OUTPUT_DIR}/HACC_M
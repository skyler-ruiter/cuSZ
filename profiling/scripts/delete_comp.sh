#! /usr/bin/bash

#DATA_DIR=${HOME}/sz_compression/cusz-dev/data
DATA_DIR=${HOME}/sz_compression/cusz-dev/profiling/data

CESM=1800x3600
EXAALT=2869440
HURR=100x500x500
HACC_M=280953867

CESM_DIR=${DATA_DIR}/${CESM}
EXAALT_DIR=${DATA_DIR}/${EXAALT}
HURR_DIR=${DATA_DIR}/${HURR}
HACC_M_DIR=${DATA_DIR}/${HACC_M}

DATA=(CESM_DIR EXAALT_DIR HURR_DIR HACC_M_DIR)

# for each dataset, delete alll .cusza files
for data in "${DATA[@]}"; do
  echo "Deleting .cusza files in ${data} data"

  for datum in "${!data}"/*.cusza; do
    if [ -f "$datum" ]; then
      echo "Deleting ${datum}"
      rm ${datum}
    else
      echo "No .cusza files found in ${!data}"
    fi
  done

  for datum in "${!data}"/*.cuszx; do
    if [ -f "$datum" ]; then
      echo "Deleting ${datum}"
      rm ${datum}
    else
      echo "No .cusza files found in ${!data}"
    fi
  done
done
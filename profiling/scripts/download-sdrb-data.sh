#!/bin/bash -
#title           :sh.download-sdrb-data
#description     :This script will download sample dataset from SDRBench.
#author          :Cody Rivera
#copyright       :(C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
#license         :See LICENSE in top-level directory
#date            :2020-09-28
#version         :0.1
#usage           :./sh.download-sdrb-data data-dir
#==============================================================================

banner()
{
    echo "+------------------------------------------+"
    printf "| %-40s |\n" "`date`"
    echo "|                                          |"
    printf "|`tput bold` %-40s `tput sgr0`|\n" "$@"
    echo "+------------------------------------------+"
}

sbanner()
{
    echo "+------------------------------------------+"
    printf "|`tput bold` %-40s `tput sgr0`|\n" "$@"
    echo "+------------------------------------------+"
}

pwd; hostname; date;

if [ "$#" -ne 1 ]; then
    echo "usage: $0 data-dir";
    exit 2;
fi

if ! mkdir -p $1; then
    echo "Cannot download data to $1";
    exit 2;
fi

DATA_DIR=$1

banner "Downloading Data";

banner "CESM";
if [ ! -f "$DATA_DIR/SDRBENCH-CESM-ATM-1800x3600.tar.gz" ]; then
    wget -c -P $DATA_DIR https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/CESM-ATM/SDRBENCH-CESM-ATM-1800x3600.tar.gz
fi

if [ ! -d "$DATA_DIR/CESM" ]; then
    tar -C $DATA_DIR -xvf $DATA_DIR/SDRBENCH-CESM-ATM-1800x3600.tar.gz
fi

banner "EXAALT";
if [ ! -f "$DATA_DIR/SDRBENCH-EXAALT-2869440.tar.gz" ]; then
    wget -c -P $DATA_DIR https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXAALT/SDRBENCH-EXAALT-2869440.tar.gz
fi
  
if [ ! -d "$DATA_DIR/EXAALT" ]; then
    tar -C $DATA_DIR -xvf $DATA_DIR/SDRBENCH-EXAALT-2869440.tar.gz
fi

banner "Hurricane Isabel";
if [ ! -f "$DATA_DIR/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz" ]; then
    wget -c -P $DATA_DIR https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz
fi

if [ ! -d "$DATA_DIR/HURR" ]; then
    tar -C $DATA_DIR -xvf $DATA_DIR/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz
fi

# !!! LARGE DATASETS !!! -- test script will not fail if these aren't present
banner "HACC 1GB";
if [ ! -f "$DATA_DIR/EXASKY-HACC-data-medium-size.tar.gz" ]; then
    wget -c -P $DATA_DIR https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/HACC/EXASKY-HACC-data-medium-size.tar.gz
fi

if [ ! -d "$DATA_DIR/HACC" ]; then
    tar -C $DATA_DIR -xvf $DATA_DIR/EXASKY-HACC-data-medium-size.tar.gz
fi

# banner "HACC 4GB";
# if [ ! -f "$DATA_DIR/EXASKY-HACC-data-big-size.tar.gz" ]; then
#     wget -c -P $DATA_DIR https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/HACC/EXASKY-HACC-data-big-size.tar.gz
# fi

# if [ ! -d "$DATA_DIR/1billionparticles_onesnapshot" ]; then
#     tar -C $DATA_DIR -xvf $DATA_DIR/EXASKY-HACC-data-big-size.tar.gz
# fi

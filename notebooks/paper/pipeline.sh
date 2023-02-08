#!/bin/bash
VIEW=${1:-tab_adult}
SIZE=${2:-500k}
PASTEUR=${PASTEUR:-venv/bin/pasteur}

case $VIEW in
    tab_adult)
        DATASET=adult
        PARAMS=
        SUFFIX=
        ;;
    mimic_tab_admissions)
        DATASET=mimic
        PARAMS=
        SUFFIX=
        ;;
    mimic_billion)
        DATASET=mimic
        case $SIZE in
            1M)
                PARAMS="ratios.wrk=0.001 ratios.ref=0.001 alg.e1=0.3 alg.e2=0.7 alg.e=1 alg.batched=False"
                SUFFIX=1m.
                ;;
            10M)
                PARAMS="ratios.wrk=0.01 ratios.ref=0.01 alg.e1=0.03 alg.e2=0.07 alg.e=0.1 alg.batched=False"
                SUFFIX=10m.
                ;;
            100M)
                PARAMS="ratios.wrk=0.1 ratios.ref=0.02 alg.e1=0.003 alg.e2=0.007 alg.e=0.01"
                SUFFIX=100m.
                ;;
            500M)
                PARAMS="ratios.wrk=0.5 ratios.ref=0.02 alg.e1=0.0006 alg.e2=0.0014 alg.e=0.002" # AIM -> "random_state=512"
                SUFFIX=500m.
                ;;
            1B)
                PARAMS="ratios.wrk=0.98 ratios.ref=0.02 alg.e1=0.0003 alg.e2=0.0007 alg.e=0.001"
                SUFFIX=1b.
                ;;
            500Msingle)
                PARAMS="ratios.wrk=0.5 ratios.ref=0.02 alg.e1=0.0006 alg.e2=0.0014 alg.e=0.002" # AIM -> "random_state=512"
                SUFFIX=500m.
                export _DEBUG=1
                ;;
        esac
        ;;
    texas_billion)
        DATASE=texas
        PARAMS=
        SUFFIX=1b.
        ;;
esac

# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# $SCRIPT_DIR/ramuse.sh &

function pause(){
   read -p "Finished $*, press [Enter] key to continue..."
}

echo Run the following command once to load the dataset and write down the time:
echo time $PASTEUR p $DATASET.ingest
pause printing dataset message

time $PASTEUR p $VIEW.ingest $PARAMS
pause ingest

time $PASTEUR p $VIEW.privbayes $PARAMS alg.rebalance=False --synth
pause Privbayes rebalance=False
rm -r data/synth

time $PASTEUR p $VIEW.privbayes $PARAMS alg.rebalance=True --synth
pause Privbayes rebalance=True
rm -r data/synth

time $PASTEUR p $VIEW.aim $PARAMS --synth
rm -r data/synth

# pkill -P $$
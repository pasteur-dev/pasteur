
PASTEUR=${PASTEUR:-venv/bin/pasteur}

mkdir -p external
echo raw.csv
time pasteur export mimic_billion.wrk.table ./external/mimic_billion.wrk.raw.1b.csv
echo raw.csv.gz
time pasteur export mimic_billion.wrk.table ./external/mimic_billion.wrk.raw.1b.csv.gz
echo raw.pq
time pasteur export mimic_billion.wrk.table ./external/mimic_billion.wrk.raw.1b.pq

echo idx.csv
time pasteur export mimic_billion.wrk.idx_table ./external/mimic_billion.wrk.idx.1b.csv
echo idx.csv.gz
time pasteur export mimic_billion.wrk.idx_table ./external/mimic_billion.wrk.idx.1b.csv.gz
echo idx.pq
time pasteur export mimic_billion.wrk.idx_table ./external/mimic_billion.wrk.idx.1b.pq

ls -lah external
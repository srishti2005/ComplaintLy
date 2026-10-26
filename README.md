# ComplaintLy Intelligence

### Distributed complaint analytics on a Dockerized Hadoop cluster (HDFS + YARN) with PySpark, Spark SQL, Parquet and Streamlit

ComplaintLy Intelligence is the Big Data module of the **ComplaintLy** project. It processes ~18.1M consumer complaints from the CFPB database on a multi-node Hadoop cluster (3 DataNodes, 2 NodeManagers) run locally with Docker Compose, and serves the results through an interactive Streamlit dashboard.

> **Related components**
> - Complaint classifier API (Flask, TF-IDF + Logistic Regression): `app.py`, `models/` in this repo
> - Web frontend (React): [ComplaintLy_frontend](https://github.com/srishti2005/ComplaintLy_frontend)

---

## Table of contents
1. [Problem statement](#problem-statement)
2. [Dataset](#dataset)
3. [Architecture](#architecture)
4. [Technology stack](#technology-stack)
5. [Repository layout](#repository-layout)
6. [Hadoop cluster setup](#hadoop-cluster-setup)
7. [Load data into HDFS](#load-data-into-hdfs)
8. [Run the pipeline on YARN](#run-the-pipeline-on-yarn)
9. [Dashboard](#dashboard)
10. [Cluster configuration](#cluster-configuration)
11. [Operations](#operations)
12. [Troubleshooting](#troubleshooting)
13. [Results](#results)
14. [Roadmap](#roadmap)

---

## Problem statement

Consumer complaint data spans millions of records (products, companies, issues, locations, response types, timelines). Single-machine processing becomes slow as volume grows.

Objectives:
- Store the raw dataset on **HDFS** with block-level replication
- Clean and transform the data with **PySpark** executing on **YARN**
- Persist year-partitioned **Parquet** back to HDFS
- Run large-scale analytical queries with **Spark SQL**
- Explore results in an interactive **Streamlit** dashboard
- Make the whole stack reproducible with **Docker**

## Dataset

**CFPB Consumer Complaint Database** (U.S. Consumer Financial Protection Bureau)
Source: https://www.consumerfinance.gov/data-research/consumer-complaints/

| Property | Value |
|---|---|
| Records | ~18.1 million |
| Raw size | ~5.5 GB CSV |
| Key fields | Complaint ID, date received, product, sub-product, issue, sub-issue, company, state, ZIP, submitted via, company response, timely response |

The CSV is **not committed** to Git (`hadoop-cluster/data/` is git-ignored). Download it manually (see [Load data into HDFS](#load-data-into-hdfs)).

## Architecture

```
        CFPB Consumer Complaint CSV (~5.5 GB)
                       │  hdfs dfs -put
                       ▼
        HDFS  /complaintly/raw        (128 MB blocks, replication 2)
                       │
                       ▼
        PySpark on YARN  (ingest → clean → transform → repartition)
                       │
                       ▼
        HDFS  /complaintly/processed  (year-partitioned Parquet)
                       │
                       ▼
        Spark SQL  (product / company / state analysis)
                       │
                       ▼
        HDFS  /complaintly/analytics  (aggregated Parquet)
                       │  hdfs dfs -get
                       ▼
        Streamlit dashboard  (Plotly)  →  Docker  →  Browser
```

**Cluster topology (Docker Compose)**

| Service | Role | Replicas | Exposed |
|---|---|---|---|
| `namenode` | HDFS metadata | 1 | `9870` (HDFS UI) |
| `datanode` | HDFS block storage | 3 | – |
| `resourcemanager` | YARN scheduler | 1 | `8088` (YARN UI) |
| `nodemanager` | YARN container runtime | 2 | – |
| `client` | Gateway: `hdfs`, `spark-submit`, your jobs | 1 | – |

## Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Distributed storage | Apache Hadoop HDFS (image `apache/hadoop:3`) |
| Resource manager | Apache Hadoop YARN |
| Processing | Apache Spark 3.5.1 (PySpark), Spark SQL |
| Storage format | Parquet (pandas 2.1.4, numpy 1.26.4, pyarrow 15.0.2 in the cluster image) |
| Dashboard | Streamlit, Plotly |
| Containerization | Docker, Docker Compose |
| Runtime | Java 1.8 (bundled with the Hadoop image) |
| Dataset | CFPB Consumer Complaint Database |

## Repository layout

```
ComplaintLy/
├── bigdata/                  # PySpark jobs + Streamlit dashboard
│   ├── src/                  # ingestion / transformation / analytics scripts
│   ├── dashboard.py
│   ├── Dockerfile            # dashboard container
│   └── requirements.txt
├── hadoop-cluster/           # Dockerized Hadoop + YARN cluster
│   ├── Dockerfile            # Hadoop image + Python 3.11 env + PySpark
│   ├── docker-compose.yml
│   ├── config                # Hadoop/YARN settings (env-style)
│   └── data/                 # complaints.csv (git-ignored)
├── models/                   # classifier artifacts (Flask API)
├── app.py                    # Flask classifier API
└── README.md
```

## Hadoop cluster setup

### Prerequisites
- Docker Desktop (Windows: WSL2 backend)
- **At least 8 GB RAM allocated to Docker** (10 GB recommended), 4+ CPUs
- ~20 GB free disk (CSV + 2× HDFS replicas + images)

**Windows (WSL2):** set the memory limit in `%UserProfile%\.wslconfig`, then run `wsl --shutdown` and restart Docker Desktop:
```ini
[wsl2]
memory=10GB
processors=6
swap=4GB
```
Verify: `docker info --format "{{.MemTotal}}"` should print about 10 GB in bytes.

### Build and start
```bash
cd hadoop-cluster
docker compose build            # first build is slow (~10-50 min depending on network)
docker compose up -d --scale datanode=3 --scale nodemanager=2
docker compose ps
```

### Verify
```bash
docker compose exec client hdfs dfsadmin -report     # Live datanodes (3)
docker compose exec client yarn node -list           # Total Nodes: 2
docker compose run --rm client python --version      # Python 3.11.x
docker compose run --rm client spark-submit --version  # Spark 3.5.1
```
- HDFS UI: http://localhost:9870
- YARN UI: http://localhost:8088

### Smoke test (Spark on YARN)
```bash
docker compose exec client spark-submit --master yarn --deploy-mode client \
  --num-executors 2 --executor-cores 1 --executor-memory 1g \
  /opt/pyenv/lib/python3.11/site-packages/pyspark/examples/src/main/python/pi.py 20
```
Expected output: `Pi is roughly 3.14...`

## Load data into HDFS

1. Download the full CSV from the CFPB site, unzip, and place it at `hadoop-cluster/data/complaints.csv`.
2. Load it:
```bash
docker compose exec client hdfs dfs -mkdir -p /complaintly/raw /complaintly/processed /complaintly/analytics
docker compose exec client hdfs dfs -put /data/complaints.csv /complaintly/raw/
docker compose exec client hdfs dfs -ls -h /complaintly/raw
docker compose exec client hdfs fsck /complaintly/raw -files -blocks
```
Expected: ~44 blocks of 128 MB, 2 replicas each (~11 GB of raw disk).

## Run the pipeline on YARN

The `client` container mounts `bigdata/` at `/jobs`. Scripts read and write HDFS paths:

```python
BASE      = "hdfs://namenode:8020/complaintly"
RAW       = f"{BASE}/raw/complaints.csv"
PROCESSED = f"{BASE}/processed"     # year-partitioned Parquet
ANALYTICS = f"{BASE}/analytics"
```

Submit jobs (replace the placeholders with your script names under `bigdata/src/`):

```bash
# 1. Ingest, clean, transform, write partitioned Parquet
docker compose exec client spark-submit --master yarn --deploy-mode client \
  --num-executors 2 --executor-cores 1 --executor-memory 1g \
  --conf spark.sql.shuffle.partitions=48 \
  /jobs/src/<ingest_transform_script>.py

# 2. Spark SQL analytics
docker compose exec client spark-submit --master yarn --deploy-mode client \
  --num-executors 2 --executor-cores 1 --executor-memory 1g \
  /jobs/src/<analytics_script>.py
```

Monitor applications at http://localhost:8088 and verify output:
```bash
docker compose exec client hdfs dfs -ls /complaintly/processed     # year=YYYY folders
docker compose exec client hdfs dfs -du -h /complaintly/analytics
```

**Optional speed-up:** stage Spark's jars in HDFS once so each submit skips the ~10 s upload:
```bash
docker compose exec client bash -c "hdfs dfs -mkdir -p /spark-jars && hdfs dfs -put /opt/pyenv/lib/python3.11/site-packages/pyspark/jars/* /spark-jars/"
# then add to spark-submit:
#   --conf spark.yarn.jars=hdfs://namenode:8020/spark-jars/*.jar
```

## Dashboard

The analytics output is small, so copy it out of HDFS and let Streamlit read it locally:
```bash
docker compose exec client hdfs dfs -get /complaintly/analytics /data/analytics
```
Run Streamlit from the repo root, pointing it at `hadoop-cluster/data/analytics`:
```bash
pip install -r bigdata/requirements.txt
streamlit run bigdata/dashboard.py        # http://localhost:8501
```
Or with Docker (see `bigdata/Dockerfile`; port `8501`).

> TODO: document how `dashboard.py` locates its data path (env var or hardcoded) after the HDFS migration.

## Cluster configuration

Defined in `hadoop-cluster/config` (injected by the `apache/hadoop` image as XML):

| Setting | Value | Why |
|---|---|---|
| `fs.defaultFS` | `hdfs://namenode:8020` | Default filesystem |
| `dfs.replication` | `2` | Fault tolerance with 3 DataNodes |
| `mapreduce.framework.name` | `yarn` | Run on YARN |
| `yarn.nodemanager.resource.memory-mb` | `4096` | Per-NodeManager memory cap |
| `yarn.nodemanager.resource.cpu-vcores` | `2` | Per-NodeManager vcores |
| `yarn.scheduler.maximum-allocation-mb` | `4096` | Max container size |
| `yarn.nodemanager.pmem-check-enabled` / `vmem-check-enabled` | `false` | Avoid containers killed by memory checks in Docker |

**Why a custom image?** `apache/hadoop:3` is CentOS 7-based (end-of-life repos, old glibc, no Python 3.8+). `hadoop-cluster/Dockerfile` installs Miniforge, builds an isolated **Python 3.11** env at `/opt/pyenv`, installs pandas/numpy/pyarrow via conda and PySpark via pip, and sets `PYSPARK_PYTHON=/opt/pyenv/bin/python`. Driver and YARN executors share the same image, so Python versions always match.

## Operations

| Task | Command |
|---|---|
| Pause cluster (keeps HDFS data) | `docker compose stop` |
| Resume | `docker compose start` |
| Shell into the gateway | `docker compose exec client bash` |
| Logs | `docker compose logs <service> --tail 50` |
| **Destroy everything incl. HDFS data** | `docker compose down` |

> **Warning:** `docker compose down` deletes the containers, and with them all HDFS data (no volumes are defined). Use `stop` / `start` for day-to-day work. Re-run the [load step](#load-data-into-hdfs) after a `down`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `yum` errors, `mirrorlist.centos.org` unresolved | CentOS 7 is EOL. Don't use `yum`; the provided Dockerfile uses Miniforge instead. |
| `conda install python=3.10` solver conflict | Latest Miniforge ships Python 3.14. Create a fresh env (`conda create -p /opt/pyenv python=3.11`) instead of downgrading base. |
| `pip install pandas` fails with `Unknown compiler(s): cc, gcc` | No prebuilt wheel for the old glibc, so pip builds from source. Install pandas/numpy/pyarrow with **conda**, PySpark with pip. |
| YARN app stuck in `ACCEPTED` | Not enough memory. Raise Docker RAM or lower `--executor-memory` / `--num-executors`. |
| `Connection refused namenode:8020` | Run commands from inside the Docker network (`docker compose exec client ...`). |
| `Python in worker has different version` | Rebuild all services from the same image: `docker compose build --no-cache`. |
| NameNode won't start after a crash | `docker compose down` then `up` and reload data (wipes HDFS). |
| Windows: slow I/O or path errors | Keep the repo inside the WSL2 filesystem (`~/projects`), not `/mnt/c/...`. |

## Results

> Fill in after running the full pipeline.

| Metric | Value |
|---|---|
| Records processed | ~18.1M |
| Raw CSV size | ~5.5 GB |
| Parquet output size | _TBD_ |
| Ingest + transform runtime (local mode) | _TBD_ |
| Ingest + transform runtime (YARN, 2 executors) | _TBD_ |
| Spark SQL analytics runtime | _TBD_ |

Screenshots to add: HDFS UI (3 live DataNodes), YARN UI (completed application), dashboard.

## Roadmap

- Persistent HDFS volumes (separate `datanode1..3` services) so `down` doesn't wipe data
- Stage Spark jars in HDFS (`spark.yarn.jars`) permanently
- Orchestrate the pipeline (Airflow or a `Makefile`)
- Scale-out run on AWS EMR with `s3://` paths
- Wire the classifier (`app.py`) to consume the processed Parquet for retraining

---

*Dataset: U.S. Consumer Financial Protection Bureau, Consumer Complaint Database.*

# ComplaintLy Intelligence
### Distributed Complaint Analytics using PySpark, Spark SQL, Streamlit and Docker

ComplaintLy Intelligence is a Big Data analytics module built on top of the ComplaintLy project. It processes millions of consumer complaints using Apache PySpark and generates interactive analytics through a Streamlit dashboard.

The project demonstrates a complete Big Data pipeline from large-scale data ingestion and distributed processing to analytical visualization and Docker-based deployment.

---

## Problem Statement

Consumer complaint datasets contain millions of records covering products, companies, issues, locations, response types and complaint timelines.

Analyzing such a large dataset using traditional single-machine processing can become inefficient as the data volume increases.

The objective of this project is to build a scalable Big Data analytics pipeline that can:

- Process millions of consumer complaint records
- Clean and transform the raw dataset
- Distribute the processed data using Spark partitions
- Store the processed dataset in Parquet format
- Perform large-scale analytical queries using Spark SQL
- Provide an interactive dashboard for data exploration
- Containerize the dashboard using Docker

---

## Dataset

The project uses the **Consumer Complaint Database** published by the U.S. Consumer Financial Protection Bureau (CFPB).

Dataset source:

https://www.consumerfinance.gov/data-research/consumer-complaints/

The downloaded dataset contains approximately **18.1 million complaint records**.

### Important fields

- Complaint ID
- Date received
- Product
- Sub-product
- Issue
- Sub-issue
- Company
- State
- ZIP code
- Submitted via
- Company response to consumer
- Timely response

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Big Data Processing | Apache PySpark |
| Query Engine | Spark SQL |
| Data Storage | Parquet |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Containerization | Docker |
| Dataset | CFPB Consumer Complaint Database |

---

## Architecture

```text
                CFPB Consumer Complaint Dataset
                         │
                         ▼
                 Raw CSV (~5.5 GB)
                         │
                         ▼
                 PySpark Data Ingestion
                         │
                         ▼
              Data Cleaning & Transformation
                         │
                         ▼
                 Spark Repartitioning
                         │
                         ▼
              Year-Partitioned Parquet Data
                         │
                         ▼
                    Spark SQL
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Product        Company         State
       Analysis       Analysis       Analysis
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                 Analytics Parquet
                         │
                         ▼
                  Streamlit Dashboard
                         │
                         ▼
                  Docker Container
                         │
                         ▼
                    Web Browser

\# ComplaintLy Intelligence

\### Distributed Complaint Analytics using PySpark, Spark SQL, Streamlit and Docker



ComplaintLy Intelligence is a Big Data analytics module built on top of the ComplaintLy project. It processes millions of consumer complaints using Apache PySpark and generates interactive analytics through a Streamlit dashboard.



The project demonstrates a complete Big Data pipeline from large-scale data ingestion and distributed processing to analytical visualization and Docker-based deployment.



\---



\## Problem Statement



Consumer complaint datasets contain millions of records covering products, companies, issues, locations, response types and complaint timelines.



Analyzing such a large dataset using traditional single-machine processing can become inefficient as the data volume increases.



The objective of this project is to build a scalable Big Data analytics pipeline that can:



\- Process millions of consumer complaint records

\- Clean and transform the raw dataset

\- Distribute the processed data using Spark partitions

\- Store the processed dataset in Parquet format

\- Perform large-scale analytical queries using Spark SQL

\- Provide an interactive dashboard for data exploration

\- Containerize the dashboard using Docker



\---



\## Dataset



The project uses the \*\*Consumer Complaint Database\*\* published by the U.S. Consumer Financial Protection Bureau (CFPB).



Dataset source:



https://www.consumerfinance.gov/data-research/consumer-complaints/



The downloaded dataset contains approximately \*\*18.1 million complaint records\*\*.



\### Important fields



\- Complaint ID

\- Date received

\- Product

\- Sub-product

\- Issue

\- Sub-issue

\- Company

\- State

\- ZIP code

\- Submitted via

\- Company response to consumer

\- Timely response



\---



\## Technology Stack



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



\---



\## Architecture



```text

&#x20;               CFPB Consumer Complaint Dataset

&#x20;                        │

&#x20;                        ▼

&#x20;                Raw CSV (\~5.5 GB)

&#x20;                        │

&#x20;                        ▼

&#x20;                PySpark Data Ingestion

&#x20;                        │

&#x20;                        ▼

&#x20;             Data Cleaning \& Transformation

&#x20;                        │

&#x20;                        ▼

&#x20;                Spark Repartitioning

&#x20;                        │

&#x20;                        ▼

&#x20;             Year-Partitioned Parquet Data

&#x20;                        │

&#x20;                        ▼

&#x20;                   Spark SQL

&#x20;                        │

&#x20;         ┌──────────────┼──────────────┐

&#x20;         ▼              ▼              ▼

&#x20;      Product        Company         State

&#x20;      Analysis       Analysis       Analysis

&#x20;         │              │              │

&#x20;         └──────────────┼──────────────┘

&#x20;                        ▼

&#x20;                Analytics Parquet

&#x20;                        │

&#x20;                        ▼

&#x20;                 Streamlit Dashboard

&#x20;                        │

&#x20;                        ▼

&#x20;                 Docker Container

&#x20;                        │

&#x20;                        ▼

&#x20;                   Web Browser



Project Structure

bigdata/

│

├── dashboard.py

├── Dockerfile

├── requirements.txt

├── README.md

├── .gitignore

│

└── src/

&#x20;   ├── analytics.py

&#x20;   ├── etl.py

&#x20;   ├── inspect\_data.py

&#x20;   ├── spark\_session.py

&#x20;   └── test\_spark.py



The raw dataset and processed Parquet files are intentionally excluded from GitHub using .gitignore because of their large size.

Big Data ETL Pipeline

The ETL pipeline is implemented using PySpark.

1\. Data Ingestion

The raw CSV dataset is loaded using Spark's CSV reader with schema inference and support for multiline records.

2\. Data Cleaning

String columns are cleaned using trimming operations and missing categorical values are replaced with Unknown.

3\. Feature Engineering

Additional time-based features are generated:

\- Year

\- Month

\- YearMonth

4\. Repartitioning

The dataset is repartitioned into 16 Spark partitions before writing the processed data.

5\. Parquet Storage

The processed dataset is stored in Parquet format and partitioned by year.

Example:

complaints\_parquet/

├── Year=2011/

├── Year=2012/

├── ...

├── Year=2025/

└── Year=2026/



Spark SQL Analytics

Spark SQL is used to perform analytical queries on the processed complaint data.

The project analyzes:

Product Analysis

Complaint volume across different financial products.

Company Analysis

Companies with the highest complaint volumes.

Complaint volume should not be interpreted directly as company quality because larger companies may naturally receive more complaints due to their customer base and market share.



Geographic Analysis

Complaint distribution across U.S. states.

Monthly Trends

Complaint volume over time using monthly aggregation.

Response Analysis

Analysis of company responses to consumers.

Timely Response Analysis

Comparison of complaints receiving timely versus non-timely responses.

Issue Analysis

Identification of the most frequently reported complaint issues.

Streamlit Dashboard

The Streamlit dashboard provides an interactive interface for exploring the processed analytics.

Dashboard Features

\- KPI cards

\- Year filtering

\- Top-N controls

\- Monthly complaint trends

\- Product analysis

\- Company complaint volume

\- State-wise complaint distribution

\- Company response analysis

\- Timely response analysis

\- Top complaint issues

\- Interactive Plotly visualizations

\- Analytical data tables

The dashboard reads the processed analytics outputs rather than repeatedly processing the original 5.5 GB CSV.

Docker

The Streamlit application is containerized using Docker.

Build Docker Image

From the bigdata directory:

docker build -t complaintly-bigdata .



Run Container

docker run -d -p 8501:8501 --name complaintly-dashboard complaintly-bigdata



Open Dashboard

http://localhost:8501



The application has been successfully tested inside a Docker container.

Running Without Docker

Create and activate a Python virtual environment:

python -m venv venv



Windows PowerShell:

.\\venv\\Scripts\\Activate.ps1



Install dependencies:

pip install -r requirements.txt



Run the Streamlit dashboard:

streamlit run dashboard.py



Running the PySpark Pipeline

Test Spark

python src/test\_spark.py



Inspect Dataset

python src/inspect\_data.py



Run ETL

python src/etl.py



Run Analytics

python src/analytics.py



The ETL pipeline generates the processed Parquet dataset, while the analytics pipeline generates the datasets consumed by the Streamlit dashboard.

Results

The pipeline successfully processed approximately:

18,145,013 complaint records

Major analytical dimensions include:

\- Product

\- Company

\- State

\- Monthly trends

\- Company response

\- Timely response

\- Complaint issue

The processed dataset is stored as a partitioned Parquet dataset and the resulting analytics are visualized through the Streamlit dashboard.

Big Data Concepts Demonstrated

This project demonstrates practical usage of:

\- Distributed data processing

\- Spark partitions

\- PySpark DataFrames

\- Spark SQL

\- ETL pipelines

\- Data cleaning

\- Feature engineering

\- Parquet storage

\- Partitioned datasets

\- Large-scale aggregation

\- Interactive analytics

\- Docker containerization

Project Objective

The overall objective is to demonstrate how a large consumer complaint dataset can be transformed into an analytical system using Big Data technologies while providing an accessible interface for exploring the results.

Author

Srishti Tripathi

B.Tech — Artificial Intelligence \& Machine Learning

Symbiosis Institute of Technology, Pune


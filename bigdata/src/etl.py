from spark_session import create_spark_session
from pyspark.sql.functions import col, trim, year, month, date_format

spark = create_spark_session()

INPUT_PATH = "../data/raw/complaints.csv"
OUTPUT_PATH = "../data/processed/complaints_parquet"

print("\n========================================")
print("     COMPLAINTLY BIG DATA ETL")
print("========================================\n")

# --------------------------------------------------
# 1. READ DATASET
# --------------------------------------------------

print("Reading dataset...")

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("multiLine", True)
    .option("escape", '"')
    .csv(INPUT_PATH)
)

print(f"Raw records: {df.count():,}")
print(f"Input partitions: {df.rdd.getNumPartitions()}")

# --------------------------------------------------
# 2. CLEAN IMPORTANT COLUMNS
# --------------------------------------------------

print("\nCleaning data...")

string_columns = [
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Company",
    "State",
    "ZIP code",
    "Submitted via",
    "Company response to consumer",
    "Timely response?"
]

for column_name in string_columns:
    df = df.withColumn(
        column_name,
        trim(col(column_name))
    )

df = df.fillna({
    "Product": "Unknown",
    "Sub-product": "Unknown",
    "Issue": "Unknown",
    "Sub-issue": "Unknown",
    "Company": "Unknown",
    "State": "Unknown",
    "Submitted via": "Unknown",
    "Company response to consumer": "Unknown",
    "Timely response?": "Unknown"
})

# --------------------------------------------------
# 3. CREATE TIME FEATURES
# --------------------------------------------------

print("Creating time features...")

df = (
    df
    .withColumn("Year", year(col("Date received")))
    .withColumn("Month", month(col("Date received")))
    .withColumn(
        "YearMonth",
        date_format(col("Date received"), "yyyy-MM")
    )
)

# --------------------------------------------------
# 4. SELECT RELEVANT COLUMNS
# --------------------------------------------------

df = df.select(
    "Complaint ID",
    "Date received",
    "Year",
    "Month",
    "YearMonth",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Company",
    "State",
    "ZIP code",
    "Submitted via",
    "Company response to consumer",
    "Timely response?"
)

# --------------------------------------------------
# 5. WRITE DISTRIBUTED PARQUET
# --------------------------------------------------

print("\nWriting distributed Parquet dataset...")

(
    df
    .repartition(16)
    .write
    .mode("overwrite")
    .partitionBy("Year")
    .parquet(OUTPUT_PATH)
)

print("\n========================================")
print("       ETL COMPLETED SUCCESSFULLY")
print("========================================")

print(f"Output location: {OUTPUT_PATH}")

spark.stop()
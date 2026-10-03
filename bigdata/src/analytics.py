from spark_session import create_spark_session
from pyspark.sql.functions import col, desc, count, sum as spark_sum


# ============================================================
# COMPLAINTLY BIG DATA ANALYTICS
# ============================================================

spark = create_spark_session()

INPUT_PATH = "../data/processed/complaints_parquet"


print("\n========================================")
print("   COMPLAINTLY BIG DATA ANALYTICS")
print("========================================\n")


# ------------------------------------------------------------
# 1. READ PROCESSED PARQUET DATA
# ------------------------------------------------------------

print("Reading processed Parquet dataset...")

df = spark.read.parquet(INPUT_PATH)

print(f"Total records: {df.count():,}")
print(f"Number of partitions: {df.rdd.getNumPartitions()}")

df.createOrReplaceTempView("complaints")

print("\nDataset loaded successfully.")


# ============================================================
# ANALYSIS 1: COMPLAINTS BY PRODUCT
# ============================================================

print("\n========================================")
print("1. COMPLAINTS BY PRODUCT")
print("========================================")

product_analysis = spark.sql("""
    SELECT
        Product,
        COUNT(*) AS complaint_count
    FROM complaints
    GROUP BY Product
    ORDER BY complaint_count DESC
""")

product_analysis.show(15, truncate=False)


# ============================================================
# ANALYSIS 2: TOP COMPANIES BY COMPLAINT VOLUME
# ============================================================

print("\n========================================")
print("2. TOP COMPANIES BY COMPLAINT VOLUME")
print("========================================")

company_analysis = spark.sql("""
    SELECT
        Company,
        COUNT(*) AS complaint_count
    FROM complaints
    WHERE Company IS NOT NULL
      AND Company != 'Unknown'
    GROUP BY Company
    ORDER BY complaint_count DESC
    LIMIT 15
""")

company_analysis.show(15, truncate=False)


# ============================================================
# ANALYSIS 3: COMPLAINTS BY STATE
# ============================================================

print("\n========================================")
print("3. COMPLAINTS BY STATE")
print("========================================")

state_analysis = spark.sql("""
    SELECT
        State,
        COUNT(*) AS complaint_count
    FROM complaints
    WHERE State IS NOT NULL
      AND State != 'Unknown'
    GROUP BY State
    ORDER BY complaint_count DESC
    LIMIT 15
""")

state_analysis.show(15, truncate=False)


# ============================================================
# ANALYSIS 4: MONTHLY COMPLAINT TREND
# ============================================================

print("\n========================================")
print("4. MONTHLY COMPLAINT TREND")
print("========================================")

monthly_analysis = spark.sql("""
    SELECT
        YearMonth,
        COUNT(*) AS complaint_count
    FROM complaints
    WHERE YearMonth IS NOT NULL
    GROUP BY YearMonth
    ORDER BY YearMonth
""")

monthly_analysis.show(30, truncate=False)


# ============================================================
# ANALYSIS 5: COMPANY RESPONSE ANALYSIS
# ============================================================

print("\n========================================")
print("5. COMPANY RESPONSE ANALYSIS")
print("========================================")

response_analysis = spark.sql("""
    SELECT
        `Company response to consumer` AS response,
        COUNT(*) AS complaint_count
    FROM complaints
    GROUP BY `Company response to consumer`
    ORDER BY complaint_count DESC
""")

response_analysis.show(15, truncate=False)


# ============================================================
# ANALYSIS 6: TIMELY RESPONSE ANALYSIS
# ============================================================

print("\n========================================")
print("6. TIMELY RESPONSE ANALYSIS")
print("========================================")

timely_analysis = spark.sql("""
    SELECT
        `Timely response?` AS timely_response,
        COUNT(*) AS complaint_count
    FROM complaints
    GROUP BY `Timely response?`
    ORDER BY complaint_count DESC
""")

timely_analysis.show(truncate=False)


# ============================================================
# ANALYSIS 7: TOP COMPLAINT ISSUES
# ============================================================

print("\n========================================")
print("7. TOP COMPLAINT ISSUES")
print("========================================")

issue_analysis = spark.sql("""
    SELECT
        Issue,
        COUNT(*) AS complaint_count
    FROM complaints
    WHERE Issue IS NOT NULL
      AND Issue != 'Unknown'
    GROUP BY Issue
    ORDER BY complaint_count DESC
    LIMIT 20
""")

issue_analysis.show(20, truncate=False)


# ============================================================
# SAVE ANALYTICS RESULTS
# ============================================================

print("\n========================================")
print("SAVING ANALYTICS RESULTS")
print("========================================")


product_analysis.write.mode("overwrite").parquet(
    "../data/processed/analytics_product"
)

company_analysis.write.mode("overwrite").parquet(
    "../data/processed/analytics_company"
)

state_analysis.write.mode("overwrite").parquet(
    "../data/processed/analytics_state"
)

monthly_analysis.write.mode("overwrite").parquet(
    "../data/processed/analytics_monthly"
)

response_analysis.write.mode("overwrite").parquet(
    "../data/processed/analytics_response"
)

timely_analysis.write.mode("overwrite").parquet(
    "../data/processed/analytics_timely"
)

issue_analysis.write.mode("overwrite").parquet(
    "../data/processed/analytics_issue"
)


print("\n========================================")
print("     ANALYTICS COMPLETED SUCCESSFULLY")
print("========================================\n")


spark.stop()
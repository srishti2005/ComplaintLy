from spark_session import create_spark_session

spark = create_spark_session()

file_path = "../data/raw/complaints.csv"

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("multiLine", True)
    .option("escape", '"')
    .csv(file_path)
)

print("\n========== DATASET SCHEMA ==========\n")
df.printSchema()

print("\n========== TOTAL RECORDS ==========\n")
print(df.count())

print("\n========== COLUMNS ==========\n")
print(df.columns)

print("\n========== SAMPLE DATA ==========\n")
df.show(5, truncate=True)

print("\n========== PARTITIONS ==========\n")
print(df.rdd.getNumPartitions())

spark.stop()
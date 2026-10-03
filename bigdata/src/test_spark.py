from spark_session import create_spark_session


spark = create_spark_session()

print("\n================================")
print("      SPARK TEST SUCCESS")
print("================================")

print("Spark Version:", spark.version)

print("Default Parallelism:",
      spark.sparkContext.defaultParallelism)

print("Number of Cores:",
      spark.sparkContext.defaultParallelism)

spark.stop()
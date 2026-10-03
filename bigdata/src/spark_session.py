from pyspark.sql import SparkSession


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("ComplaintLy-BigData")
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.files.maxPartitionBytes", "128m")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark
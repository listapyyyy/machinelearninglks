"""
TechnoMart - AWS Glue ETL Job
==============================
ETL pipeline untuk memproses raw data dari S3/Glue Catalog menjadi
processed data (Parquet) yang siap untuk ML training.

Proses per dataset:
  - user_profiles       : cleaning, feature engineering, churn label
  - product_catalog     : cleaning, normalisasi harga, feature scoring
  - user_interactions   : dedup, implicit score, session features
  - transaction_history : cleaning, fraud signal, aggregasi per user & produk
  - product_reviews     : cleaning, text features, sentiment label
  - search_logs         : cleaning, search features, CTR label

Output: s3://technomart-s3/processed-data/<dataset>/
"""

import sys
import logging

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType
from pyspark.sql.window import Window

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Init Glue / Spark ──────────────────────────────────────────────────────────
args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc   = SparkContext()
gc   = GlueContext(sc)
spark = gc.spark_session
job  = Job(gc)
job.init(args["JOB_NAME"], args)

spark.conf.set("spark.sql.session.timeZone", "Asia/Jakarta")
spark.conf.set("spark.sql.shuffle.partitions", "200")

# ── Konstanta ──────────────────────────────────────────────────────────────────
GLUE_DB  = "raw-data" #yang menyimpan raw data di glue
S3_OUT   = "s3://technomart-s3/processed-data"
S3_RAW   = "s3://technomart-s3/raw-data"

# ── Helper: baca dari Glue Catalog ─────────────────────────────────────────────
def read_catalog(table_name: str):
    logger.info(f"Reading table: {GLUE_DB}.{table_name}")
    dyf = gc.create_dynamic_frame.from_catalog(
        database=table_name,
        table_name=table_name,
        transformation_ctx=f"read_{table_name}",
    )
    df = dyf.toDF()
    logger.info(f"  → {df.count():,} rows | {len(df.columns)} cols")
    return df

# ── Helper: tulis Parquet ke S3 ────────────────────────────────────────────────
def write_parquet(df, path: str, partition_by: list = None, n_partitions: int = 10):
    logger.info(f"Writing → {path}  (partitions={partition_by})")
    df = df.repartition(n_partitions)
    writer = df.write.mode("overwrite").option("compression", "snappy")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.parquet(path)
    logger.info(f"  ✅ Done: {path}")

# ── Helper: hapus duplikat & baris null kritis ─────────────────────────────────
def basic_clean(df, pk_cols: list, not_null_cols: list = None):
    before = df.count()
    df = df.dropDuplicates(pk_cols)
    if not_null_cols:
        df = df.dropna(subset=not_null_cols)
    after = df.count()
    logger.info(f"  clean: {before:,} → {after:,} rows (dropped {before-after:,})")
    return df


# ==============================================================================
# 1. USER PROFILES
# ==============================================================================
def process_user_profiles():
    df = read_catalog("user_profiles")
    df = basic_clean(df, pk_cols=["user_id"], not_null_cols=["user_id", "age"])

    df = (
        df
        # ── tipe data
        .withColumn("age",                    F.col("age").cast(IntegerType()))
        .withColumn("tenure_days",            F.col("tenure_days").cast(IntegerType()))
        .withColumn("loyalty_score",          F.col("loyalty_score").cast(DoubleType()))
        .withColumn("total_spend_idr",        F.col("total_spend_idr").cast(DoubleType()))
        .withColumn("avg_order_value_idr",    F.col("avg_order_value_idr").cast(DoubleType()))
        .withColumn("days_since_last_purchase", F.col("days_since_last_purchase").cast(IntegerType()))
        .withColumn("email_open_rate",        F.col("email_open_rate").cast(DoubleType()))

        # ── impute null numerik dengan median/0
        .fillna({
            "loyalty_score":             50.0,
            "total_spend_idr":           0.0,
            "avg_order_value_idr":       0.0,
            "days_since_last_purchase":  365,
            "email_open_rate":           0.0,
            "referral_count":            0,
        })

        # ── standarisasi teks
        .withColumn("gender",   F.upper(F.trim(F.col("gender"))))
        .withColumn("city",     F.initcap(F.trim(F.col("city"))))

        # ── feature engineering
        # kelompok usia
        .withColumn("age_group", F.when(F.col("age") < 25, "Gen-Z")
                                  .when(F.col("age") < 35, "Millennial")
                                  .when(F.col("age") < 45, "Gen-X")
                                  .otherwise("Boomer"))

        # recency bucket (mirip RFM)
        .withColumn("recency_bucket", F.when(F.col("days_since_last_purchase") <= 30, "active")
                                       .when(F.col("days_since_last_purchase") <= 90, "warm")
                                       .when(F.col("days_since_last_purchase") <= 180, "cooling")
                                       .otherwise("churned"))

        # spend tier (percentile-based pada runtime)
        .withColumn("spend_tier", F.ntile(4).over(
            Window.orderBy(F.col("total_spend_idr"))
        ))  # 1=low … 4=high

        # engagement score gabungan (0–1)
        .withColumn("engagement_score", F.round(
            (F.col("email_open_rate") * 0.3
             + F.col("loyalty_score") / 100 * 0.4
             + F.when(F.col("days_since_last_purchase") <= 30, 1.0)
               .when(F.col("days_since_last_purchase") <= 90, 0.6)
               .otherwise(0.2) * 0.3),
            4
        ))

        # flag akun baru (< 30 hari)
        .withColumn("is_new_user", (F.col("tenure_days") < 30).cast(IntegerType()))

        # metadata
        .withColumn("processed_at", F.current_timestamp())
        .withColumn("etl_version",  F.lit("v2.0"))
    )

    write_parquet(df, f"{S3_OUT}/user_profiles/", partition_by=["user_segment"])
    return df


# ==============================================================================
# 2. PRODUCT CATALOG
# ==============================================================================
def process_product_catalog():
    df = read_catalog("product_catalog")
    df = basic_clean(df, pk_cols=["product_id"], not_null_cols=["product_id", "category"])

    df = (
        df
        .withColumn("base_price_idr",   F.col("base_price_idr").cast(DoubleType()))
        .withColumn("final_price_idr",  F.col("final_price_idr").cast(DoubleType()))
        .withColumn("discount_pct",     F.col("discount_pct").cast(DoubleType()))
        .withColumn("avg_rating",       F.col("avg_rating").cast(DoubleType()))
        .withColumn("review_count",     F.col("review_count").cast(LongType()))
        .withColumn("stock_quantity",   F.col("stock_quantity").cast(IntegerType()))
        .withColumn("views_last_30d",   F.col("views_last_30d").cast(LongType()))
        .withColumn("sales_last_30d",   F.col("sales_last_30d").cast(LongType()))

        # harga tidak boleh negatif
        .filter(F.col("final_price_idr") > 0)

        # impute
        .fillna({"avg_rating": 0.0, "review_count": 0, "stock_quantity": 0})

        # standarisasi
        .withColumn("category",    F.initcap(F.trim(F.col("category"))))
        .withColumn("brand",       F.upper(F.trim(F.col("brand"))))
        .withColumn("availability", F.when(F.col("stock_quantity") > 0, "in_stock")
                                     .otherwise("out_of_stock"))

        # ── feature engineering
        # price tier per kategori (ntile)
        .withColumn("price_tier_in_category", F.ntile(5).over(
            Window.partitionBy("category").orderBy("final_price_idr")
        ))  # 1=cheapest … 5=most expensive

        # apakah ada diskon besar
        .withColumn("is_big_discount", (F.col("discount_pct") >= 30).cast(IntegerType()))

        # popularity score: kombinasi sales, rating, review
        .withColumn("popularity_score", F.round(
            (F.log1p(F.col("sales_last_30d").cast(DoubleType())) * 0.5
             + F.col("avg_rating") / 5 * 0.3
             + F.log1p(F.col("review_count").cast(DoubleType())) * 0.2),
            4
        ))

        # produk baru (listed < 90 hari)
        .withColumn("is_new_arrival", (F.col("days_since_listed") < 90).cast(IntegerType()))

        # profitability proxy
        .withColumn("margin_proxy", F.round(
            F.col("final_price_idr") * (1 - F.col("discount_pct") / 100),
            2
        ))

        .withColumn("processed_at", F.current_timestamp())
        .withColumn("etl_version",  F.lit("v2.0"))
    )

    write_parquet(df, f"{S3_OUT}/product_catalog/", partition_by=["category"])
    return df


# ==============================================================================
# 3. USER INTERACTIONS
# ==============================================================================
def process_user_interactions():
    df = read_catalog("user_interactions")
    df = basic_clean(df,
                     pk_cols=["interaction_id"],
                     not_null_cols=["user_id", "product_id", "interaction_type"])

    # Map implicit score jika belum ada
    score_map = F.create_map(
        F.lit("view"),         F.lit(1),
        F.lit("search"),       F.lit(1),
        F.lit("like"),         F.lit(3),
        F.lit("add_to_cart"),  F.lit(4),
        F.lit("checkout"),     F.lit(5),
        F.lit("purchase"),     F.lit(7),
        F.lit("share"),        F.lit(3),
        F.lit("review"),       F.lit(5),
    )

    df = (
        df
        .withColumn("timestamp", F.to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss"))
        .withColumn("implicit_score", F.coalesce(
            F.col("implicit_score"),
            score_map[F.col("interaction_type")]
        ).cast(IntegerType()))

        # ── temporal features
        .withColumn("hour_of_day",  F.hour("timestamp"))
        .withColumn("day_of_week",  F.dayofweek("timestamp"))     # 1=Sun … 7=Sat
        .withColumn("week_of_year", F.weekofyear("timestamp"))
        .withColumn("month",        F.month("timestamp"))
        .withColumn("is_weekend",   (F.dayofweek("timestamp").isin(1, 7)).cast(IntegerType()))

        # ── session-level aggregasi
        .withColumn("session_interaction_rank", F.row_number().over(
            Window.partitionBy("session_id").orderBy("timestamp")
        ))

        # ── user-level cumulative interactions (RFM input)
        .withColumn("user_interaction_count", F.count("interaction_id").over(
            Window.partitionBy("user_id")
        ))

        # ── interaction funnel stage
        .withColumn("funnel_stage",
            F.when(F.col("interaction_type").isin("view", "search"), "awareness")
             .when(F.col("interaction_type").isin("like", "add_to_cart"), "consideration")
             .when(F.col("interaction_type").isin("checkout", "purchase"), "conversion")
             .otherwise("advocacy")
        )

        .withColumn("processed_at", F.current_timestamp())
        .withColumn("etl_version",  F.lit("v2.0"))
    )

    write_parquet(df, f"{S3_OUT}/user_interactions/",
                  partition_by=["month", "interaction_type"], n_partitions=20)
    return df


# ==============================================================================
# 4. TRANSACTION HISTORY
# ==============================================================================
def process_transaction_history():
    df = read_catalog("transaction_history")
    df = basic_clean(df,
                     pk_cols=["transaction_id"],
                     not_null_cols=["user_id", "product_id", "total_idr"])

    # Filter transaksi tidak valid
    df = df.filter(F.col("total_idr") > 0)

    df = (
        df
        .withColumn("timestamp",        F.to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss"))
        .withColumn("unit_price_idr",   F.col("unit_price_idr").cast(DoubleType()))
        .withColumn("discount_amt_idr", F.col("discount_amt_idr").cast(DoubleType()))
        .withColumn("shipping_fee_idr", F.col("shipping_fee_idr").cast(DoubleType()))
        .withColumn("total_idr",        F.col("total_idr").cast(DoubleType()))
        .withColumn("quantity",         F.col("quantity").cast(IntegerType()))

        # ── temporal features
        .withColumn("hour_of_day",  F.hour("timestamp"))
        .withColumn("day_of_week",  F.dayofweek("timestamp"))
        .withColumn("month",        F.month("timestamp"))
        .withColumn("year",         F.year("timestamp"))
        .withColumn("is_weekend",   (F.dayofweek("timestamp").isin(1, 7)).cast(IntegerType()))

        # ── fraud signals (fitur untuk model fraud)
        # transaksi malam hari (00:00 – 04:59)
        .withColumn("is_odd_hour", (F.hour("timestamp") < 5).cast(IntegerType()))

        # nilai sangat tinggi (> percentile 95 — dihitung via ntile)
        .withColumn("amount_percentile", F.ntile(20).over(Window.orderBy("total_idr")))
        .withColumn("is_high_value",     (F.col("amount_percentile") >= 19).cast(IntegerType()))

        # user melakukan banyak transaksi dalam 1 hari (velocity)
        .withColumn("user_txn_per_day", F.count("transaction_id").over(
            Window.partitionBy("user_id", F.to_date("timestamp").cast("string"))
        ))
        .withColumn("is_high_velocity", (F.col("user_txn_per_day") > 10).cast(IntegerType()))

        # combined fraud score (sederhana)
        .withColumn("fraud_score",
            F.col("is_odd_hour") + F.col("is_high_value") + F.col("is_high_velocity")
        )

        # ── user-level RFM features (window)
        .withColumn("user_total_spend", F.sum("total_idr").over(Window.partitionBy("user_id")))
        .withColumn("user_txn_count",   F.count("transaction_id").over(Window.partitionBy("user_id")))
        .withColumn("user_avg_order",   F.round(
            F.col("user_total_spend") / F.col("user_txn_count"), 2
        ))

        # ── diskon efektif
        .withColumn("effective_discount_pct", F.round(
            F.col("discount_amt_idr") / F.col("unit_price_idr") * 100, 2
        ))

        .withColumn("processed_at", F.current_timestamp())
        .withColumn("etl_version",  F.lit("v2.0"))
    )

    write_parquet(df, f"{S3_OUT}/transaction_history/",
                  partition_by=["year", "month"], n_partitions=20)
    return df


# ==============================================================================
# 5. PRODUCT REVIEWS
# ==============================================================================
def process_product_reviews():
    df = read_catalog("product_reviews")
    df = basic_clean(df,
                     pk_cols=["review_id"],
                     not_null_cols=["user_id", "product_id", "rating", "review_text"])

    df = (
        df
        .withColumn("timestamp", F.to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss"))
        .withColumn("rating",    F.col("rating").cast(IntegerType()))
        # rating harus 1–5
        .filter(F.col("rating").between(1, 5))

        # ── text features (tanpa NLP library eksternal)
        .withColumn("review_text",   F.trim(F.col("review_text")))
        .withColumn("char_count",    F.length("review_text"))
        .withColumn("word_count",    F.size(F.split(F.col("review_text"), "\\s+")))
        .withColumn("has_exclamation", F.col("review_text").contains("!").cast(IntegerType()))
        .withColumn("has_question",    F.col("review_text").contains("?").cast(IntegerType()))

        # ── sentiment label (dari rating)
        .withColumn("sentiment",
            F.when(F.col("rating") >= 4, "positive")
             .when(F.col("rating") <= 2, "negative")
             .otherwise("neutral")
        )

        # ── review kualitas (untuk weight di training)
        .withColumn("review_quality",
            F.when(
                (F.col("verified_purchase") == 1) & (F.col("word_count") >= 10),
                "high"
            ).when(F.col("word_count") >= 5, "medium")
             .otherwise("low")
        )

        # ── temporal
        .withColumn("month",       F.month("timestamp"))
        .withColumn("year",        F.year("timestamp"))

        # ── product-level aggregasi rating
        .withColumn("product_avg_rating", F.round(
            F.avg("rating").over(Window.partitionBy("product_id")), 2
        ))
        .withColumn("product_review_count", F.count("review_id").over(
            Window.partitionBy("product_id")
        ))

        # ── apakah review di atas rata-rata produk
        .withColumn("is_above_avg_rating",
            (F.col("rating") > F.col("product_avg_rating")).cast(IntegerType())
        )

        .withColumn("processed_at", F.current_timestamp())
        .withColumn("etl_version",  F.lit("v2.0"))
    )

    write_parquet(df, f"{S3_OUT}/product_reviews/",
                  partition_by=["year", "sentiment"], n_partitions=15)
    return df


# ==============================================================================
# 6. SEARCH LOGS
# ==============================================================================
def process_search_logs():
    df = read_catalog("search_logs")
    df = basic_clean(df,
                     pk_cols=["log_id"],
                     not_null_cols=["user_id", "query"])

    df = (
        df
        .withColumn("timestamp", F.to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss"))

        # ── query features
        .withColumn("query",        F.lower(F.trim(F.col("query"))))
        .withColumn("query_length", F.length("query"))
        .withColumn("query_word_count", F.size(F.split("query", "\\s+")))

        # ── has filter
        .withColumn("has_category_filter",
            F.col("filter_category").isNotNull().cast(IntegerType()))
        .withColumn("has_price_filter",
            (F.col("filter_min_price").isNotNull() | F.col("filter_max_price").isNotNull())
            .cast(IntegerType()))

        # ── temporal features
        .withColumn("hour_of_day",  F.hour("timestamp"))
        .withColumn("day_of_week",  F.dayofweek("timestamp"))
        .withColumn("month",        F.month("timestamp"))
        .withColumn("year",         F.year("timestamp"))
        .withColumn("is_weekend",   (F.dayofweek("timestamp").isin(1, 7)).cast(IntegerType()))

        # ── zero result flag
        .withColumn("is_zero_result", (F.col("n_results") == 0).cast(IntegerType()))

        # ── CTR label
        .withColumn("is_clicked", F.coalesce(
            F.col("is_clicked"), F.lit(0)
        ).cast(IntegerType()))

        # ── query popularity (berapa kali query muncul)
        .withColumn("query_frequency", F.count("log_id").over(
            Window.partitionBy("query")
        ))

        # ── user search frequency
        .withColumn("user_search_count", F.count("log_id").over(
            Window.partitionBy("user_id")
        ))

        # ── clicked position bucket
        .withColumn("click_position_bucket",
            F.when(F.col("clicked_position") == -1, "no_click")
             .when(F.col("clicked_position") <= 3,  "top3")
             .when(F.col("clicked_position") <= 10, "top10")
             .otherwise("below10")
        )

        .withColumn("processed_at", F.current_timestamp())
        .withColumn("etl_version",  F.lit("v2.0"))
    )

    write_parquet(df, f"{S3_OUT}/search_logs/",
                  partition_by=["year", "month"], n_partitions=20)
    return df


# ==============================================================================
# 7. FEATURE STORE: USER × PRODUCT MATRIX
#    Gabungan lintas dataset untuk training model rekomendasi & ranking
# ==============================================================================
def build_user_product_features(ui_df, th_df):
    """
    Membuat tabel fitur gabungan user–produk dari interaction & transaction.
    Output ini langsung bisa dipakai sebagai input Collaborative Filtering
    atau sebagai fitur untuk model LightGBM/XGBoost ranking.
    """
    logger.info("Building user–product feature matrix...")

    # Aggregasi interaction per user-product
    ui_agg = (
        ui_df.groupBy("user_id", "product_id")
        .agg(
            F.sum("implicit_score").alias("total_implicit_score"),
            F.count("interaction_id").alias("n_interactions"),
            F.countDistinct("interaction_type").alias("n_interaction_types"),
            F.max("timestamp").alias("last_interaction_ts"),
            F.sum(F.when(F.col("interaction_type") == "purchase", 1).otherwise(0))
              .alias("n_purchases"),
            F.sum(F.when(F.col("interaction_type") == "add_to_cart", 1).otherwise(0))
              .alias("n_add_to_cart"),
        )
    )

    # Aggregasi transaksi per user-product
    th_agg = (
        th_df.filter(F.col("payment_status") == "paid")
        .groupBy("user_id", "product_id")
        .agg(
            F.sum("total_idr").alias("total_spend_on_product"),
            F.sum("quantity").alias("total_qty_bought"),
            F.count("transaction_id").alias("n_transactions"),
            F.avg("total_idr").alias("avg_order_value"),
            F.max("timestamp").alias("last_purchase_ts"),
        )
    )

    # Gabung
    feat = (
        ui_agg
        .join(th_agg, on=["user_id", "product_id"], how="left")
        .fillna(0)
        .withColumn("has_purchased",   (F.col("n_purchases") > 0).cast(IntegerType()))
        .withColumn("days_since_last", F.datediff(F.current_timestamp(), "last_interaction_ts"))
        .withColumn("processed_at",    F.current_timestamp())
    )

    write_parquet(feat, f"{S3_OUT}/features/user_product_matrix/", n_partitions=30)
    logger.info("  ✅ User–product feature matrix selesai.")
    return feat


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    logger.info("=" * 70)
    logger.info("TechnoMart — AWS Glue ETL Pipeline v2.0")
    logger.info("=" * 70)

    up_df = process_user_profiles()
    pc_df = process_product_catalog()
    ui_df = process_user_interactions()
    th_df = process_transaction_history()
    _     = process_product_reviews()
    _     = process_search_logs()

    build_user_product_features(ui_df, th_df)

    logger.info("=" * 70)
    logger.info("✅ Semua dataset berhasil diproses dan disimpan ke S3.")
    logger.info(f"   Output: {S3_OUT}/")
    logger.info("=" * 70)

    job.commit()


if __name__ == "__main__":
    main()
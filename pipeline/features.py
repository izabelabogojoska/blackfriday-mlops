import hashlib
import pandas as pd

def hash_value(value, buckets=1000):
    return int(hashlib.md5(value.encode()).hexdigest(), 16) % buckets

def build_features(df):
    df["product_hash"] = df["product_id"].apply(hash_value)
    df["seller_hash"] = df["seller_id"].apply(hash_value)
    df["category_hash"] = df["category"].apply(hash_value)

    df["seller_category_cross"] = (
        df["seller_id"] + "_" + df["category"]
    ).apply(hash_value)

    return df[
        [
            "product_hash",
            "seller_hash",
            "category_hash",
            "seller_category_cross",
            "price_change_percent",
            "price_change_count",
        ]
    ]

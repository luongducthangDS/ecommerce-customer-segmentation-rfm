"""
E-commerce Customer Segmentation & Retention Analysis
Dataset: Olist Brazilian E-Commerce (99,441 orders, 2016-09 to 2018-10)

Key gotcha handled: `customer_id` in Olist is unique PER ORDER (a surrogate key),
not per real customer. Repeat-purchase / RFM analysis must use `customer_unique_id`
from olist_customers_dataset.csv instead, or every customer looks like a one-time buyer.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, json

DATA = "data"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ---------- Load ----------
orders = pd.read_csv(f"{DATA}/olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
customers = pd.read_csv(f"{DATA}/olist_customers_dataset.csv")
payments = pd.read_csv(f"{DATA}/olist_order_payments_dataset.csv")

orders = orders[orders["order_status"] == "delivered"].copy()

order_value = payments.groupby("order_id", as_index=False)["payment_value"].sum()

df = (
    orders[["order_id", "customer_id", "order_purchase_timestamp"]]
    .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left")
    .merge(order_value, on="order_id", how="left")
)
df["payment_value"] = df["payment_value"].fillna(0)

print(f"Delivered orders: {len(df):,}")
print(f"Unique customers (customer_unique_id): {df['customer_unique_id'].nunique():,}")

# ---------- RFM ----------
ref_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

rfm = df.groupby("customer_unique_id").agg(
    recency=("order_purchase_timestamp", lambda x: (ref_date - x.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("payment_value", "sum"),
).reset_index()

repeat_rate = (rfm["frequency"] > 1).mean()
print(f"Repeat purchase rate: {repeat_rate:.2%}")

rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["m_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

def segment(row):
    if row["frequency"] > 1 and row["r_score"] >= 4:
        return "Loyal / Repeat"
    if row["rfm_score"] >= 12:
        return "Champions"
    if row["r_score"] >= 4 and row["frequency"] == 1:
        return "New (single order, recent)"
    if row["r_score"] <= 2 and row["m_score"] >= 4:
        return "At Risk (high value, lapsed)"
    if row["r_score"] <= 2:
        return "Lost / Lapsed"
    return "Others"

rfm["segment"] = rfm.apply(segment, axis=1)
rfm.to_csv(f"{OUT}/customer_rfm.csv", index=False)

seg_summary = rfm.groupby("segment").agg(
    customers=("customer_unique_id", "count"),
    avg_monetary=("monetary", "mean"),
    total_monetary=("monetary", "sum"),
).sort_values("customers", ascending=False)
seg_summary["pct_customers"] = seg_summary["customers"] / seg_summary["customers"].sum()
seg_summary["pct_revenue"] = seg_summary["total_monetary"] / seg_summary["total_monetary"].sum()
seg_summary.to_csv(f"{OUT}/segment_summary.csv")
print("\n=== Segment summary ===")
print(seg_summary.round(2))

fig, ax = plt.subplots(figsize=(8, 5))
seg_summary["customers"].sort_values().plot(kind="barh", ax=ax, color="#2E86AB")
ax.set_xlabel("Number of customers")
ax.set_title("Customer segments (RFM) - Olist")
plt.tight_layout()
plt.savefig(f"{OUT}/segment_distribution.png", dpi=150)
plt.close()

# ---------- Cohort retention ----------
first_purchase = df.groupby("customer_unique_id")["order_purchase_timestamp"].min().rename("cohort_date")
df2 = df.merge(first_purchase, on="customer_unique_id")
df2["order_month"] = df2["order_purchase_timestamp"].dt.to_period("M")
df2["cohort_month"] = df2["cohort_date"].dt.to_period("M")
df2["cohort_index"] = (df2["order_month"] - df2["cohort_month"]).apply(lambda x: x.n)

cohort_counts = df2.groupby(["cohort_month", "cohort_index"])["customer_unique_id"].nunique().reset_index()
cohort_pivot = cohort_counts.pivot(index="cohort_month", columns="cohort_index", values="customer_unique_id")
cohort_size = cohort_pivot.iloc[:, 0]
retention = cohort_pivot.divide(cohort_size, axis=0)
retention.to_csv(f"{OUT}/cohort_retention_matrix.csv")

if 1 in retention.columns:
    month1_retention = retention[1].dropna().mean()
    print(f"\nAvg month-1 retention across cohorts: {month1_retention:.2%}")
else:
    month1_retention = None

cols = [c for c in retention.columns if c <= 6]
fig, ax = plt.subplots(figsize=(10, 8))
vmax = retention[cols].iloc[:, 1:].max().max() if len(cols) > 1 else 1
im = ax.imshow(retention[cols].values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=vmax)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols)
ax.set_yticks(range(len(retention)))
ax.set_yticklabels([str(m) for m in retention.index])
ax.set_xlabel("Months since first purchase")
ax.set_ylabel("Acquisition cohort")
ax.set_title("Cohort retention rate - Olist")
plt.colorbar(im, ax=ax, label="Retention rate")
plt.tight_layout()
plt.savefig(f"{OUT}/cohort_retention_heatmap.png", dpi=150)
plt.close()

stats = {
    "total_delivered_orders": len(df),
    "unique_customers": int(df["customer_unique_id"].nunique()),
    "repeat_purchase_rate": round(repeat_rate, 4),
    "avg_order_value": round(df["payment_value"].mean(), 2),
    "total_revenue": round(df["payment_value"].sum(), 2),
    "month1_retention_avg": round(month1_retention, 4) if month1_retention else None,
    "date_range": [str(df["order_purchase_timestamp"].min().date()), str(df["order_purchase_timestamp"].max().date())],
}
with open(f"{OUT}/headline_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("\n=== Headline stats ===")
print(json.dumps(stats, indent=2))

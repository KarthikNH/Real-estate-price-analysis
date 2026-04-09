"""
Real Estate Multi-Agent Intelligence Pipeline
Bangalore Property Market Analysis (2021-2025)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import warnings
import io
import sys
import os

warnings.filterwarnings("ignore")


class DataIngestionAgent:
    REQUIRED_COLUMNS = [
        "location", "latitude", "longitude", "zone", "property_type",
        "bhk", "total_sqft", "price_per_sqft",
        "price_2021", "price_2022", "price_2023", "price_2024", "price_2025",
        "annual_growth_rate_pct", "metro_access",
        "distance_to_it_hub_km", "distance_to_airport_km"
    ]

    def __init__(self, filepath=None):
        self.filepath = filepath
        self.raw_df = None

    def load(self):
        if self.filepath and os.path.exists(self.filepath):
            print(f"[DataIngestionAgent] ✓ Loading from file: {self.filepath}")
            df = pd.read_csv(self.filepath)
            df.columns = df.columns.str.strip()
            self.raw_df = df
        else:
            print("[DataIngestionAgent] ⚠ File not found. Generating synthetic data.")
            self.raw_df = self._generate_synthetic()
        self._validate()
        print(f"[DataIngestionAgent] ✓ Loaded {len(self.raw_df):,} records successfully.")
        return self.raw_df

    def _validate(self):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.raw_df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def _generate_synthetic(self):
        np.random.seed(42)
        n = 500
        locations = {
            "Koramangala": (12.935, 77.621, "South"),
            "Whitefield": (12.971, 77.749, "East"),
            "Indiranagar": (12.978, 77.643, "East"),
            "HSR Layout": (12.912, 77.640, "South"),
            "Electronic City": (12.845, 77.657, "South"),
            "Hebbal": (13.035, 77.597, "North"),
            "Manyata Tech Park": (13.049, 77.619, "North"),
            "MG Road": (12.975, 77.601, "Central"),
            "Malleshwaram": (13.003, 77.570, "North"),
        }
        rows = []
        for _ in range(n):
            loc = np.random.choice(list(locations.keys()))
            lat, lon, zone = locations[loc]
            base_price = np.random.randint(5000, 22000)
            growth = np.random.uniform(3, 22)
            price_2021 = base_price * 0.62
            prices = [price_2021 * ((1 + growth / 100) ** i) for i in range(5)]
            sqft = np.random.randint(400, 3000)
            rows.append({
                "location": loc, "latitude": lat + np.random.uniform(-0.01, 0.01),
                "longitude": lon + np.random.uniform(-0.01, 0.01),
                "zone": zone, "property_type": np.random.choice(["Apartment", "Villa", "Builder Floor"]),
                "bhk": np.random.randint(1, 5), "total_sqft": sqft,
                "price_per_sqft": base_price,
                "price_2021": round(prices[0]), "price_2022": round(prices[1]),
                "price_2023": round(prices[2]), "price_2024": round(prices[3]),
                "price_2025": round(prices[4]),
                "annual_growth_rate_pct": round(growth, 2),
                "metro_access": np.random.choice(["Yes", "No"]),
                "distance_to_it_hub_km": round(np.random.uniform(0.5, 20), 1),
                "distance_to_airport_km": round(np.random.uniform(8, 55), 1),
            })
        return pd.DataFrame(rows)


class DataCleaningAgent:
    def __init__(self):
        self.stats = {}

    def clean(self, df):
        print("[DataCleaningAgent] ✓ Starting data cleaning pipeline...")
        df = df.copy()
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        numeric_cols = [
            "latitude", "longitude", "total_sqft", "price_per_sqft",
            "price_2021", "price_2022", "price_2023", "price_2024", "price_2025",
            "annual_growth_rate_pct", "bhk", "distance_to_it_hub_km", "distance_to_airport_km"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        for col in ["location", "zone", "property_type", "metro_access"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
        df["metro_access"] = df["metro_access"].map(
            lambda x: 1 if str(x).lower() in ["yes", "1", "true"] else 0
        )
        q1 = df["price_per_sqft"].quantile(0.01)
        q3 = df["price_per_sqft"].quantile(0.99)
        df = df[(df["price_per_sqft"] >= q1) & (df["price_per_sqft"] <= q3)]
        df = df[(df["latitude"].between(12.5, 13.5)) & (df["longitude"].between(77.3, 78.0))]
        self.stats = {"initial_rows": initial_rows, "final_rows": len(df), "dropped": initial_rows - len(df)}
        print(f"[DataCleaningAgent] ✓ Removed {self.stats['dropped']} outlier rows. Final dataset: {self.stats['final_rows']:,} records.")
        return df.reset_index(drop=True)


class EDAAgent:
    def __init__(self):
        self.summary = {}

    def analyze(self, df):
        print("[EDAAgent] ✓ Running exploratory data analysis...")
        self.summary = {
            "total_records": len(df),
            "unique_locations": df["location"].nunique(),
            "zones": df["zone"].value_counts().to_dict(),
            "property_types": df["property_type"].value_counts().to_dict(),
            "avg_price_per_sqft": round(df["price_per_sqft"].mean(), 0),
            "median_price_per_sqft": round(df["price_per_sqft"].median(), 0),
            "avg_growth_rate": round(df["annual_growth_rate_pct"].mean(), 2),
            "max_growth_location": df.groupby("location")["annual_growth_rate_pct"].mean().idxmax(),
            "min_growth_location": df.groupby("location")["annual_growth_rate_pct"].mean().idxmin(),
            "price_range": {"min": int(df["price_per_sqft"].min()), "max": int(df["price_per_sqft"].max())},
            "metro_access_pct": round(df["metro_access"].mean() * 100, 1),
        }
        years = [2021, 2022, 2023, 2024, 2025]
        price_cols = [f"price_{y}" for y in years]
        yoy = {}
        for i in range(1, len(years)):
            valid = df[[price_cols[i - 1], price_cols[i]]].dropna()
            valid = valid[(valid[price_cols[i - 1]] > 0)]
            growth = ((valid[price_cols[i]] - valid[price_cols[i - 1]]) / valid[price_cols[i - 1]] * 100).mean()
            yoy[f"{years[i-1]}-{years[i]}"] = round(growth, 2)
        self.summary["yoy_growth"] = yoy
        print(f"[EDAAgent] ✓ Analyzed {self.summary['unique_locations']} unique micro-markets across {len(self.summary['zones'])} zones.")
        return self.summary


class FeatureEngineeringAgent:
    def __init__(self):
        self.le_zone = LabelEncoder()
        self.le_type = LabelEncoder()
        self.le_loc = LabelEncoder()

    def engineer(self, df):
        print("[FeatureEngineeringAgent] ✓ Engineering price momentum and connectivity features...")
        df = df.copy()
        df["price_growth_21_25"] = ((df["price_2025"] - df["price_2021"]) / df["price_2021"]) * 100
        df["price_growth_23_25"] = ((df["price_2025"] - df["price_2023"]) / df["price_2023"]) * 100
        df["price_cagr"] = (((df["price_2025"] / df["price_2021"]) ** (1 / 4)) - 1) * 100
        df["total_value_lakhs"] = (df["price_per_sqft"] * df["total_sqft"]) / 100000
        df["value_per_bhk"] = df["total_value_lakhs"] / df["bhk"].clip(lower=1)
        df["sqft_per_bhk"] = df["total_sqft"] / df["bhk"].clip(lower=1)
        df["connectivity_score"] = (
            df["metro_access"] * 25
            + np.clip(20 - df["distance_to_it_hub_km"], 0, 20)
            + np.clip(30 - df["distance_to_airport_km"] * 0.5, 0, 30)
        )
        for col in ["price_per_sqft", "annual_growth_rate_pct", "total_value_lakhs", "connectivity_score", "distance_to_it_hub_km"]:
            mn, mx = df[col].min(), df[col].max()
            rng = mx - mn if mx != mn else 1
            df[f"{col}_norm"] = (df[col] - mn) / rng
        df["zone_enc"] = self.le_zone.fit_transform(df["zone"])
        df["type_enc"] = self.le_type.fit_transform(df["property_type"])
        df["loc_enc"] = self.le_loc.fit_transform(df["location"])
        print(f"[FeatureEngineeringAgent] ✓ Generated {len(df.columns)} total features (including 12 engineered).")
        return df


class PredictionAgent:
    FEATURE_COLS = [
        "price_per_sqft", "total_sqft", "bhk", "annual_growth_rate_pct",
        "metro_access", "distance_to_it_hub_km", "distance_to_airport_km",
        "connectivity_score", "price_cagr", "price_growth_21_25",
        "zone_enc", "type_enc", "loc_enc"
    ]

    def __init__(self):
        self.model_1yr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42)
        self.model_3yr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42)
        self.model_5yr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42)
        self.metrics = {}
        self.scaler = StandardScaler()

    def train_and_predict(self, df):
        print("[PredictionAgent] ✓ Training Gradient Boosting models for 1yr, 3yr, 5yr horizons...")
        df = df.copy()
        available_features = [c for c in self.FEATURE_COLS if c in df.columns]
        X = df[available_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        cagr = df["price_cagr"].clip(0, 50) / 100
        y_1yr = df["price_2025"] * (1 + cagr)
        y_3yr = df["price_2025"] * (1 + cagr) ** 3
        y_5yr = df["price_2025"] * (1 + cagr) ** 5
        X_train, X_test, y1_train, y1_test, y3_train, y3_test, y5_train, y5_test = train_test_split(
            X_scaled, y_1yr, y_3yr, y_5yr, test_size=0.2, random_state=42
        )
        self.model_1yr.fit(X_train, y1_train)
        self.model_3yr.fit(X_train, y3_train)
        self.model_5yr.fit(X_train, y5_train)
        self.metrics = {
            "mape_1yr": round(mean_absolute_percentage_error(y1_test, self.model_1yr.predict(X_test)) * 100, 2),
            "mape_3yr": round(mean_absolute_percentage_error(y3_test, self.model_3yr.predict(X_test)) * 100, 2),
            "mape_5yr": round(mean_absolute_percentage_error(y5_test, self.model_5yr.predict(X_test)) * 100, 2),
        }
        print(f"[PredictionAgent] ✓ Model trained | MAPE: 1yr={self.metrics['mape_1yr']}% | 3yr={self.metrics['mape_3yr']}% | 5yr={self.metrics['mape_5yr']}%")
        df["predicted_price_1yr"] = self.model_1yr.predict(X_scaled).round(0)
        df["predicted_price_3yr"] = self.model_3yr.predict(X_scaled).round(0)
        df["predicted_price_5yr"] = self.model_5yr.predict(X_scaled).round(0)
        return df

    def get_metrics(self):
        return self.metrics


class InvestmentAnalysisAgent:
    CATEGORY_RULES = {
        "Premium":        lambda row: row["price_per_sqft"] >= 16000,
        "High Growth":    lambda row: row["annual_growth_rate_pct"] >= 15,
        "Stable":         lambda row: 5 <= row["annual_growth_rate_pct"] < 10,
        "Budget":         lambda row: row["price_per_sqft"] < 7000,
        "High Potential": lambda row: (row["annual_growth_rate_pct"] >= 10 and row["price_per_sqft"] < 12000 and row["connectivity_score"] > 30),
    }

    def analyze(self, df):
        print("[InvestmentAnalysisAgent] ✓ Computing ROI projections, growth indices, and investment scores...")
        df = df.copy()
        df["roi_1yr"] = ((df["predicted_price_1yr"] - df["price_2025"]) / df["price_2025"] * 100).round(2)
        df["roi_3yr"] = ((df["predicted_price_3yr"] - df["price_2025"]) / df["price_2025"] * 100).round(2)
        df["roi_5yr"] = ((df["predicted_price_5yr"] - df["price_2025"]) / df["price_2025"] * 100).round(2)
        df["growth_rate_index"] = (df["annual_growth_rate_pct"] * 0.40 + df["price_cagr"] * 0.35 + df["price_growth_23_25"] * 0.25).round(2)

        def norm(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx != mn else pd.Series([0.5] * len(s), index=s.index)

        df["investment_score"] = (norm(df["annual_growth_rate_pct"]) * 35 + norm(df["connectivity_score"]) * 25 + norm(df["roi_5yr"]) * 25 + (1 - norm(df["price_per_sqft"])) * 15).round(1)
        df["category"] = "Stable"
        for cat in ["Premium", "High Growth", "Budget", "High Potential", "Stable"]:
            mask = df.apply(self.CATEGORY_RULES[cat], axis=1)
            df.loc[mask, "category"] = cat
        df["location_rank"] = df.groupby("location")["investment_score"].rank(ascending=False, method="min").astype(int)
        print("[InvestmentAnalysisAgent] ✓ Classified all properties into 5 investment categories.")
        return df


class InsightGenerationAgent:
    TEMPLATES = {
        "Premium": "{loc} commands premium pricing at Rs {pps:,}/sqft with a {growth:.1f}% annual growth trajectory. 5-year ROI forecast: {roi5:.1f}%. Best suited for high-net-worth portfolio diversification.",
        "High Growth": "{loc} is among the fastest appreciating micro-markets with {growth:.1f}% YoY growth. Investment score: {score:.0f}/100. Projected 3-year ROI: {roi3:.1f}%. Ideal for growth-oriented investors with a medium horizon.",
        "High Potential": "{loc} presents an emerging opportunity — current price Rs {pps:,}/sqft is still accessible, while connectivity score of {conn:.0f} and {growth:.1f}% growth signal strong upside. 5-year ROI estimate: {roi5:.1f}%. Early-entry advantage recommended.",
        "Budget": "{loc} remains affordable at Rs {pps:,}/sqft, attracting first-time buyers and rental investors. Conservative growth at {growth:.1f}%/yr. Suitable for rental yield plays rather than capital appreciation.",
        "Stable": "{loc} offers steady {growth:.1f}% annual appreciation. Investment score: {score:.0f}/100. Low volatility makes it suitable for risk-averse investors. 5-year ROI estimate: {roi5:.1f}%.",
    }

    def generate(self, df):
        print("[InsightGenerationAgent] ✓ Generating NLP-style market insights for all locations...")
        df = df.copy()
        loc_summary = df.groupby("location").agg(
            avg_price_per_sqft=("price_per_sqft", "mean"),
            avg_growth=("annual_growth_rate_pct", "mean"),
            avg_score=("investment_score", "mean"),
            avg_roi_3yr=("roi_3yr", "mean"),
            avg_roi_5yr=("roi_5yr", "mean"),
            avg_conn=("connectivity_score", "mean"),
            dominant_category=("category", lambda x: x.mode()[0]),
        ).reset_index()
        insight_map = {}
        for _, row in loc_summary.iterrows():
            tpl = self.TEMPLATES.get(row["dominant_category"], self.TEMPLATES["Stable"])
            insight_map[row["location"]] = tpl.format(
                loc=row["location"], pps=int(row["avg_price_per_sqft"]),
                growth=row["avg_growth"], score=row["avg_score"],
                roi3=row["avg_roi_3yr"], roi5=row["avg_roi_5yr"], conn=row["avg_conn"],
            )
        df["location_insight"] = df["location"].map(insight_map)
        best = loc_summary.sort_values("avg_score", ascending=False).iloc[0]
        df["best_city_flag"] = (df["location"] == best["location"]).astype(int)
        df.attrs["best_city"] = best["location"]
        df.attrs["best_city_score"] = round(best["avg_score"], 1)
        print(f"[InsightGenerationAgent] ✓ Generated insights for {len(loc_summary)} locations. Top city: {best['location']} (score: {round(best['avg_score'],1)})")
        return df


class VisualizationAgent:
    CATEGORY_COLORS = {
        "Premium": "#C9A96E", "High Growth": "#3ECFB2",
        "High Potential": "#6E9EFF", "Stable": "#A0AEC0", "Budget": "#F6A35A",
    }

    def prepare(self, df):
        print("[VisualizationAgent] ✓ Aggregating artifacts for dashboard rendering...")
        loc_agg = df.groupby("location").agg(
            lat=("latitude", "mean"), lon=("longitude", "mean"),
            zone=("zone", lambda x: x.mode()[0]),
            avg_price_per_sqft=("price_per_sqft", "mean"),
            avg_growth=("annual_growth_rate_pct", "mean"),
            avg_investment_score=("investment_score", "mean"),
            avg_roi_1yr=("roi_1yr", "mean"), avg_roi_3yr=("roi_3yr", "mean"),
            avg_roi_5yr=("roi_5yr", "mean"), avg_conn=("connectivity_score", "mean"),
            dominant_category=("category", lambda x: x.mode()[0]),
            insight=("location_insight", "first"), count=("location", "count"),
            avg_price_2021=("price_2021", "mean"), avg_price_2022=("price_2022", "mean"),
            avg_price_2023=("price_2023", "mean"), avg_price_2024=("price_2024", "mean"),
            avg_price_2025=("price_2025", "mean"),
            pred_1yr=("predicted_price_1yr", "mean"),
            pred_3yr=("predicted_price_3yr", "mean"),
            pred_5yr=("predicted_price_5yr", "mean"),
        ).reset_index()
        loc_agg["color"] = loc_agg["dominant_category"].map(self.CATEGORY_COLORS)
        loc_agg = loc_agg.round(2)
        zone_agg = df.groupby("zone").agg(
            avg_price=("price_per_sqft", "mean"), avg_growth=("annual_growth_rate_pct", "mean"),
            avg_score=("investment_score", "mean"), count=("zone", "count"),
        ).reset_index().round(2)
        cat_dist = df["category"].value_counts().reset_index()
        cat_dist.columns = ["category", "count"]
        cat_dist["color"] = cat_dist["category"].map(self.CATEGORY_COLORS)
        price_trend = df.groupby("location").agg(
            price_2021=("price_2021", "mean"), price_2022=("price_2022", "mean"),
            price_2023=("price_2023", "mean"), price_2024=("price_2024", "mean"),
            price_2025=("price_2025", "mean"),
        ).reset_index()
        artifacts = {
            "full_df": df, "loc_agg": loc_agg, "zone_agg": zone_agg,
            "cat_dist": cat_dist,
            "top_investments": loc_agg.sort_values("avg_investment_score", ascending=False).head(20),
            "price_trend": price_trend,
            "category_colors": self.CATEGORY_COLORS,
        }
        print(f"[VisualizationAgent] ✓ Dashboard artifacts ready: {len(loc_agg)} locations, {len(zone_agg)} zones.")
        return artifacts


class RealEstatePipeline:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.artifacts = {}
        self.eda_summary = {}
        self.model_metrics = {}
        self.best_city = None

    def run(self):
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            print("=" * 58)
            print("  AREIS PIPELINE  |  BANGALORE 2021-2025")
            print("=" * 58)

            agent1 = DataIngestionAgent(self.filepath)
            df = agent1.load()

            agent2 = DataCleaningAgent()
            df = agent2.clean(df)

            agent3 = EDAAgent()
            self.eda_summary = agent3.analyze(df)

            agent4 = FeatureEngineeringAgent()
            df = agent4.engineer(df)

            agent5 = PredictionAgent()
            df = agent5.train_and_predict(df)
            self.model_metrics = agent5.get_metrics()

            agent6 = InvestmentAnalysisAgent()
            df = agent6.analyze(df)

            agent7 = InsightGenerationAgent()
            df = agent7.generate(df)

            agent8 = VisualizationAgent()
            self.artifacts = agent8.prepare(df)

            self.best_city = df.attrs.get("best_city", "N/A")

            out_path = "processed_data.csv"
            df.to_csv(out_path, index=False)
            print(f"[Pipeline] ✓ Processed dataset exported → {out_path}")
            print(f"[Pipeline] ✓ Best investment city: {self.best_city}")
            print("=" * 58)

        finally:
            sys.stdout = old_stdout

        return {
            "artifacts": self.artifacts,
            "eda_summary": self.eda_summary,
            "model_metrics": self.model_metrics,
            "best_city": self.best_city,
            "execution_logs": captured.getvalue(),
        }


if __name__ == "__main__":
    result = RealEstatePipeline(filepath="BHP_3.csv").run()
    eda = result["eda_summary"]
    print(result["execution_logs"])
    print(f"Total Records : {eda.get('total_records', 0):,}")
    print(f"Best City     : {result['best_city']}")

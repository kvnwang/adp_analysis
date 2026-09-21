from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"date", "NER_SA", "agg_RIS", "category", "timestep"}
MONTHLY_TIMESTEP = "M"
WEEKLY_TIMESTEP = "W"

def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["NER_SA"] = pd.to_numeric(df["NER_SA"], errors="coerce")
    df = df.dropna(subset=["date", "NER_SA"])
    return df.sort_values("date").reset_index(drop=True)

def get_series(
    df: pd.DataFrame,
    agg_RIS: str = "National",
    category: str = "U.S.",
    timestep: str = MONTHLY_TIMESTEP,
) -> pd.Series:
    segment = df[
        (df["agg_RIS"] == agg_RIS)
        & (df["category"] == category)
        & (df["timestep"] == timestep)
    ]
    return segment.drop_duplicates("date").set_index("date")["NER_SA"].sort_index()

def list_categories(df: pd.DataFrame, agg_RIS: str) -> list[str]:
    return sorted(df.loc[df["agg_RIS"] == agg_RIS, "category"].unique().tolist())

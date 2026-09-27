"""Create clean Andhra Pradesh Kharif-rice yield labels from the raw OGD CSV.

Run from the workspace root:
    python sic/clean_yield_area.py

The default years (2017-18 through 2021-22) use the same, pre-reorganisation
13-district geography.  Do not mix them with 2022-23's new 26-district rows.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "raw" / "yeild_area" / "crop-wise-area-production-yield.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "labels_ap_kharif_rice_2017_2021.csv"
START_YEAR = 2017
END_YEAR = 2021

# Names used in older crop records -> stable names to use in every later dataset.
DISTRICT_NAMES = {
    "Ananthapuramu": "Anantapur",
    "Sri Potti Sriramulu Nellore": "Nellore",
    "Y.S.R. Kadapa": "Kadapa",
}


def season_start_year(value: str) -> int:
    """Convert a value such as '2016-2017' to 2016."""
    return int(str(value).split("-")[0])


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)
    required = {
        "year", "state_name", "district_name", "crop_name", "season",
        "area", "area_unit", "production", "production_unit", "yield", "yield_unit",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"The raw CSV is missing columns: {', '.join(missing)}")

    selected = df.loc[
        (df["state_name"].str.strip().eq("Andhra Pradesh"))
        & (df["season"].str.strip().eq("Kharif"))
        & (df["crop_name"].str.strip().eq("Rice"))
        & (df["area_unit"].str.strip().eq("Hectare"))
        & (df["production_unit"].str.strip().eq("Tonnes"))
        & (df["yield_unit"].str.strip().eq("Tonnes/Hectare"))
    ].copy()

    selected["Year"] = selected["year"].map(season_start_year)
    selected = selected.loc[selected["Year"].between(START_YEAR, END_YEAR)].copy()
    selected["District"] = selected["district_name"].replace(DISTRICT_NAMES)
    selected["SownArea"] = pd.to_numeric(selected["area"], errors="coerce")
    selected["Production_Tonnes"] = pd.to_numeric(selected["production"], errors="coerce")
    selected = selected.loc[(selected["SownArea"] > 0) & (selected["Production_Tonnes"] >= 0)].copy()

    # Calculate the target ourselves from raw area and production, rather than
    # trusting a precomputed field. This is part of the project's own pipeline.
    selected["Yield_Tonne_per_Hectare"] = selected["Production_Tonnes"] / selected["SownArea"]
    selected["Source_Yield_Tonne_per_Hectare"] = pd.to_numeric(selected["yield"], errors="coerce")
    selected["Yield_Check_Difference"] = (
        selected["Yield_Tonne_per_Hectare"] - selected["Source_Yield_Tonne_per_Hectare"]
    ).abs()

    clean = selected[[
        "District", "Year", "SownArea", "Production_Tonnes",
        "Yield_Tonne_per_Hectare", "Source_Yield_Tonne_per_Hectare", "Yield_Check_Difference",
    ]].copy()
    clean.insert(2, "Season", "Kharif")
    clean.insert(3, "Crop", "Rice")
    clean = clean.sort_values(["Year", "District"]).reset_index(drop=True)

    duplicates = clean.duplicated(["District", "Year"])
    if duplicates.any():
        duplicate_rows = clean.loc[duplicates, ["District", "Year"]].to_dict("records")
        raise ValueError(f"Duplicate district-year labels found: {duplicate_rows}")
    if clean.empty:
        raise ValueError("No rows matched. Check the source data or filtering settings.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")

    print(f"Created: {OUTPUT_CSV}")
    print(f"Rows: {len(clean)} | districts: {clean['District'].nunique()} | years: {clean['Year'].min()}-{clean['Year'].max()}")
    print(f"Largest calculated-vs-source yield difference: {clean['Yield_Check_Difference'].max():.6f} t/ha")
    print("\nFirst five rows:")
    print(clean.head().to_string(index=False))


if __name__ == "__main__":
    main()

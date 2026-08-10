# PremierLeagueML

Machine learning project built on Premier League team-level statistics from
[FBref](https://fbref.com/), covering the 2021-22 through 2025-26 seasons.

The current focus is the data layer: loading the raw FBref tables, normalising
their column structure, and merging them into a single master team-season
dataset that models can be trained on.

## The data quirk

FBref tables saved from the browser carry a `.xls` extension but are actually
**HTML documents**, not real Excel workbooks. `pd.read_excel()` fails on them.
They must be read with:

```python
tables = pd.read_html(file_path)
df = tables[0]
```

Their headers are also two-level `MultiIndex` columns (e.g. `('Performance', 'Gls')`)
with `Unnamed: N_level_0` filler on the identifier columns. The notebooks flatten
these into flat names like `Performance_Gls` before merging.

## Layout

```
data/
  raw/                  # FBref exports, one folder per season (tracked)
    2022 PL Season/
    2023 PL Season/
    2024 PL Season/
    2025 PL Season/
    2026 PL Season/
  processed/            # intermediate cleaned tables (generated, ignored)
  final/                # model-ready datasets (generated, ignored)
notebooks/
  01_Data_Loading.ipynb    # first pass: read one table, explore, plot
  02_Load_All_Files.ipynb  # load every season, flatten columns, merge to master
src/                    # reusable modules (to be extracted from notebooks)
models/                 # trained model artefacts (generated, ignored)
outputs/                # figures and reports (generated, ignored)
```

Each season folder holds the same family of tables — Overall standings,
Home/Away splits, Standard Stats, Shooting, Goalkeeping, Playing Time and
Miscellaneous Stats — each in a squad version and an opponent version.

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install pandas lxml html5lib matplotlib jupyter openpyxl
jupyter notebook
```

`lxml` / `html5lib` are what back `pd.read_html`, so they are required rather
than optional.

## Status

- [x] Load a single FBref table and confirm the HTML-not-Excel workaround
- [x] Load every `.xls` in a season folder into a dict of DataFrames
- [x] Flatten `MultiIndex` headers into flat, merge-safe column names
- [x] Merge all squad tables for a season into one master DataFrame on `Squad`
- [x] Verify team coverage is consistent across tables within each season
- [ ] Normalise opponent tables (`vs Arsenal` → `Arsenal`) and join them
- [ ] Stack all five seasons into a single team-season dataset
- [ ] Extract the notebook helpers into `src/`
- [ ] Feature engineering and first model

## Data source

All statistics are from FBref (Sports Reference). Redistributed here for
personal, non-commercial analysis.

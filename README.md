# SF-GNN

- Core logic lives in `src/` (datasets, model, training, logging).
- Entry points live in `scripts/`.

## Folder structure

```
.
├── src/
│   └── csf_gnn/
│       ├── bias_edges.py
│       ├── datasets.py
│       ├── logging.py
│       ├── models.py
│       ├── train.py
│       ├── utils.py
├── scripts/
│   ├── run_grid.py
│   ├── german.py
│   ├── nba.py
│   ├── bail.py
│   ├── credit.py
│   └── income.py
└── data/
│   ├── nba.csv
│   ├── nba_edges.txt
│   ├── income.csv
│   ├── income_edges.txt
│   ├── credit.csv
│   ├── credit_edges.txt
│   ├── bail.csv
│   ├── bail_edges.txt
│   ├── german.csv
│   ├── german.txt
└── README
└── requirements
```

## Data files

- German: `german.csv`, `german_edges.txt`
- NBA: `nba.csv`, `nba_edges.txt`
- Bail: `bail.csv`, `bail_edges.txt`
- Credit: `credit.csv`, `credit_edges.txt`
- Income: `income.csv`, `income_edges.txt`


## Run

Run a single dataset’s default grid (same defaults as the notebook):

```bash
python scripts/german.py
```

Or use the unified runner:

```bash
python scripts/run_grid.py --dataset german
python scripts/run_grid.py --dataset credit --log my_credit_runs.csv
```

Outputs are appended to `experiments_<dataset>.csv`.

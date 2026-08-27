1. Program description

This project modifies the CNN structure to build ResCFE, a residual convolutional feature‑extraction module for retaining temporal information, which is coupled with LSTM and attention mechanisms for landslide displacement prediction.

2. Data Preparation & Format

The input is a CSV file sorted chronologically, containing the following columns:

|Column|Description|
|---|---|
| `date` | Date column, e.g. `2020-01-01`|
| `trend` |CEEMDAN trend component|
| `period` |CEEMDAN period component|
| `ZD1` |Raw cumulative displacement (for final evaluation)|
| `Reservior-water`| Covariate columns (reservoir water level)|

Sample Data
```csv
date,ZD1,trend,period,Reservior_water
2018-01-01,12.34,10.12,2.22,145.6
2018-01-02,12.45,10.18,2.27,145.8
2018-01-03,12.50,10.24,2.26,146.1
...
```
The monitoring data are not included in this repository. They are sourced from the Chinese National Cryosphere Desert Data Center ([http://www.ncdc.ac.cn](http://www.ncdc.ac.cn)). Due to data
restrictions, the authors do not have permission to upload or redistribute the data.  

3. Installation and execution

```bash
python -m pip install -r requirements.txt
```

```bash
python ResCFE_LSTM_Attention.py \
  --data_path data/ZD3_components.csv \
  --covariates RE1 \
  --seq_len 30 \
  --horizon 1 \
  --output results \
  --device auto
```

4. results 

 All results are saved under the directory specified by `--output` (default: `results/`).
 
|File|Description|
|---|---|
| `trend_grid_search_results.csv` |  All grid search results for trend (sorted by val MSE) |
| `trend_best_config.json` |  Best hyperparameter config for trend |
| `trend_best_model.pt` | Best model weights for trend |
| `trend_test_metrics.json` | Test metrics for trend |
| `Trend_metrics.txt` | Train & test metrics for trend (text) |
| `period_grid_search_results.csv` | All grid search results for period |
| `period_best_config.json` | Best hyperparameter config for period |
| `period_best_model.pt` | Best model weights for period |
| `period_test_metrics.json` | Test metrics for period |
| `Period_metrics.txt` | Train & test metrics for period (text) |
| `final_total_metrics.json` | Final metrics for total displacement (trend + period) |

5. Debug Run
```bash
# Search only the first 5 combinations for quick validation
python ResCFE_LSTM_Attention.py --data_path data/sample.csv --max-combinations 5
```
6.License
MIT License


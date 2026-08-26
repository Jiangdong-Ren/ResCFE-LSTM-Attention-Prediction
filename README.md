1. Program description
   This project modifies the CNN structure to build ResCFE, a residual convolutional feature‑extraction module for retaining temporal information, which is coupled with LSTM and attention mechanisms for landslide displacement prediction.
2. Input data
   This program adopts CEEMDAN‑decomposed and reconstructed landslide monitoring data, with CSV files as the mandatory input format. The first column corresponds to time; the second and third columns are target columns (`period_term`/`trend_term`) configured by the `--target` argument. The fourth column represents the covariate column, set through the `--features` argument.
   Example CSV：

    ```text
    date,period_term,trend_term,reservoir_water_level
    2017-01-01,12.50,0.0,174.20
    2017-01-02,12.63,5.2,174.35
    2017-01-03,12.71,0.8,174.10
    ```
3. Installation and execution
   ```bash
python -m pip install -r requirements.txt
```

Run the ResCFE_LSTM_Attention.py

```bash
python ResCFE_LSTM_Attention.py --data path/to/data.csv --target displacement --features rainfall reservoir --output results
```

For a target-only data set, omit `--features`. Use `--device cpu` to force CPU execution. Use `--max-combinations 1` only for a short debugging run; omit it for the complete search.

The program fits the Min-Max scaler using the training portion, trains every requested architecture combination, and uses validation MSE for model selection. 

4. results
All files are saved in the directory specified by `--output` (default: `results`). 
| File | Description |
| --- | --- |
| `grid_search_results.csv` | Validation MSE and hyperparameters for every evaluated combination. 
| `best_config.json` | The selected configuration and its validation MSE. 
| `test_metrics.json` | Test-set RMSE, MAE, and R² after inverse scaling. 
| `best_model.pt` | PyTorch state dictionary of the selected model. 

5. Test program 

The monitoring data are not included in this repository. They are sourced from the Chinese National Cryosphere Desert Data Center ([http://www.ncdc.ac.cn](http://www.ncdc.ac.cn)). Due to data
restrictions, the authors do not have permission to upload or redistribute the data.

test.py creates temporary synthetic data, checks the model forward pass, and runs one grid-search configuration. It does not require the unavailable monitoring data.

Run the test 

```bash
python test.py
```


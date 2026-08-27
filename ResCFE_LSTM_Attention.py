import argparse
import copy
import itertools
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
DROPOUT = 0.1

GRID = {
    "lstm_hidden": (32, 64, 128),
    "lstm_layers": (1, 2, 3),
    "cnn_filters": (16, 32, 64),
    "cnn_kernel": (3, 5, 7),
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


class ResCFE_LSTM_Attention(nn.Module):
    """Residual 1D convolution + stacked LSTM + additive Bahdanau-style attention."""

    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.cnn_filters = config["cnn_filters"]
        self.cnn_kernel = config["cnn_kernel"]
        self.lstm_hidden = config["lstm_hidden"]
        self.dropout_rate = config["dropout"]

        self.conv1 = nn.Conv1d(input_dim, self.cnn_filters, self.cnn_kernel,
                               padding=self.cnn_kernel // 2)
        self.residual_mapping = (
            nn.Conv1d(input_dim, self.cnn_filters, kernel_size=1)
            if input_dim != self.cnn_filters else nn.Identity()
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.lstm = nn.LSTM(self.cnn_filters, self.lstm_hidden,
                            num_layers=config.get("lstm_layers", 1),
                            batch_first=True, bidirectional=False)
        self.attention_linear = nn.Linear(self.lstm_hidden, self.lstm_hidden)
        self.query_linear = nn.Linear(self.lstm_hidden, self.lstm_hidden)
        self.score_linear = nn.Linear(self.lstm_hidden, 1)
        self.fc = nn.Linear(self.lstm_hidden * 2, output_dim)

    def attention(self, lstm_output, final_state):
        query = final_state.unsqueeze(1).repeat(1, lstm_output.shape[1], 1)
        energy = torch.tanh(self.attention_linear(lstm_output) + self.query_linear(query))
        scores = self.score_linear(energy)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(lstm_output * weights, dim=1)
        return context, weights

    def forward(self, x):
        x_perm = x.permute(0, 2, 1)
        cnn_out = self.dropout(self.relu(self.conv1(x_perm)))
        res_out = self.residual_mapping(x_perm)
        x = (cnn_out + res_out).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        final_state = lstm_out[:, -1, :]
        context, _ = self.attention(lstm_out, final_state)
        combined = torch.cat((context, final_state), dim=1)
        return self.fc(combined)


def prepare_data(df, target_col, covariate_cols, seq_len,
                 train_ratio=0.7, val_ratio=0.1, use_diff=False):
    target_data = df[target_col].values
    processed_target = (
        pd.Series(target_data).diff().fillna(0).values if use_diff else target_data
    )
    if covariate_cols:
        data_matrix = np.column_stack([processed_target, df[covariate_cols].values])
    else:
        data_matrix = processed_target.reshape(-1, 1)

    total_len = len(data_matrix)
    train_size = int(total_len * train_ratio)
    val_size = int(total_len * val_ratio)

    train_raw = data_matrix[:train_size]
    val_raw = data_matrix[train_size:train_size + val_size]
    test_raw = data_matrix[train_size + val_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw)
    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    if use_diff:
        test_start_idx = train_size + val_size
        base_value = target_data[test_start_idx - 1]
        raw_test_target_original = target_data[test_start_idx:]
    else:
        base_value = None
        raw_test_target_original = target_data[train_size + val_size:]

    return train_scaled, val_scaled, test_scaled, scaler, raw_test_target_original, base_value


def create_dataset(data, seq_len):
    Xs, ys = [], []
    if len(data) <= seq_len:
        return np.array(Xs), np.array(ys)
    for i in range(len(data) - seq_len):
        Xs.append(data[i:(i + seq_len)])
        ys.append(data[i + seq_len, 0])
    return np.array(Xs), np.array(ys)


def evaluate_split(model, data_seq, seq_len, scaler, device):
    X, y_true = create_dataset(data_seq, seq_len)
    if len(X) == 0:
        return None
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X).to(device)).cpu().numpy().squeeze()
    y_true = y_true.squeeze()
    min_t, max_t = scaler.data_min_[0], scaler.data_max_[0]
    y_true_inv = y_true * (max_t - min_t) + min_t
    y_pred_inv = preds * (max_t - min_t) + min_t
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))),
        "mae": float(mean_absolute_error(y_true_inv, y_pred_inv)),
        "mape": float(calculate_mape(y_true_inv, y_pred_inv)),
        "r2": float(r2_score(y_true_inv, y_pred_inv)),
    }


def save_metrics_txt(name, train_metrics, test_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}_metrics.txt")
    with open(path, "w", encoding="utf-8") as f:
        if train_metrics:
            f.write("[Train]\n")
            f.write(f"RMSE: {train_metrics['rmse']:.6f}\n")
            f.write(f"MAE:  {train_metrics['mae']:.6f}\n")
            f.write(f"MAPE: {train_metrics['mape']:.6f}%\n")
            f.write(f"R2:   {train_metrics['r2']:.6f}\n\n")
        if test_metrics:
            f.write("[Test]\n")
            f.write(f"RMSE: {test_metrics['rmse']:.6f}\n")
            f.write(f"MAE:  {test_metrics['mae']:.6f}\n")
            f.write(f"MAPE: {test_metrics['mape']:.6f}%\n")
            f.write(f"R2:   {test_metrics['r2']:.6f}\n")


def recursive_predict_multivariate(model, history_seq, future_covariates, horizon, device):
    model.eval()
    preds = []
    current_seq = torch.FloatTensor(history_seq).unsqueeze(0).to(device)
    with torch.no_grad():
        for i in range(horizon):
            pred_val = model(current_seq).item()
            preds.append(pred_val)
            if future_covariates is not None and len(future_covariates) > i:
                new_step_features = np.concatenate(([pred_val], future_covariates[i]))
            else:
                new_step_features = current_seq[0, -1, :].cpu().numpy()
                new_step_features[0] = pred_val
            new_point = torch.tensor(new_step_features.astype(np.float32)).view(1, 1, -1).to(device)
            current_seq = torch.cat([current_seq[:, 1:, :], new_point], dim=1)
    return np.array(preds)


def test_rolling_forecast(model, test_data, train_val_data, seq_len, horizon, device, scaler):
    predictions = []
    full_history = np.concatenate([train_val_data, test_data])
    test_start_idx = len(train_val_data)
    total_test_len = len(test_data)
    n_feats = test_data.shape[1]
    min_target, max_target = scaler.data_min_[0], scaler.data_max_[0]

    for i in range(0, total_test_len, horizon):
        current_idx = test_start_idx + i
        current_horizon = min(horizon, total_test_len - i)
        input_seq = full_history[current_idx - seq_len:current_idx]
        future_covs = (
            full_history[current_idx:current_idx + current_horizon, 1:]
            if n_feats > 1 else None
        )
        pred_scaled = recursive_predict_multivariate(
            model, input_seq, future_covs, current_horizon, device
        )
        pred_inv = pred_scaled * (max_target - min_target) + min_target
        predictions.extend(pred_inv)
    return np.array(predictions)


def train_one(train_seq, val_seq, seq_len, config, device, epochs=EPOCHS):
    X_train, y_train = create_dataset(train_seq, seq_len)
    X_val, y_val = create_dataset(val_seq, seq_len)
    input_dim = train_seq.shape[1]

    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train).to(device),
            torch.FloatTensor(y_train).unsqueeze(1).to(device),
        ),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    model = ResCFE_LSTM_Attention(input_dim=input_dim, output_dim=1, config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )

    best_loss = float("inf")
    best_model = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        if len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                v_loss = criterion(
                    model(torch.FloatTensor(X_val).to(device)),
                    torch.FloatTensor(y_val).unsqueeze(1).to(device),
                ).item()
            scheduler.step(v_loss)
            if v_loss < best_loss:
                best_loss = v_loss
                best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    return model, best_loss


def grid_search_component(name, train_seq, val_seq, test_seq, seq_len, scaler,
                          device, output_dir, max_combinations=None):
    os.makedirs(output_dir, exist_ok=True)
    keys = tuple(GRID)
    total = 3 ** len(GRID)
    combinations = itertools.product(*(GRID[key] for key in keys))
    rows, best = [], None

    for index, values in enumerate(combinations, start=1):
        if max_combinations is not None and index > max_combinations:
            break
        config = dict(zip(keys, values))
        config["dropout"] = DROPOUT
        print(f"[{name}] [{index}/{total}] {config}")
        model, val_loss = train_one(train_seq, val_seq, seq_len, config, device)
        row = {**config, "validation_mse": val_loss, "combination": index}
        rows.append(row)
        if best is None or val_loss < best["validation_mse"]:
            best = {**config, "validation_mse": val_loss, "model": model}

    if best is None:
        raise ValueError("No grid combinations were evaluated.")

    results = pd.DataFrame(rows).sort_values("validation_mse").reset_index(drop=True)
    prefix = name.lower()
    results.to_csv(os.path.join(output_dir, f"{prefix}_grid_search_results.csv"), index=False)
    torch.save(best["model"].state_dict(), os.path.join(output_dir, f"{prefix}_best_model.pt"))

    summary = {key: best[key] for key in (*keys, "validation_mse")}
    with open(os.path.join(output_dir, f"{prefix}_best_config.json"), "w") as f:
        json.dump(summary, f, indent=2)

    train_metrics = evaluate_split(best["model"], train_seq, seq_len, scaler, device)
    test_metrics = evaluate_split(best["model"], test_seq, seq_len, scaler, device)
    save_metrics_txt(name, train_metrics, test_metrics, output_dir)
    with open(os.path.join(output_dir, f"{prefix}_test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    return best["model"], summary, results


def main():
    parser = argparse.ArgumentParser(
        description="ResCFE-LSTM-Attention with grid search for CEEMDAN landslide displacement prediction"
    )
    parser.add_argument("--data_path", type=str, required=True, help="CSV file path")
    parser.add_argument("--date_col", type=str, default="date", help="Date column name")
    parser.add_argument("--seq_len", type=int, default=30, help="Input sequence length")
    parser.add_argument("--horizon", type=int, default=1, help="Rolling forecast horizon")
    parser.add_argument("--covariates", nargs="+", default=["Reservior_water"],
                        help="Covariate columns for the period component")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-combinations", type=int, default=None,
                        help="Limit grid search combinations (debug)")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    print(f"Using device: {device}")

    df = pd.read_csv(args.data_path)
    df.columns = [c.strip() for c in df.columns]
    if args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col])

    os.makedirs(args.output, exist_ok=True)
    final_preds, final_trues, test_start_indices = {}, {}, {}

    # ---- Trend component (first-order differencing) ----
    if "trend" in df.columns:
        print("\n>>> Trend Component")
        t_train, t_val, t_test, t_scaler, t_true_origin, _ = prepare_data(
            df, "trend", [], args.seq_len, use_diff=True
        )
        t_model, t_best, _ = grid_search_component(
            "Trend", t_train, t_val, t_test, args.seq_len, t_scaler,
            device, args.output, args.max_combinations,
        )
        print(f"Best config: {t_best}")

        t_train_val = np.concatenate([t_train, t_val])
        t_pred_diff = test_rolling_forecast(
            t_model, t_test, t_train_val, args.seq_len, args.horizon, device, t_scaler
        )
        t_pred_reconstructed = []
        full_trend_true = df["trend"].values
        test_start_idx = len(full_trend_true) - len(t_test)
        for i in range(0, len(t_pred_diff), args.horizon):
            chunk_start = test_start_idx + i
            base = full_trend_true[chunk_start - 1]
            chunk_diffs = t_pred_diff[i:i + args.horizon]
            t_pred_reconstructed.extend(base + np.cumsum(chunk_diffs))

        final_preds["trend"] = np.array(t_pred_reconstructed)
        final_trues["trend"] = t_true_origin
        test_start_indices["trend"] = len(t_train) + len(t_val)

    # ---- Period component (with covariates) ----
    if "period" in df.columns:
        print("\n>>> Period Component")
        p_train, p_val, p_test, p_scaler, p_true_origin, _ = prepare_data(
            df, "period", args.covariates, args.seq_len, use_diff=False
        )
        p_model, p_best, _ = grid_search_component(
            "Period", p_train, p_val, p_test, args.seq_len, p_scaler,
            device, args.output, args.max_combinations,
        )
        print(f"Best config: {p_best}")

        p_train_val = np.concatenate([p_train, p_val])
        p_pred = test_rolling_forecast(
            p_model, p_test, p_train_val, args.seq_len, args.horizon, device, p_scaler
        )
        final_preds["period"] = p_pred
        final_trues["period"] = p_true_origin
        test_start_indices["period"] = len(p_train) + len(p_val)

    # ---- Final aggregation ----
    if "trend" in final_preds and "period" in final_preds:
        min_len = min(len(v) for v in final_preds.values())
        pred_trend = final_preds["trend"][:min_len]
        pred_period = final_preds["period"][:min_len]
        target_col = "ZD1" if "ZD1" in df.columns else df.columns[1]
        true_total = df[target_col].values[-min_len:]
        pred_total = pred_trend + pred_period

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(true_total, pred_total))),
            "mae": float(mean_absolute_error(true_total, pred_total)),
            "mape": float(calculate_mape(true_total, pred_total)),
            "r2": float(r2_score(true_total, pred_total)),
        }
        print(f"\n{'=' * 50}")
        print("Final Total Displacement Metrics")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"MAPE: {metrics['mape']:.4f}%")
        print(f"R2:   {metrics['r2']:.4f}")
        print(f"{'=' * 50}")
        with open(os.path.join(args.output, "final_total_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

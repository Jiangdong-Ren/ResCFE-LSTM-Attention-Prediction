import argparse
import copy
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Fixed parameters
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 30

#Grid search range
GRID = {
    "lstm_hidden": (32, 64, 128),
    "lstm_layers": (1, 2, 3),
    "cnn_filters": (16, 32, 64),
    "cnn_kernel": (3, 5, 7),
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResCFELSTMAttention(nn.Module):
    """Residual one-dimensional convolution, LSTM stack, and additive attention."""

    def __init__(self, input_dim: int, config: dict[str, int], dropout: float = 0.1):
        super().__init__()
        filters = config["cnn_filters"]
        hidden = config["lstm_hidden"]
        kernel = config["cnn_kernel"]
        self.conv = nn.Conv1d(input_dim, filters, kernel, padding=kernel // 2)
        self.residual = nn.Conv1d(input_dim, filters, 1) if input_dim != filters else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(filters, hidden, num_layers=config["lstm_layers"], batch_first=True,
                            dropout=dropout if config["lstm_layers"] > 1 else 0.0)
        self.key = nn.Linear(hidden, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.score = nn.Linear(hidden, 1)
        self.output = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = torch.relu(self.conv(z)) + self.residual(z)
        z = self.dropout(z).transpose(1, 2)
        sequence, _ = self.lstm(z)
        last = sequence[:, -1]
        energy = torch.tanh(self.key(sequence) + self.query(last).unsqueeze(1))
        weights = torch.softmax(self.score(energy), dim=1)
        context = (sequence * weights).sum(dim=1)
        return self.output(torch.cat((context, last), dim=1))


@dataclass
class SplitData:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    scaler: MinMaxScaler


def load_and_split(path: str | Path, target: str, features: list[str], train_ratio: float = 0.7,
                   validation_ratio: float = 0.1) -> SplitData:
    frame = pd.read_csv(path)
    columns = [target] + features
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    values = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    n_train = int(len(values) * train_ratio)
    n_validation = int(len(values) * validation_ratio)
    n_test = len(values) - n_train - n_validation
    if n_train <= SEQUENCE_LENGTH or n_validation < 1 or n_test < 1:
        raise ValueError("The data set is too short for sequence_length=30 and the requested split.")
    scaler = MinMaxScaler().fit(values[:n_train])
    scaled = scaler.transform(values).astype(np.float32)
    return SplitData(scaled[:n_train], scaled[n_train:n_train + n_validation],
                     scaled[n_train + n_validation:], scaler)


def make_windows(values: np.ndarray, sequence_length: int = SEQUENCE_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= sequence_length:
        return np.empty((0, sequence_length, values.shape[1]), dtype=np.float32), np.empty((0, 1), dtype=np.float32)
    x = np.stack([values[i:i + sequence_length] for i in range(len(values) - sequence_length)])
    y = values[sequence_length:, 0:1]
    return x.astype(np.float32), y.astype(np.float32)


def validation_windows(train: np.ndarray, validation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    context = np.concatenate((train[-SEQUENCE_LENGTH:], validation), axis=0)
    return make_windows(context)


def train_one(train: np.ndarray, validation: np.ndarray, config: dict[str, int], device: torch.device,
              epochs: int = EPOCHS) -> tuple[ResCFELSTMAttention, float]:
    x_train, y_train = make_windows(train)
    x_val, y_val = validation_windows(train, validation)
    model = ResCFELSTMAttention(train.shape[1], config).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_state, best_loss = copy.deepcopy(model.state_dict()), float("inf")
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(torch.from_numpy(x_val).to(device)), torch.from_numpy(y_val).to(device)).item()
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_loss


def inverse_metrics(model: nn.Module, history: np.ndarray, future: np.ndarray, scaler: MinMaxScaler,
                    device: torch.device) -> dict[str, float]:
    windows, y = validation_windows(history, future)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(windows).to(device)).cpu().numpy().reshape(-1)
    target_min, target_max = scaler.data_min_[0], scaler.data_max_[0]
    truth = y.reshape(-1) * (target_max - target_min) + target_min
    predictions = predictions * (target_max - target_min) + target_min
    return {
        "rmse": float(np.sqrt(mean_squared_error(truth, predictions))),
        "mae": float(mean_absolute_error(truth, predictions)),
        "r2": float(r2_score(truth, predictions)),
    }


def grid_search(data: SplitData, device: torch.device, output_dir: Path, max_combinations: int | None = None,
                epochs: int = EPOCHS) -> tuple[dict, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = tuple(GRID)
    combinations = itertools.product(*(GRID[key] for key in keys))
    rows, best = [], None
    for index, values in enumerate(combinations, start=1):
        if max_combinations is not None and index > max_combinations:
            break
        config = dict(zip(keys, values))
        print(f"[{index}] {config}")
        model, validation_loss = train_one(data.train, data.validation, config, device, epochs)
        row = {**config, "validation_mse": validation_loss, "combination": index}
        rows.append(row)
        if best is None or validation_loss < best["validation_mse"]:
            best = {**config, "validation_mse": validation_loss, "model": model}
    if best is None:
        raise ValueError("No grid combinations were evaluated.")
    results = pd.DataFrame(rows).sort_values("validation_mse").reset_index(drop=True)
    results.to_csv(output_dir / "grid_search_results.csv", index=False)
    torch.save(best["model"].state_dict(), output_dir / "best_model.pt")
    summary = {key: best[key] for key in (*keys, "validation_mse")}
    (output_dir / "best_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    test_metrics = inverse_metrics(best["model"], np.concatenate((data.train, data.validation)), data.test,
                                   data.scaler, device)
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV file; target is the first selected column.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--features", nargs="*", default=[], help="Optional covariate column names.")
    parser.add_argument("--output", default="results", help="Directory for search results and model.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-combinations", type=int, default=None, help="Debug option; omit for all 81 combinations.")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    data = load_and_split(args.data, args.target, args.features)
    best, results = grid_search(data, device, Path(args.output), args.max_combinations)
    print(json.dumps({"evaluated": len(results), "best": best, "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()

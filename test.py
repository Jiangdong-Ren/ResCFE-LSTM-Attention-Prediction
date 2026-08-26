import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ResCFE_LSTM_Attention import (
    GRID,
    ResCFELSTMAttention,
    grid_search,
    load_and_split,
    make_windows,
    set_seed,
)


def main() -> None:
    set_seed(7)
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        n = 180
        t = np.arange(n, dtype=np.float32)
        frame = pd.DataFrame({"target": np.sin(t / 8), "covariate": np.cos(t / 11)})
        data_path = root / "synthetic.csv"
        frame.to_csv(data_path, index=False)
        data = load_and_split(data_path, "target", ["covariate"])
        config = {key: values[0] for key, values in GRID.items()}
        model = ResCFELSTMAttention(2, config)
        sample, _ = make_windows(data.train)
        sample = torch.from_numpy(sample[:2])
        assert model(sample).shape == (2, 1)
        best, results = grid_search(data, torch.device("cpu"), root / "results", max_combinations=1, epochs=2)
        assert len(results) == 1 and set(best) >= set(GRID)
    print("test passed")


if __name__ == "__main__":
    main()

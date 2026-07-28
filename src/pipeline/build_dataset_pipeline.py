from src.dataset import (
    DatasetBuilder,
    DatasetLoader,
    DatasetSaver,
)
from src.pipeline import SignalPreprocessor
from src.windowing import WindowGenerator

DATA_DIR = "data/raw"  # Change to your .bin folder

# Step 1: Preprocess
signals = SignalPreprocessor.run(DATA_DIR)

# Step 2: Create windows
windows = WindowGenerator.generate(signals)

# Step 3: Build datasets
datasets, metadata = DatasetBuilder.build(
    signals,
    windows,
)

# Step 4: Save datasets
DatasetSaver.save(
    datasets,
    metadata,
)

# Step 5: Reload datasets
datasets, metadata = DatasetLoader.load()

print("\n========== SUMMARY ==========\n")

for name in datasets:
    print(f"{name:<6}" f" Shape: {datasets[name].shape}")

print("\nPipeline completed successfully.")

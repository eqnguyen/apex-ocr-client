from pathlib import Path

# Path to log directory
LOG_DIRECTORY = Path(__file__).parent / "logs"
LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Path to data
DATA_DIRECTORY = Path(__file__).parent / "data"
DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

import os

# Ensure logs directory exists
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)  # Create logs directory if it doesn’t exist

# Define the log file path
LOG_FILE_PATH = os.path.join(LOG_DIR, "app.log")

# Split ratios
RATIOS = (0.5, 0.3, 0.2)
NAME = "BTC-USD"

# Sequence and prediction lengths
SEQ_LENGTH = 30
PRED_LENGTH = 1

# Model architecture
INPUT_SIZE = 25 
HIDDEN_SIZE = 128
DROPOUT_RATE = 0.1

NUM_LAYERS=2   
DROPOUT=0.1

# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 30
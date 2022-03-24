from ml import predict_new
from joblib import load
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
import threading as th
import numpy as np

# Load in models to use for predictions
models = ['test_7_rf.pkl', 'test_7_log.pkl', 'test_7_rf_raw.pkl']
#models = ['test_rbg1.pkl', 'test_rbg2.pkl']

PREDICT_EVERY = 1.9
INCLUDE_LAST = 400

params = BrainFlowInputParams()
params.serial_port = 'COM4'
board_id = BoardIds.CYTON_DAISY_BOARD.value
board = BoardShim(board_id, params)


# Set which are eeg channels
eeg_channels = BoardShim.get_eeg_channels(board_id)
eeg_channels.append(-1)

#  Load models
models_and_params = [load(m) for m in models]

# Prepare session
board.prepare_session()
board.start_stream()

# Weird at start, same w/ training, also
# we could use the little gap at the start
time.sleep(5)

# Grab everything and keep in variable buffer
buffer = board.get_board_data()[eeg_channels, -INCLUDE_LAST:]

keep_going = True
def key_capture_thread():
    global keep_going
    input()
    keep_going = False

def run_predictions(board, buffer):
    th.Thread(target=key_capture_thread, args=(),
    name='key_capture_thread', daemon=True).start()
    while keep_going:

        # Record for a fixed amount of time
        time.sleep(PREDICT_EVERY)

        # Grap all of the saved data
        data = board.get_board_data()[eeg_channels, :]

        # Pass the data as buffer + data
        predict_data = np.concatenate([buffer, data], axis=1)

        # Set the new buffer to 
        buffer = predict_data[:, -INCLUDE_LAST:]

        # Predict
        pred_color, pred_probs = predict_new(predict_data, models_and_params)
        print(pred_color, pred_probs)

# Start predictions loop
run_predictions(board, buffer)

# After end
board.stop_stream()
board.release_session()
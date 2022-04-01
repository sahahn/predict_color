from ml import predict_new
from joblib import load
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
import threading as th
import numpy as np
from load import EVENT_IDS

# Load in models to use for predictions
models = ['all_1.pkl', 'all_5.pkl']

PREDICT_EVERY = 1.9
INCLUDE_LAST = 400

params = BrainFlowInputParams()
params.serial_port = 'COM4'
board_id = BoardIds.SYNTHETIC_BOARD.value
#board_id = BoardIds.CYTON_DAISY_BOARD.value
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

def norm(p):
    scale = 1 / np.sum(p)
    return p * scale

def run_predictions(board, buffer):
    th.Thread(target=key_capture_thread, args=(),
    name='key_capture_thread', daemon=True).start()

    # Init penalties, penalize predicting the same
    # color too many times
    penalties = np.ones(len(EVENT_IDS))

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
        _, pred_probs = predict_new(predict_data, models_and_params)

        # Instead of using the base prediction, we define some logic to make
        # the predictions more likely to choose other colors
        # Use base predictions
        p = np.array(list(pred_probs.values()))

        # Divide by previous, so reduces chance of selecting previously chosen
        p /= penalties

        # Choices are the colors
        choices = np.array(list(pred_probs))

        # Make choice as weighted probability
        color_choice = np.random.choice(choices, p=norm(p))

        # Increment penalties
        penalties[np.where(choices == color_choice)[0][0]] += 1

        print(color_choice)


# Start predictions loop
run_predictions(board, buffer)

# After end
board.stop_stream()
board.release_session()
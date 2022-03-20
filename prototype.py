import tkinter as tk
import random
import time
import numpy as np
import os
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Location to save recorded session
save_dr = 'data'

# Color to show after and before every other
base_color = 'black'

# Colors to cycle through
color_codes = {'red': 2, 'green': 3, 'blue': 4}
colors = list(color_codes)

# Duration of each baseline
base_dur = 3

# Duration of each color
color_dur = 3

# Number of times to repeat
repeats = 1

# Create tkinter window, fullscreen
window = tk.Tk()
window.attributes('-fullscreen', True)

# Init Board
params = BrainFlowInputParams()
params.serial_port = 'COM4'
board_id = BoardIds.CYTON_DAISY_BOARD.value
board = BoardShim(board_id, params)

# Get some info based on board id
ch_names = BoardShim.get_eeg_names(board_id)
sfreq = BoardShim.get_sampling_rate(board_id)

# Get which are eeg channels
eeg_channels = BoardShim.get_eeg_channels(board_id)

# Also keep track of last marker channel
eeg_channels.append(-1)

# Prepare session
board.prepare_session()
board.start_stream()

def show_color(color, dur):
    
    window.configure(background=color)
    window.update()
    time.sleep(dur)

# Weird at start
time.sleep(5)

# For each desired repeat
for _ in range(repeats):
    
    # Show baseline then color
    random.shuffle(colors)
    for color in colors:

        # Add marker for base
        board.insert_marker(1)

        # Show base
        show_color(base_color, base_dur)

        # Add marker for color
        board.insert_marker(color_codes[color])

        # Show color of interest
        show_color(color, color_dur)

# Stop recording
board.stop_stream()

# Extract recorded data
data = board.get_board_data()

# Get just channels of interest
eeg_data = data[eeg_channels, :]

# At end, make sure to release session
board.release_session()

# Make sure data dr exists
param_dr = os.path.join(save_dr, f'{base_dur}_{color_dur}')
os.makedirs(param_dr, exist_ok=True)

# Check to see what runs already exist
existing = [int(f.replace('.npy', '')) for f in os.listdir(param_dr)]
if len(existing) == 0:
    next_save = 0
else:
    next_save = max(existing) + 1

print('Saving EEG data with shape:', eeg_data.shape)
print('Saves at:', np.where(eeg_data[-1] != 0))

# Save
save_loc = os.path.join(param_dr, str(next_save) + '.npy')
np.save(save_loc, eeg_data)

print (save_loc)
print ('Tag1 : ', ch_names)
print ('Tag2 : ', sfreq)
print ('Tag3 : ', eeg_channels)

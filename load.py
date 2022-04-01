import os
import numpy as np
from brainflow.data_filter import DataFilter
import mne
from scipy import stats
import warnings

try:
    from autoreject import AutoReject, Ransac
except ImportError:
    pass

# Fixed parameters related to the type of EEG used
SAMPLING_RATE = 125
CH_NAMES = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8', 'F5', 'F7', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8']
N_CHANNELS = len(CH_NAMES)
CH_TYPES = ['eeg'] * N_CHANNELS

# Might change based on parameters of setup
# Note: the baseline color is always 1, and shouldn't
# be referenced explicitly here
EVENT_IDS = {'red' :2 , 'green' : 3, 'blue' :4,
             '#FC5C04' :5 , '#844CD4' :6 , '#2CFCFC' :7, '#FCFB20' : 8,
             '#FC5C9C' : 9,'#7C04FC' : 10,'#9C3C34' : 11,
             '#E02B2C' : 12,'#1074CA' : 13, '#D7E3F7': 14,
             '#FFFFFF' : 15}
REV_EVENT_IDS = {EVENT_IDS[e]: e for e in EVENT_IDS}

def load_runs(data_dr):
    
    # Get avaliable runs
    runs = os.listdir(data_dr)
    run_files = [os.path.join(data_dr, f) for f in runs]
    
    # Load in as list
    data = [np.load(f) for f in run_files]

    return data

def extract_avg_power(runs):
    
    # Save features in dictionary
    feats = {}

    # Process eeach run
    for data in runs:
        
        # Isolate events
        event_ids = data[-1]
        event_inds = np.where(event_ids>=1)[0]

        # For each event
        for i in range(len(event_inds)):

            # Extract correct chunk of data corresponding to
            # to just that label
            try:
                chunk = data[:, event_inds[i]:event_inds[i+1]]
            except IndexError:
                chunk = data[:, event_inds[i]:]
                
            chunk = np.ascontiguousarray(chunk)

            # Find associated label with this epoch / chunk
            label = event_ids[event_inds[i]]
            
            # Extract avg band powers per channel
            feat_vector = []
            for i in range(N_CHANNELS):
                eeg_channels = [i]
                bands = DataFilter.get_avg_band_powers(chunk, eeg_channels, SAMPLING_RATE, True)
                feat_vector += list(bands[0])

            # Add to feats under correct label
            feat_vector = np.array(feat_vector)

            try:
                feats[label].append(feat_vector)
            except KeyError:
                feats[label] = [feat_vector]
                
    return feats

def split_by(feats, g1, g2):
    
    x1 = sum([feats[g] for g in g1], [])
    labels = [0 for _ in range(len(x1))]
    
    x2 = sum([feats[g] for g in g2], [])
    labels += [1 for _ in range(len(x2))]
    
    x = x1 + x2
    
    return np.array(x), np.array(labels)

def conv_to_mne_raw(run_data):
    '''The input run_data here is the raw 17 channel
    recorded output from one saved session.'''

    # Do not include last channel
    data = run_data[:-1]

    # Convert from uV to V
    data = data / 1000000

    # Create MNE info object
    info = mne.create_info(ch_names=CH_NAMES, sfreq=SAMPLING_RATE, ch_types=CH_TYPES)
    
    # Set montage giving info about where channels are in real space
    info.set_montage(mne.channels.make_standard_montage('standard_1020'))

    # Convert data into mne RawArray object
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Return RawArray + last channel, which
    # includes the label information
    return raw, run_data[-1]

def extract_events(event_channel):
    
    # Get vals that correspond to colors shown
    # Don't include the baseline color here
    vals = list(EVENT_IDS.values())

    # Get locs
    event_locs = np.where(np.isin(event_channel, vals))[0]

    # Put into expected 3-column format for MNE (loc, 0, label)
    events = np.stack([event_locs,
                       np.zeros(len(event_locs)),
                       event_channel[event_locs]], axis=1).astype('int')
    
    return events

def conv_raw_to_epochs(raw, event_channel, tmin=-2, tmax=1.9):
    
    # Extract events from event channel in expected format
    events = extract_events(event_channel)

    # Set event ids to just subset
    event_ids = {e: EVENT_IDS[e] for e in EVENT_IDS if EVENT_IDS[e] in events[:, -1]}

    # Conv to Epochs object
    # Note: don't apply any baseline correction here
    epochs = mne.Epochs(raw, events, event_ids,
                        tmin=tmin, tmax=tmax,
                        baseline=None, preload=True,
                        verbose=False)
    
    return epochs

def get_as_epochs(runs, l_freq=None, h_freq=None, tmin=-2, tmax=1.9):

    # Load epochs for each run
    all_epochs = []
    for run in runs:
        
        # Load raw + event channel
        raw, event_channel = conv_to_mne_raw(run)
        
        # Apply signal filtering if either l_freq or h_freq is passed
        # otherwise skip.
        if l_freq is not None and h_freq is not None:
            raw = raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

        # Convert this run to epochs and append
        epochs = conv_raw_to_epochs(raw, event_channel, tmin=tmin, tmax=tmax)
        all_epochs.append(epochs)
        
    # Concat epochs from each session and return
    concat_epochs = mne.concatenate_epochs(all_epochs)
    return concat_epochs

def apply_auto_reject(concat_epochs, auto_reject):

    # Skip if None Requested
    if auto_reject is None:
        return concat_epochs
    if auto_reject is False:
        return concat_epochs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Use base AutoReject object in these cases
        if auto_reject is True or auto_reject == 'ar':
            return AutoReject(verbose=False, random_state=2).fit_transform(concat_epochs)
        
        # Can alternatively use Ransac
        elif auto_reject == 'ransac':
            return Ransac(verbose=False, random_state=2).fit_transform(concat_epochs)
        
        # Some other str, raise error
        else:
            raise RuntimeError('invalid auto_reject passed')

def conv_runs_to_epochs(runs,
                        l_freq=None, h_freq=None,
                        tmin=-2, tmax=1.9,
                        auto_reject=False, ica=None,
                        set_average_ref=False,
                        apply_baseline=False,
                        drop_ref_ch=False,
                        crop=False):
    '''The order of the optional params are the order in which
    these various options are applied if at all, e.g.,
    the passed l_freq and h_freq are applied first to the raw
    signal from each session, then the data is epoched according
    to the passed tmin and tmax, then optionally auto reject is applied
    and so on.
    
    By default, none of the processing options are applied.'''

    # Load, conv to raw, then epoch, then concat
    concat_epochs = get_as_epochs(runs,
                                  l_freq=l_freq, h_freq=h_freq,
                                  tmin=tmin, tmax=tmax)
    
    # Optionally apply auto reject
    concat_epochs = apply_auto_reject(concat_epochs, auto_reject)
    
    # If using ICA, passed param will be list of
    # components to remove
    if ica is not None:

        # Mute warnings ... 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ica_obj = mne.preprocessing.ICA(verbose=False, random_state=2)
            ica_obj.fit(concat_epochs)
            concat_epochs = ica_obj.apply(concat_epochs, exclude=ica)
    
    else:
        ica_obj = None
    
    # Set ref to average
    if set_average_ref:
        concat_epochs = concat_epochs.set_eeg_reference('average')

    # Apply baseline correction to all epochs here if at all
    if apply_baseline:
        concat_epochs = concat_epochs.apply_baseline((tmin, 0))
        
    # Drop ref channels after average, those are channels w/ z
    if drop_ref_ch:
        concat_epochs = concat_epochs.drop_channels(['Fz', 'Cz', 'Pz', 'Oz'])
    
    # Optionally crop out basseline
    if crop:
        concat_epochs = concat_epochs.crop(0, tmax)
    
    # Return final concat epochs w/ requested processing applied
    return concat_epochs, ica_obj

def proc_new(new_data, l_freq=None, h_freq=None,
             epoch_len=239, ica_obj=None, ica=None,
             set_average_ref=True, drop_ref_ch=False,
             **params):
    '''The passed raw data here should have the last two seconds be the event of interest,
    so that filtering will work out nicer'''

    # Conv to mne raw
    raw, _ = conv_to_mne_raw(new_data)

    # Apply filter if any
    if l_freq is not None and h_freq is not None:
        raw = raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

    # Instead of converting to Epoch, just crop epoch len
    raw._data = raw._data[:, -epoch_len:]

    # Apply the saved ica if any
    if ica is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = ica_obj.apply(raw, exclude=ica)

    # Set average channel ref if requested
    if set_average_ref:
        raw = raw.set_eeg_reference('average')

    # We skip apply baseline, as we are assuming cont. prediction
    # and therefore don't want to use this extra info

    # Note, we also skip crop
    if drop_ref_ch:
        raw = raw.drop_channels(['Fz', 'Cz', 'Pz', 'Oz'])
    
    # Return as np array
    return raw.get_data()

def extract_summary_stats(epochs):
    '''Extract from an epochs object, a series
    of summary statistics per channel'''

    # Get data as numpy array
    if isinstance(epochs, np.ndarray):
        data = epochs
    else:
        data = epochs.get_data()

    def rms(x, axis=-1):
        return np.sqrt(np.mean(x ** 2, axis=axis))

    def abs_diff_signal(x, axis=-1):
        return np.sum(np.abs(np.diff(x, axis=axis)), axis=axis)

    # List of funcs to apply, all that take data then axis arg
    funcs = [np.mean, np.std, np.ptp, np.var, np.min,
             np.max, np.argmin, np.argmax, rms, abs_diff_signal,
             stats.skew, stats.kurtosis]

    # Get as concat numpy array, applies each func per channel per epoch
    concat_feats = np.concatenate([func(data, axis=-1) for func in funcs], axis=-1)

    return concat_feats


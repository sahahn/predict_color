
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from mne.decoding import Vectorizer
from sklearn.svm import SVC
import warnings

from load import proc_new, extract_summary_stats, REV_EVENT_IDS


def get_pipe(model='logistic'):

    if model == 'logistic':
        base_model = LogisticRegression(solver='lbfgs', max_iter=100)
    elif model == 'logistic_cv':
        base_model = LogisticRegressionCV(solver='saga', max_iter=5000,
                                          penalty='elasticnet',
                                          l1_ratios=[.1, .5, .7, .9], n_jobs=8)
    elif model == 'rf':
        base_model = RandomForestClassifier(n_jobs=8)
    
    elif model == 'svm':
        base_model = SVC(probability=True)


    # Init pipeline
    clf = make_pipeline(Vectorizer(), StandardScaler(), base_model)

    return clf

def eval(data, labels, model='logistic'):

    # Get the pipeline
    clf =  get_pipe(model='logistic')

    # Eval w/ stratified balanced acc
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_bac = np.mean(cross_val_score(clf, data, labels, scoring='balanced_accuracy', cv=cv))

    return mean_bac

def eval_all(data, labels, verbose=True):

    means = []
    for model in ['logistic', 'rf']:
        mean_bac = eval(data, labels, model)
        means.append(mean_bac)

        if verbose:
            print(f'{model}: {mean_bac}')

    # Return avg from both models
    return np.mean(means)

def predict_new(new_data, models_and_params):
    '''Note all passed trained models must have same classes.'''

    # Design for multiple, if just one, match format
    if not isinstance(models_and_params, list):
        models_and_params = [models_and_params]

    # For each passed model, proc and then predicts
    pred_probas = []
    for m_and_p in models_and_params:

        # Unpack
        model, params = m_and_p

        # Preproc
        data = proc_new(new_data, **params)

        # If this model uses summary stats, conv
        if params['use_summary']:
            data = extract_summary_stats(data)

        # Predict w/ pred proba
        pred_probas.append(model.predict_proba([data]))
        classes = model.classes_

    # Get mean across each
    pred_probas = np.array(pred_probas)
    mean_preds = np.mean(pred_probas, axis=0)[0]

    # Get single highest pred
    pred = classes[np.argmax(mean_preds)]
    pred_color = REV_EVENT_IDS[pred]

    # Also make dict w/ each prob
    pred_probs = {}
    for pred, cls in zip(mean_preds, classes):
        pred_probs[REV_EVENT_IDS[cls]] = pred

    return pred_color, pred_probs

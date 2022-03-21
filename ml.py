
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from mne.decoding import Vectorizer
import warnings


def eval(data, labels, model='logistic'):

    if model == 'logistic':
        base_model = LogisticRegression(solver='lbfgs', max_iter=100)
    elif model == 'logistic_cv':
        base_model = LogisticRegressionCV(solver='saga', max_iter=5000,
                                          penalty='elasticnet',
                                          l1_ratios=[.1, .5, .7, .9], n_jobs=8)
    elif model == 'rf':
        base_model = RandomForestClassifier(n_jobs=8)


    # Init pipeline
    clf = make_pipeline(Vectorizer(), StandardScaler(), base_model)

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
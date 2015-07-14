#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: high_gmm.py
# date: Tue July 14 15:40 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""high_gmm:

"""

from __future__ import division

from contextlib import contextmanager
import sys
import numpy as np
import os
import os.path as path
import pandas as pd
import glob
from subprocess import Popen, PIPE
from sklearn.mixture import GMM
from time import time, sleep
import joblib

@contextmanager
def verb_print(msg, verbose=False):
    """Helper for verbose printing with timing around pieces of code.
    """
    if verbose:
        t0 = time()
        msg = msg + '...'
        print msg,
        sys.stdout.flush()
    try:
        yield
    finally:
        if verbose:
            print 'done. time: {0:.3f}s'.format(time() - t0)
            sys.stdout.flush()

VERBOSE = True

datadir = path.join(os.environ['HOME'], 'data')
zerospeechdir = path.join(datadir, 'zerospeech')
english_vtln_dir = path.join(zerospeechdir, 'english_vtln')
npy_dir = path.join(english_vtln_dir, 'vtln2')
out_dir = path.join(english_vtln_dir, 'vtln_gmm')
try:
    os.makedirs(out_dir)
except OSError:
    pass

FRATE = 100
intervals = pd.read_csv(path.join(zerospeechdir, 'english.split'))
intervals = {
    name: zip(group.start.values, group.end.values)
    for name, group in intervals.groupby(intervals.f_id)
}


def load_frames():
    cache_file = 'cache/vtln_cache'
    if not path.exists(cache_file):
        r = {}
        for npy_file in glob.iglob(path.join(npy_dir, '*.npy')):
            bname = path.splitext(path.basename(npy_file))[0]
            r[bname] = np.load(npy_file)
        joblib.dump(r, cache_file, compress=0)
    else:
        r = joblib.load(cache_file)
    return r


if __name__ == '__main__':
    with verb_print('loading vtln frames', VERBOSE):
        frames_per_bname = load_frames()

    with verb_print('cutting out intervals', VERBOSE):
        frames = []
        for bname, mfc in frames_per_bname.iteritems():
            for start, end in intervals[bname]:
                frames.append(mfc[start * FRATE: end * FRATE])
        frames = np.vstack(frames)

    with verb_print('fitting {} frames'.format(frames.shape[0]), VERBOSE):
        gmm = GMM(n_components=400, covariance_type='full', n_init=10)
        gmm.fit(frames)

    with verb_print('saving model to gmm.joblib.pkl', VERBOSE):
        joblib.dump(gmm, 'gmm.joblib.pkl')

    with verb_print('computing posteriors', VERBOSE):
        posteriors = {}
        for bname, mfc in frames_per_bname.iteritems():
            posteriors[bname] = gmm.predict_proba(mfc)

    with verb_print('saving posteriors', VERBOSE):
        for bname, post in posteriors.iteritems():
            outfile = path.join(out_dir, bname + '.npy')
            np.save(outfile, post)

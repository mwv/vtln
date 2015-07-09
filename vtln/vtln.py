#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: vtln.py
# date: Fri July 03 14:18 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""vtln:

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

import htkmfc

VERBOSE = True


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

datadir = path.join(os.environ['HOME'], 'data')
zerospeechdir = path.join(datadir, 'zerospeech')
english_wavdir = path.join(zerospeechdir, 'english_wav')
english_files = sorted(glob.glob(path.join(english_wavdir, '*.wav')))
english_vtln_dir = path.join(zerospeechdir, 'english_vtln')

config_dir = path.join(zerospeechdir, 'htk_configs')
bnames = [path.splitext(path.basename(wavfile))[0]
          for wavfile in english_files]

def bname2speaker(bname):
    return bname[:3]

speakers = sorted(set([bname2speaker(bname)
                       for bname in bnames]))

bnames_per_speaker = {
    speaker: [bname for bname in bnames if bname2speaker(bname) == speaker]
    for speaker in speakers
}


with verb_print('writing config files', VERBOSE):
    try:
        os.makedirs(config_dir)
    except OSError:
        pass
    msg = """SOURCEFORMAT = WAV
TARGETKIND = MFCC_0_D_A_Z
TARGETRATE = 100000.0
SAVECOMPRESSED = T
SAVEWITHCRC = T
WINDOWSIZE = 250000.0
USEHAMMING = T
PREEMCOEF = 0.97
NUMCHANS = 26
CEPLIFTER = 12
NUMCEPS = 12
ENORMALIZE = T
WARPLCUTOFF = 350
WARPUCUTOFF = 3350
WARPFREQ = {:.2f}"""

    warpfreqs = np.arange(0.8, 1.21, 0.01)
    for warpfreq in warpfreqs:
        with open(path.join(config_dir, 'htk_config_{:.2f}'.format(warpfreq)),
                  'w') as fout:
            fout.write(msg.format(warpfreq))


REBUILD = False
if REBUILD or not path.exists(english_vtln_dir):
    with verb_print('extracting mfccs', VERBOSE):
        # async pump everything through HCopy
        for warpfreq in warpfreqs:
            subdir = path.join(
                english_vtln_dir,
                'warp_freq_{:.2f}'.format(warpfreq)
            )
            config_file = path.join(
                config_dir,
                'htk_config_{:.2f}'.format(warpfreq)
            )
            print warpfreq
            try:
                os.makedirs(subdir)
            except OSError:
                pass
            running_procs = [
                Popen(['HCopy', '-C', config_file, wavfile,
                       path.join(
                           subdir,
                           path.splitext(path.basename(wavfile))[0]) + '.mfc'],
                      stdout=PIPE, stderr=PIPE)
                for wavfile in english_files]
            while running_procs:
                for proc in running_procs:
                    retcode = proc.poll()

                    if retcode is not None:
                        proc.stdout.close()

                        running_procs.remove(proc)
                        break
                    else:
                        sleep(.1)
                        continue

FRATE = 100
intervals = pd.read_csv(path.join(zerospeechdir, 'english.split'))
intervals = {
    name: zip(group.start.values, group.end.values)
    for name, group in intervals.groupby(intervals.f_id)
}


_cache = {}
def get_frames(bname, warpfreq):
    """Return concatenated vad frames from bname
    """
    key = (bname, warpfreq)
    if key not in _cache:
        r = []
        for start, end in intervals[bname]:
            mfcfile = path.join(
                english_vtln_dir,
                'warp_freq_{:.2f}'.format(warpfreq),
                bname + '.mfc')
            mfc = htkmfc.open(mfcfile).getall()
            start_fr = start * FRATE
            end_fr = end * FRATE
            r.append(mfc[start_fr:end_fr])
        _cache[key] = np.vstack(r)
    return _cache[key]


with verb_print('loading cache', VERBOSE):
    if not path.exists('cache'):
        for bname in bnames:
            print bname
            for warpfreq in warpfreqs:
                print ' ', warpfreq
                get_frames(bname, warpfreq)
        with verb_print('dumping cache', VERBOSE):
            joblib.dump(_cache, 'cache', compress=0)
    else:
        _cache = joblib.load('cache')

alphas = {speaker: 1.0 for speaker in speakers}
alphas_prev = alphas.copy()
max_iter = 10
it = 1
while True:
    print 'iteration:', it

    with verb_print('stacking frames', VERBOSE):
        frames = np.vstack(
            (get_frames(bname, alphas[speaker])
             for speaker in speakers
             for bname in bnames_per_speaker[speaker])
        )
    with verb_print('fitting gmm', VERBOSE):
        gmm = GMM(
            n_components=50,
            covariance_type='diag'
        ).fit(frames)

    alphas_prev = alphas.copy()
    with verb_print('calculating log-likelihoods', VERBOSE):
        for speaker in speakers:
            scores = []
            for warpfreq in warpfreqs:
                frames = np.vstack(
                    (get_frames(bname, warpfreq))
                    for bname in bnames_per_speaker[speaker]
                )
                score = np.exp(gmm.score(frames)).mean()
                scores.append((warpfreq, score))
            best_alpha = max(scores, key=lambda x: x[1])[0]
            alphas[speaker] = best_alpha

    with open('ideal_warpfreq_iteration_{}.txt'.format(it), 'w') as fout:
        fout.write('speaker,filename,warpfreq\n')
        for speaker in sorted(speakers):
            for filename in bnames_per_speaker[speaker]:
                fout.write('{},{},{}\n'.format(
                    speaker, filename, alphas[speaker])
                )

    if alphas == alphas_prev:
        break
    if it >= max_iter:
        break

    it += 1

with open('ideal_warpfreq_final.txt', 'w') as fout:
    fout.write('speaker,filename,warpfreq\n')
    for speaker in sorted(speakers):
        for filename in bnames_per_speaker[speaker]:
            fout.write('{},{},{}\n'.format(speaker, filename, alphas[speaker]))

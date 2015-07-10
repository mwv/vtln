#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: arrange_mfcc_vtln.py
# date: Tue July 07 18:09 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""arrange_mfcc_vtln:

"""

from __future__ import division

import os
import os.path as path
import shutil
import glob

import numpy as np
import pandas as pd

import htkmfc


datadir = path.join(os.environ['HOME'], 'data')
zerospeechdir = path.join(datadir, 'zerospeech')
english_wavdir = path.join(zerospeechdir, 'english_wav')
english_files = sorted(glob.glob(path.join(english_wavdir, '*.wav')))
english_vtln_dir = path.join(zerospeechdir, 'english_vtln')

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

if __name__ == '__main__':
    ideal_warpfreq_file = 'ideal_warpfreq_5.txt'
    ideal_warps = pd.read_csv(ideal_warpfreq_file)

    # outrawdir = path.join(datadir, 'raw')
    outwarpeddir = path.join(english_vtln_dir, 'vtln2')
    try:
        os.makedirs(outwarpeddir)
    except OSError:
        pass

    for ix, (_, filename, warpfreq) in ideal_warps.iterrows():
        # print '{} ({}/{})'.format(speaker, ix+1, len(ideal_warps))
        # for filename in bnames_per_speaker[speaker]:
        print '{} ({}/{})'.format(filename, ix+1, len(ideal_warps))

        infile_warp = path.join(
            english_vtln_dir, 'warp_freq_{:.2f}'.format(warpfreq),
            filename+'.mfc'
        )
        mfc_warp = htkmfc.open(infile_warp).getall()
        np.save(path.join(outwarpeddir, filename+'.npy'), mfc_warp)

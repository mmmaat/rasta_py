#!/usr/bin/env python
# Use it as:
# ./replicate.py ../shennong/test/data/test.wav ../features_extraction/rastamat

import argparse
import os
import rasta
import scipy.io
import tempfile
import matplotlib.pyplot as plt


def pprint(feats):
    print(type(feats), feats.shape, feats.dtype, feats.min(), feats.max())


def do_octave(wav, rastamat):
    with tempfile.TemporaryDirectory() as tmp:
        script_file = os.path.join(tmp, 'script.m')
        feats_file = os.path.join(tmp, 'feats.mat')

        script = f"""
        pkg load signal;
        addpath('{rastamat}');
        wav_file = '{wav}';
        [data, sample_rate] = audioread(wav_file);
        feats = rastaplp(data, sample_rate);
        save('-mat', '{feats_file}', 'feats');
        """
        open(script_file, 'w').write(script)

        os.system('octave {}'.format(script_file))
        feats = scipy.io.loadmat(feats_file)['feats']

        pprint(feats)
        return feats


def do_python(wav):
    sample_rate, data = scipy.io.wavfile.read(wav)
    data = data / 2**15
    feats = rasta.rastaplp(data, fs=sample_rate, win_time=0.025, hop_time=0.01)
    pprint(feats)
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('wav_file')
    parser.add_argument('rastamat', help='path to rastamat package')
    args = parser.parse_args()

    fpy = do_python(args.wav_file)
    foc = do_octave(args.wav_file, args.rastamat)

    # plot the audio signal and the resulting features
    fig, axes = plt.subplots(nrows=2)
    axes[0].imshow(fpy, aspect='auto')
    axes[0].set_title('python')
    axes[1].imshow(foc, aspect='auto')
    axes[1].set_title('matlab')
    plt.show()


if __name__ == '__main__':
    main()

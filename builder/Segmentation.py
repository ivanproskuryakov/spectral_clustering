import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

from config import BINS_PER_OCTAVE


##############################################
# Next, we'll compute and plot a log-power CQT
class Segmentation:
    def build(self, k, C, sr, seg_ids, beats, colors, track_dir: str):
        ###############################################################
        # Locate segment boundaries from the label sequence
        bound_beats = np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

        # Count beat 0 as a boundary
        bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

        # Compute the segment label for each boundary
        bound_segs = list(seg_ids[bound_beats])

        # Convert beat indices to frames
        bound_frames = beats[bound_beats]

        # Make sure we cover to the end of the track
        bound_frames = librosa.util.fix_frames(bound_frames,
                                               x_min=None,
                                               x_max=C.shape[1] - 1)

        ###################################################
        # And plot the final segmentation over original CQT

        # sphinx_gallery_thumbnail_number = 5

        plt.figure(figsize=(12, 4))

        bound_times = librosa.frames_to_time(bound_frames)
        freqs = librosa.cqt_frequencies(n_bins=C.shape[0],
                                        fmin=librosa.note_to_hz('C1'),
                                        bins_per_octave=BINS_PER_OCTAVE)

        librosa.display.specshow(C, y_axis='cqt_hz', sr=sr,
                                 bins_per_octave=BINS_PER_OCTAVE,
                                 x_axis='time')
        ax = plt.gca()

        for interval, label in zip(zip(bound_times, bound_times[1:]), bound_segs):
            ax.add_patch(patches.Rectangle((interval[0], freqs[0]),
                                           interval[1] - interval[0],
                                           freqs[-1],
                                           facecolor=colors(label),
                                           alpha=0.50))

        ax.xaxis.set_major_locator(MultipleLocator(30))
        ax.xaxis.set_minor_locator(AutoMinorLocator(6))

        plt.tight_layout()
        plt.savefig('{track_dir}/segmentation_{k}.png'.format(track_dir=track_dir, k=k))
        plt.close()

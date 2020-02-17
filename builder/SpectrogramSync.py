import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


##########################################################
# To reduce dimensionality, we'll beat-synchronous the CQT
class SpectrogramSync:
    def __init__(self, bins_per_octave: int):
        self._bins_per_octave = bins_per_octave

    def build(self, y, sr, C, track_dir: str):
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        C_sync = librosa.util.sync(C, beats, aggregate=np.median)

        # For plotting purposes, we'll need the timing of the beats
        # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
        beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats, x_min=0, x_max=C.shape[1]),
                                            sr=sr)

        plt.figure(figsize=(12, 4))
        ax = plt.subplot(1, 1, 1)

        librosa.display.specshow(C_sync,
                                 bins_per_octave=self._bins_per_octave,
                                 y_axis='cqt_hz',
                                 x_axis='time',
                                 x_coords=beat_times)
        plt.tight_layout()

        ax.xaxis.set_major_locator(MultipleLocator(30))
        ax.xaxis.set_minor_locator(AutoMinorLocator(6))

        plt.savefig('{track_dir}/spectrogram_sync.png'.format(track_dir=track_dir))
        plt.close()

        return C_sync, beats, beat_times

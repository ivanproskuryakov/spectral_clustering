import librosa.display
import numpy as np
import matplotlib.pyplot as plt


##############################################
# Next, we'll compute and plot a log-power CQT
class Spectrogram:
    def __init__(self, bins_per_octave: int, octaves: int):
        self._bins_per_octave = bins_per_octave
        self._octaves = octaves

    def build(self, y, sr, track_dir: str):
        c = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y,
                                                       sr=sr,
                                                       bins_per_octave=self._bins_per_octave,
                                                       n_bins=self._octaves * self._bins_per_octave)),
                                    ref=np.max)

        plt.figure(figsize=(12, 4))
        librosa.display.specshow(c,
                                 y_axis='cqt_hz',
                                 sr=sr,
                                 bins_per_octave=self._bins_per_octave,
                                 x_axis='time')

        plt.tight_layout()
        plt.savefig('{track_dir}/spectrogram.png'.format(track_dir=track_dir))

        return c

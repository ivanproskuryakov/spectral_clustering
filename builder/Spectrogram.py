import librosa.display
import numpy as np

##############################################
# Next, we'll compute and plot a log-power CQT
class Spectrogram:
    def __init__(self, bins_per_octave: int, octaves: int):
        self._bins_per_octave = bins_per_octave
        self._octaves = octaves

    def build(self, y, sr):
        C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y,
                                                       sr=sr,
                                                       bins_per_octave=self._bins_per_octave,
                                                       n_bins=self._octaves * self._bins_per_octave)),
                                    ref=np.max)
        return C

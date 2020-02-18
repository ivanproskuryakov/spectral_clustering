import librosa.display
import numpy as np


##########################################################
# To reduce dimensionality, we'll beat-synchronous the CQT
class SpectrogramSync:
    def __init__(self, bins_per_octave: int):
        self._bins_per_octave = bins_per_octave

    def build(self, y, sr, C):
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        C_sync = librosa.util.sync(C, beats, aggregate=np.median)

        # For plotting purposes, we'll need the timing of the beats
        # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
        beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats, x_min=0, x_max=C.shape[1]),
                                            sr=sr)
        return C_sync, beats, beat_times

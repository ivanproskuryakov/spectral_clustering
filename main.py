import librosa

from TracksScanner import TracksScanner
from builder.Spectrogram import Spectrogram
from builder.SpectrogramSync import SpectrogramSync
from builder.Matrix import Matrix
from builder.Laplacian import Laplacian
from builder.Segmentation import Segmentation

from config import ALBUM_DIR, N_OCTAVES, BINS_PER_OCTAVE

# Init
spectrogram = Spectrogram(bins_per_octave=BINS_PER_OCTAVE, octaves=N_OCTAVES)
spectrogramSync = SpectrogramSync(bins_per_octave=BINS_PER_OCTAVE)
matrix = Matrix()
laplacian = Laplacian()
segmentation = Segmentation()

# Load
scanner = TracksScanner(album_dir=ALBUM_DIR)
scanner.scan_tracks()
scanner.make_output_dirs()
track = scanner.tracks[0]
y, sr = librosa.load(track['file'])

# Build
c = spectrogram.build(y, sr, track_dir=track['dir'])
C_sync, beats, beat_times = spectrogramSync.build(y, sr, c, track_dir=track['dir'])
A, Rf = matrix.build(y, sr, C_sync, beats, beat_times, track_dir=track['dir'])
seg_ids, colors = laplacian.build(A, Rf, beat_times, track_dir=track['dir'])
segmentation.build(c, sr, seg_ids, beats, colors, track_dir=track['dir'])

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
track = scanner.tracks[1]
clusters = [4, 8, 12, 16, 20, 24]

# Build
for track in scanner.tracks:
    y, sr = librosa.load(track['file'])

    C = spectrogram.build(y, sr)
    C_sync, beats, beat_times = spectrogramSync.build(y, sr, C)
    A = matrix.build(y, sr, C_sync, beats)

    for k in clusters:
        seg_ids, colors = laplacian.build(k, A, beat_times, track_dir=track['dir'])
        segmentation.build(k, C, sr, seg_ids, beats, colors, track_dir=track['dir'])

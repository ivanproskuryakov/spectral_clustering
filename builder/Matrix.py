import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy


class Matrix:
    def build(self, y, sr, c_sync, beats, beat_times, track_dir: str):
        #####################################################################
        # Let's build a weighted recurrence matrix using beat-synchronous CQT
        # (Equation 1)
        # width=3 prevents links within the same bar
        # mode='affinity' here implements S_rep (after Eq. 8)
        R = librosa.segment.recurrence_matrix(c_sync,
                                              width=3,
                                              mode='affinity',
                                              sym=True)

        # Enhance diagonals with a median filter (Equation 2)
        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        Rf = df(R, size=(1, 7))

        ###################################################################
        # Now let's build the sequence matrix (S_loc) using mfcc-similarity
        #
        #   :math:`R_\text{path}[i, i\pm 1] = \exp(-\|C_i - C_{i\pm 1}\|^2 / \sigma^2)`
        #
        # Here, we take :math:`\sigma` to be the median distance between successive beats.
        #
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        Msync = librosa.util.sync(mfcc, beats)

        path_distance = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
        sigma = np.median(path_distance)
        path_sim = np.exp(-path_distance / sigma)

        R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

        ##########################################################
        # And compute the balanced combination (Equations 6, 7, 9)

        deg_path = np.sum(R_path, axis=1)
        deg_rec = np.sum(Rf, axis=1)

        mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec) ** 2)

        A = mu * Rf + (1 - mu) * R_path

        ###########################################################
        # Plot the resulting graphs (Figure 1, left and center)
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 3, 1)
        librosa.display.specshow(Rf, cmap='inferno_r', y_axis='time',
                                 y_coords=beat_times)
        plt.title('Recurrence similarity')

        plt.subplot(1, 3, 2)
        librosa.display.specshow(R_path, cmap='inferno_r')
        plt.title('Path similarity')

        plt.subplot(1, 3, 3)
        librosa.display.specshow(A, cmap='inferno_r')
        plt.title('Combined graph')

        plt.tight_layout()
        plt.savefig('{track_dir}/matrix.png'.format(track_dir=track_dir))
        plt.close()

        return A, Rf

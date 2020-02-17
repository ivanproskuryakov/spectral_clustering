import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn.cluster
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


class Laplacian:
    def build(self, k, A, Rf, beat_times, track_dir: str):
        #####################################################
        # Now let's compute the normalized Laplacian (Eq. 10)
        L = scipy.sparse.csgraph.laplacian(A, normed=True)

        # and its spectral decomposition
        evals, evecs = scipy.linalg.eigh(L)

        # We can clean this up further with a median filter.
        # This can help smooth over small discontinuities
        evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

        # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
        Cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5

        # If we want k clusters, use the first k normalized eigenvectors.
        # Fun exercise: see how the segmentation changes as you vary k
        X = evecs[:, :k] / Cnorm[:, k - 1:k]

        # Plot the resulting representation (Figure 1, center and right)
        KM = sklearn.cluster.KMeans(n_clusters=k)

        seg_ids = KM.fit_predict(X)

        # and plot the results
        plt.figure(figsize=(12, 4))
        colors = plt.get_cmap('Paired', k)

        ax = plt.subplot(1, 3, 2)
        librosa.display.specshow(Rf, cmap='inferno_r')
        plt.title('Recurrence matrix')

        ax.yaxis.set_major_locator(MultipleLocator(30))
        ax.yaxis.set_minor_locator(AutoMinorLocator(6))

        ax = plt.subplot(1, 3, 1)
        librosa.display.specshow(X,
                                 y_axis='time',
                                 y_coords=beat_times)
        plt.title('Structure components')

        ax.yaxis.set_major_locator(MultipleLocator(30))
        ax.yaxis.set_minor_locator(AutoMinorLocator(6))

        plt.subplot(1, 3, 3)
        librosa.display.specshow(np.atleast_2d(seg_ids).T, cmap=colors)
        plt.title('Estimated segments')
        plt.colorbar(ticks=range(k))

        plt.tight_layout()
        plt.savefig('{track_dir}/laplacian_{k}.png'.format(track_dir=track_dir, k=k))
        plt.close()

        return seg_ids, colors

import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn.cluster
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib import gridspec

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

        # Detailed results
        plt.figure(figsize=(10, 4))
        gs = gridspec.GridSpec(2, 1, height_ratios=[8, 2])

        ax = plt.subplot(gs[0])
        librosa.display.specshow(X.transpose(),
                                 x_axis='time',
                                 x_coords=beat_times)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        plt.title('Structure components')

        plt.subplot(gs[1])
        colors = plt.get_cmap('Paired', k)
        librosa.display.specshow(np.atleast_2d(seg_ids).T.transpose(), cmap=colors)
        plt.title('Estimated segments')

        plt.tight_layout()
        plt.savefig('{track_dir}/laplacian_{k}_structure.png'.format(track_dir=track_dir, k=k))
        plt.close()

        return seg_ids, colors

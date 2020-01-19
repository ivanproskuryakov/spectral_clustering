import os


class TracksScanner:
    def __init__(self, album_dir: str):
        self.album_dir = album_dir
        self.tracks = []

    def scan_tracks(self):
        for f in os.listdir(self.album_dir):
            if f.endswith(".mp3"):
                t = f.strip('.mp3')
                track = {
                    'dir': '{album_dir}/{t}'.format(album_dir=self.album_dir, t=t),
                    'file': '{album_dir}/{t}.mp3'.format(album_dir=self.album_dir, t=t),
                }
                self.tracks.append(track)

    def make_output_dirs(self):
        for t in self.tracks:
            if not os.path.exists(t['dir']):
                os.makedirs(t['dir'])

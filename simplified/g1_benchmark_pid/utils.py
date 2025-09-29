import numpy as np
from numba import njit


class VizColor:
    goal = [48/255, 245/255, 93/255, 0.3]
    collision_volume = [0.1, 0.1, 0.1, 0.7]
    collision_volume_ignored = [0.1, 0.1, 0.1, 0.0]
    obstacle_debug = [0.5, 0.5, 0.5, 0.7]
    obstacle_task = [0.5, 0.5, 0.5, 0.7]

    safe = [0, 1, 0, 0.5]
    hold = [245/255, 243/255, 48/255, 0.5]
    unsafe = [255/255, 84/255, 48/255, 1.0]

    slack_positive = [255/255, 84/255, 48/255, 0.5]


class Geometry:
    def __init__(self, type, **kwargs):
        self.type = type
        self.attributes = {}
        self.color = kwargs.get("color", np.array([1, 1, 1, 0.5]))

        required_attr = []
        if self.type == 'sphere':
            required_attr = ['radius']
        elif self.type == 'box':
            required_attr = ['length', 'width', 'height']
        else:
            raise ValueError(f'Unknown geometry type: {self.type}')

        for attr in required_attr:
            if attr in kwargs:
                self.attributes[attr] = kwargs[attr]
            else:
                raise ValueError(
                    f'Missing required attribute: {attr} for geometry type: {self.type}'
                )

    def get_attributes(self):
        return self.attributes


def compute_pairwise_dist(
    frame_list_1, geom_list_1, frame_list_2, geom_list_2
):
    radii_1 = np.array(
        [geom.attributes['radius'] for geom in geom_list_1]
    )
    radii_2 = np.array(
        [geom.attributes['radius'] for geom in geom_list_2]
    )
    frames_1 = np.array(frame_list_1)
    frames_2 = np.array(frame_list_2)
    return _compute_distances(frames_1, radii_1, frames_2, radii_2)

@njit
def _compute_distances(
    frames_1, radii_1, frames_2, radii_2
):
    N1 = frames_1.shape[0]
    N2 = frames_2.shape[0]
    dist = np.zeros((N1, N2))
    for i in range(N1):
        p1 = frames_1[i, :3, 3]
        for j in range(N2):
            p2 = frames_2[j, :3, 3]
            dist[i, j] = np.linalg.norm(p1 - p2) - radii_1[i] - radii_2[j]
    return dist


class Logger:
    def __init__(
        self, log_dir, n_logged_samples=10, 
        summary_writer=None, counter=0
    ):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._n_logged_samples = n_logged_samples
        self.counter = counter
        try:
            from tensorboardX import SummaryWriter
            self._summ_writer = SummaryWriter(
                log_dir, flush_secs=1, max_queue=1
            )
            self._use_tb = True
        except Exception:
            self._summ_writer = None
            self._use_tb = False

    def log_scalar(self, scalar, name):
        if self._use_tb:
            self._summ_writer.add_scalar(
                '{}'.format(name), scalar, self.counter
            )

    def flush(self):
        if self._use_tb:
            self._summ_writer.flush()
        self.counter += 1

__all__ = [
    "Geometry",
    "VizColor",
    "compute_pairwise_dist",
    "Logger",
]



from typing import Any
import numpy as np
import os
import json
import h5py
from particle import Particle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# TODO:
# - Support for multiprocessing alonside the multitrheading that we now have.
# - Custom distrortions/post-processing of the rings.

# ------------------------------------------------------------------
# helpers for threaded dataset generation
_worker_gen = None
def _init_worker(state: dict[str, Any]):
    global _worker_gen
    _worker_gen = SCGen(**state)
def _produce_batch(n_list: list[int]):
    out = []
    for n in n_list:
        out.append(_worker_gen.generate_event(n))
        if _pbar:
            _pbar.update(1)
    return out
# ------------------------------------------------------------------

class UniformDist:
    """
    A simple, dummy class for uniform distribution, providing a resample method.
    """
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def resample(self, size: int) -> np.ndarray:
        return np.random.uniform(self.min_val, self.max_val, size)

class SCGen:
    """
    Synthetic LHCb's RICH detector data generator.
    This class generates synthetic data for the LHCb's RICH detector, simulating Cherenkov rings.
    It utlizes distributions from realistc data, generated with full-Monte Carlo simulation.

    Attributes
    ----------

    """
    def __init__(self,
                 particle_types: list[int],
                 refractive_index: float,
                 detector_size: tuple[tuple[float, float], tuple[float, float]],
                 momenta_log_distributions: dict[str, Any],
                 centers_distribution: Any,
                 radial_noise,
                 N_init: int,
                 max_radius: float = 100.0,
                 masses: dict[int, float] = None
                 ):
        """
        Initialize the SCGen instance with the given parameters.

        Parameters
        ----------
        particle_types : list[int]
            List of particle types to be generated. It should contain PDG IDs of the particles.
        refractive_index : float
            The refractive index of the RICH detector medium.
        detector_size : tuple[tuple[float, float], tuple[float, float]]
            The sizes of the detector in ((x_min, x_max), (y_min, y_max)) format.
        momenta_log_distributions : dict[str, Any]
            The momentum log distributions for the different particle types.
            It should be a dictionary where keys are particle types and values are distributions (scipy KDEs or TruncatedKDEs).
            We use **log** of the momentum to avoid numerical issues with KDEs.
            In alternative, one can provide a tuple of (min_momentum, max_momentum) for uniform distribution; note that in this case
            the distribution will be uniform in linear scale, not logarithmic.
        centers_distribution : Any
            The distribution of the centers of the Cherenkov rings.
            It should be a distribution object (scipy KDE or TruncatedKDE)
        radial_noise : tuple or float
            The radial noise to be applied to the generated rings.
            If a single float is provided, it is used as standard deviation for Gaussian noise.
            If a tuple is provided, it should be in the form (mean, std_dev) for Gaussian noise.
        N_init : int
            The initial value used to compute the mean of the possonian distribution for the number of photon hits.
        max_radius : float, optional
            The maximum radius of the Cherenkov rings. Default is 100.0.
        masses : dict[int, float], optional
            A dictionary mapping particle types to their masses in GeV/c^2.
            If not provided, the generator will try to initialize masses based on the particle types using the Particle library.
        """
        self.particle_types = particle_types
        self.refractive_index = refractive_index
        self.detector_size = detector_size
        self.centers_distribution = centers_distribution
        if isinstance(radial_noise, float):
            self.radial_noise = (0, radial_noise)
        else:
            self.radial_noise = radial_noise

        for p, momenta_dist in momenta_log_distributions.items():
            if isinstance(momenta_dist, tuple):
                # If a tuple is provided, create a uniform distribution
                momenta_log_distributions[p] = UniformDist(*momenta_dist)

        self.momenta_log_distributions = momenta_log_distributions
        self.N_init = N_init

        self.max_cherenkov_radius = max_radius
        self.max_cherenkov_angle = np.arccos(1 / refractive_index)

        # Initialize masses for the particle types
        if masses is None:
            self.masses = {pt: Particle.from_pdgid(pt).mass for pt in particle_types}
        else:
            self.masses = masses

        # Initialize the dataset as an empty list
        self.dataset = []

    def _create_rings(self, centers: np.ndarray, radii: np.ndarray, N_hits: np.ndarray) -> list[np.ndarray]:
        """
        Create Cherenkov rings based on the centers, radii, and number of hits.

        Parameters
        ----------
        centers : np.ndarray
            The centers of the Cherenkov rings.
        radii : np.ndarray
            The radii of the Cherenkov rings.
        N_hits : np.ndarray
            The number of hits for each ring.

        Returns
        -------
        list[np.ndarray]
            A list of arrays, each containing the (x, y) coordinates of the hits for each ring.
        """

        rings = np.empty(len(centers), dtype=object)
        for i in range(len(centers)):
            if N_hits[i] > 0:
                # Sample radial noise
                radial_noise = np.random.normal(self.radial_noise[0], self.radial_noise[1], N_hits[i])
                # Sample angles uniformly around the ring
                angles = np.random.uniform(0, 2 * np.pi, N_hits[i])
                # Compute the x and y coordinates of the hits
                x = centers[i][0] + (radii[i] + radial_noise) * np.cos(angles)
                y = centers[i][1] + (radii[i] + radial_noise) * np.sin(angles)
                rings[i] = np.column_stack((x, y))
            else:
                rings[i] = np.empty((0, 2))
        return rings

    def generate_event(self, num_particles: int) -> dict[str, Any]:
        """
        Generate a single event with the specified number of particles.

        Parameters
        ----------
        num_particles : int
            The number of particles to generate in the event.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the generated event data:
            - 'centers': The centers of the Cherenkov rings.
            - 'momenta': The momenta of the particles.
            - 'particle_types': The types of the particles.
            - 'rings': The generated Cherenkov rings. If a particle has no hits, its ring will be an empty array.
            - 'tracked_rings': mapping of (center, momentum, particle_type) to the corresponding ring index in the rings array.
            - 'radii': The Cherenkov radii for each particle.
        """
        # 1. Sample generation parameters
        ptypes = np.random.choice(self.particle_types, size=num_particles)
        centers = self.centers_distribution.resample(num_particles).T
        momenta = np.zeros(num_particles)

        # 1b. Sample valid momenta (beta above Cherenkov threshold)
        for i, pt in enumerate(ptypes):
            valid = False
            while not valid:
                p = np.exp(self.momenta_log_distributions[pt].resample(1)[0])
                m = self.masses[pt]
                beta = p / np.sqrt(p**2 + m**2)
                arccos_arg = 1 / (self.refractive_index * beta)
                if arccos_arg <= 1:  # particle above threshold
                    valid = True
                    momenta[i] = p

        # 2. Compute Cherenkov angles and radii
        masses = np.array([self.masses[pt] for pt in ptypes])
        beta = momenta / np.sqrt(np.square(momenta) + np.square(masses))
        arccos_arg = 1 / (self.refractive_index * beta)
        cherenkov_angles = np.arccos(arccos_arg)  # all valid now
        radii = np.tan(cherenkov_angles) * self.max_cherenkov_radius / np.tan(self.max_cherenkov_angle)

        # 3. Compute the number of photon hits for each particle
        N_mean = np.round(
            self.N_init * np.square(np.sin(cherenkov_angles)) / np.square(np.sin(self.max_cherenkov_angle))
        ).astype(int)
        N_hits = np.random.poisson(N_mean)

        # 4. Generate the Cherenkov rings
        rings = self._create_rings(centers, radii, N_hits)

        # 5. Track mapping
        tracked_rings = np.arange(num_particles)  # each particle corresponds to a ring

        return {
            'centers': centers,
            'momenta': momenta,
            'particle_types': ptypes,
            'rings': rings,
            'tracked_rings': tracked_rings,
            'radii': radii
        }


    def generate_dataset(self, num_events: int, num_particles_per_event, parallel = False, batch_size: int = 64, progress_bar: bool = True) -> list[dict[str, Any]]:
        """
        Generate a dataset of synthetic events.

        Parameters
        ----------
        num_events : int
            The number of events to generate.
        num_particles_per_event : int or tuple
            The number of particles in each event. If an integer is provided, all events will have that number of particles.
            If a tuple is provided, it should be in the form (min_particles, max_particles), and the number of particles
            in each event will be sampled uniformly from that range.
        parallel : bool or int, optional
            If True, the dataset generation will be parallelized using multiprocessing.
            If an integer is provided, it specifies the number of processes to use for parallelization.
            If False, the dataset will be generated sequentially. Default is False.
        batch_size : int, optional
            The number of events to generate in each batch when using parallel generation. Default is 64.
            If the total number of events is not divisible by the batch size, the last batch will contain the remaining events.
            If the total number of events is less than the batch size, it will generate all events in one batch.
        progress_bar : bool, optional
            If True, a progress bar will be displayed during dataset generation. Default is True.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries, each containing the data for one event.
        """
        if isinstance(num_particles_per_event, tuple):
            counts = np.random.randint(
                num_particles_per_event[0],
                num_particles_per_event[1] + 1,
                size=num_events,
            ).tolist()
        else:
            counts = [num_particles_per_event] * num_events
        if not parallel:
            bar = tqdm(total=num_events, disable=not progress_bar)
            data = [self.generate_event(n) for n in counts if not bar.update() and True]
            bar.close()
            return data

        n_workers = os.cpu_count() if parallel is True else int(parallel)
        n_workers = max(1, n_workers)

        state = dict(
            particle_types=self.particle_types,
            refractive_index=self.refractive_index,
            detector_size=self.detector_size,
            momenta_log_distributions=self.momenta_log_distributions,
            centers_distribution=self.centers_distribution,
            radial_noise=self.radial_noise,
            N_init=self.N_init,
            max_radius=self.max_cherenkov_radius,
            masses=self.masses,
        )

        global _pbar
        _pbar = tqdm(total=num_events, disable=not progress_bar)

        batches = [counts[i:i + batch_size] for i in range(0, len(counts), batch_size)]
        dataset: list[dict[str, Any]] = []
        with ThreadPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(state,),
        ) as pool:
            futs = [pool.submit(_produce_batch, b) for b in batches]
            for fut in as_completed(futs):
                dataset.extend(fut.result())

        _pbar.close()
        _pbar = None
        self.dataset = dataset  # Store the generated dataset in the instance for later use
        return dataset
    
    def prune_centers(self, keep_probability: float) -> None:
        """
        Prune the centers of the rings based on a keep_probability.

        Parameters
        ----------
        keep_probability : float
            The probability of keeping each center. Should be between 0 and 1.

        Raises
        ------
        ValueError
            If keep_probability is not between 0 and 1.

        Returns
        -------
        list[dict[str, Any]]
            The dataset with pruned centers.
        """
        if not (0 <= keep_probability <= 1):
            raise ValueError("keep_probability must be between 0 and 1.")
        
        for event in self.dataset:
            centers = event['centers']
            mask = np.random.rand(len(centers)) < keep_probability
            event['centers'] = centers[mask]
            event['momenta'] = event['momenta'][mask]
            event['particle_types'] = event['particle_types'][mask]
            event['tracked_rings'] = event['tracked_rings'][mask]

        return self.dataset
    
    def save_dataset(self, filename: str):
        """
        Save the generated dataset to an HDF5 file.

        Parameters
        ----------
        filename : str
            The path to the HDF5 file where the dataset will be saved.
        """
        if not filename.endswith('.h5'):
            filename += '.h5'

        # Define dtype for efficient storage of ring coordinates
        xy_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        vlen_xy = h5py.vlen_dtype(xy_dtype)

        with h5py.File(filename, 'w', libver='latest') as f:
            # Add metadata about the generator
            f.attrs.update({
                'refractive_index': self.refractive_index,
                'detector_size': np.asarray(self.detector_size, np.float32),
                'radial_noise': np.asarray(self.radial_noise, np.float32),
                'N_init': self.N_init,
                'max_cherenkov_radius': self.max_cherenkov_radius,
                'particle_types': np.asarray(self.particle_types, np.int32),
                'masses': json.dumps(self.masses),  # Store as JSON for better compatibility
                'num_events': len(self.dataset),
            })

            # Create a group for each event
            for i, event in enumerate(self.dataset):
                grp = f.create_group(f'event_{i}')
                grp.attrs['num_particles'] = len(event['rings'])
                
                # Save the event data with appropriate data types
                grp.create_dataset('centers', data=event['centers'].astype('float32'))
                grp.create_dataset('momenta', data=event['momenta'].astype('float32'))
                grp.create_dataset('particle_types', data=event['particle_types'].astype('int32'))
                grp.create_dataset('tracked_rings', data=event['tracked_rings'].astype('int32'))
                grp.create_dataset('radii', data=event['radii'].astype('float32'))
                
                # Convert each ring to a 1-D recarray of (x,y) pairs for efficient storage
                rings = [np.ascontiguousarray(list(map(tuple, ring)), dtype=xy_dtype)
                         for ring in event['rings']]
                grp.create_dataset('rings',
                                   shape=(len(rings),),
                                   dtype=vlen_xy,
                                   data=rings,
                                   compression='gzip', compression_opts=4)  # Add compression

            f.flush()  # Extra safety; forces metadata to disk

    def insert_rings(self, event: int, centers: np.ndarray, momenta: np.ndarray, particle_types: np.ndarray, N_hits: int = None) -> None:
        """
        Insert new rings into an existing event.

        Parameters
        ----------
        event : int
            The index of the event to insert rings into.
        centers : np.ndarray
            The centers of the new rings, shape (num_new_rings, 2).
        momenta : np.ndarray
            The momenta of the new particles, shape (num_new_rings,).
        particle_types : np.ndarray
            The types of the new particles, shape (num_new_rings,).
        N_hits : int, optional
            The number of hits for each new ring. If None, it will be computed as in `generate_event`.
            If provided, it should be an array of shape (num_new_rings,).
            Default is None.

        Raises
        ------
        ValueError
            If the shapes of centers, momenta, and particle_types do not match.
        IndexError
            If the event index is out of range.
        """
        if event < 0 or event >= len(self.dataset):
            raise IndexError("Event index out of range.")
        if centers.shape[0] != momenta.shape[0] or centers.shape[0] != particle_types.shape[0]:
            raise ValueError("Shapes of centers, momenta, and particle_types must match.")

        beta = momenta / np.sqrt(np.square(momenta) + np.square([self.masses[pt] for pt in particle_types]))
        arccos_arg = 1 / (self.refractive_index * beta)
        cherenkov_angles = np.where(arccos_arg <= 1, np.arccos(arccos_arg), 0)
        radii = np.tan(cherenkov_angles) * self.max_cherenkov_radius / np.tan(self.max_cherenkov_angle)

        if N_hits is None:
            # Compute the number of hits as in `generate_event`
            N_mean = np.where(
                radii > 0,
                np.round(
                    self.N_init * np.square(np.sin(cherenkov_angles)) / np.square(np.sin(self.max_cherenkov_angle))
                ),
                0
            ).astype(int)
            N_hits = np.random.poisson(N_mean)

        # Create the new rings
        new_rings = self._create_rings(centers, radii, N_hits)
        new_tracked_rings = np.arange(len(self.dataset[event]['rings']), len(self.dataset[event]['rings']) + len(new_rings))
        
        # Insert the new rings into the event
        event_data = self.dataset[event]
        event_data['centers'] = np.vstack((event_data['centers'], centers))
        event_data['momenta'] = np.concatenate((event_data['momenta'], momenta))
        event_data['particle_types'] = np.concatenate((event_data['particle_types'], particle_types))
        # Always treat 'rings' as a list of arrays, concatenate as list
        event_data['rings'] = list(event_data['rings']) + list(new_rings)
        event_data['tracked_rings'] = np.concatenate((event_data['tracked_rings'], new_tracked_rings))

    def event_image(self, idx: int, img_size: tuple[int, int], normalize: bool = False) -> np.ndarray:
        """
        Rasterise the selected event to a 2-D hit-count map.

        Parameters
        ----------
        idx        : event index
        img_size   : (H, W) in pixels
        normalize  : if True, scale to [0,1]

        Returns
        -------
        np.ndarray[H,W]   float32 image
        """
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError("Index out of range.")

        h, w = img_size
        (x_min, x_max), (y_min, y_max) = self.detector_size
        sx = (w - 1) / (x_max - x_min)
        sy = (h - 1) / (y_max - y_min)

        img = np.zeros((h, w), dtype=np.float32)
        for ring in self.dataset[idx]['rings']:
            if not len(ring):
                continue
            x_pix = np.rint((ring[:, 0] - x_min) * sx).astype(int)
            y_pix = np.rint((y_max - ring[:, 1]) * sy).astype(int)         # y origin at top
            m = (x_pix >= 0) & (x_pix < w) & (y_pix >= 0) & (y_pix < h)
            img[y_pix[m], x_pix[m]] += 1

        if normalize and img.max() > 0:
            img /= img.max()

        return img

    def num_rings(self) -> int:
        """
        Return the total number of rings in the dataset.

        Returns
        -------
        int
            The total number of rings across all events.
        """
        return sum(len(event['rings']) for event in self.dataset)

    def __len__(self):
        """
        Return the number of events in the dataset.
        """
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get the event at the specified index.

        Parameters
        ----------
        idx : int
            The index of the event to retrieve.

        Returns
        -------
        dict[str, Any]
            The event data at the specified index.
        """
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError("Index out of range.")
        return self.dataset[idx]



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.Utility.utils import load_kde

    ptypes = [211, 321, 2212, 11, 13]
    log_momenta_dist = {}
    for ptype in ptypes:
        #D:\uni\YOLO RICH PID\data\distributions\log_momenta_kdes\211-kde.npz
        log_momenta_dist[ptype] = load_kde(f'D:/uni/YOLO RICH PID/data/distributions/log_momenta_kdes/{ptype}-kde.npz')

    centers_dist = load_kde('D:/uni/YOLO RICH PID/data/distributions/centers_R1-kde.npz')
    radial_noise = (0, 1.5)  # Mean and standard deviation for radial noise
    scgen = SCGen(
        particle_types=ptypes,
        refractive_index=1.003,
        detector_size=((-600, 600), (-600, 600)),
        momenta_log_distributions=log_momenta_dist,
        centers_distribution=centers_dist,
        radial_noise=radial_noise,
        N_init=20,
        max_radius=100.0
    )
    dataset = scgen.generate_dataset(
        num_events=10,
        num_particles_per_event=(2, 2),
        parallel=True,
        batch_size=5,
        progress_bar=True
    )
    print(f"Generated {len(dataset)} events with {sum(len(event['rings']) for event in dataset)} total rings.")

    dataset = scgen.prune_centers(keep_probability=0.0)
    scgen.insert_rings(
        event=0,
        centers=np.array([[0, 0], [100, 100]]),
        momenta=np.array([10e9, 15e9]),
        particle_types=np.array([211, 321]),
        N_hits=None  # Automatically compute N_hits based on the radii
    )

    # Plot the first event
    event = dataset[0]
    plt.figure(figsize=(10, 10))
    for i, ring in enumerate(event['rings']):
        if len(ring) > 0:
            plt.scatter(ring[:, 0], ring[:, 1], s=1)
    plt.scatter(event['centers'][:, 0], event['centers'][:, 1], c='red', s=10, label='Centers')
    plt.xlim(scgen.detector_size[0])
    plt.ylim(scgen.detector_size[1])
    plt.title(f"Event 0: {len(event['rings'])} rings")
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    
    plt.imshow(scgen.event_image(0, (32, 32), normalize=False), cmap='gray', extent=scgen.detector_size[0] + scgen.detector_size[1])
    plt.title("Event 0 Image")
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.colorbar(label='Number of hits')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
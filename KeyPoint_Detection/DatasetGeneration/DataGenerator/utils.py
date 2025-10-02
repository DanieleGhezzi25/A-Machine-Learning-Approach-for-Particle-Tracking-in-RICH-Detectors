from math import e
import numpy as np
import h5py
import os
import json
from regex import F
from sqlalchemy import all_
from sympy import N
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

# TODO:
# - Sharding: save images in subfolders to avoid too many files in one folder

def calculate_cherenkov_angle( momentum: float, mass: float, refractive_index: float) -> float:
    beta = momentum / np.sqrt(momentum**2 + mass**2)
    arg = 1 / (refractive_index * beta)
    if arg > 1: # If arguement of arccos is greater than 1, then we are below the threshold for Cherenkov radiation, set radius to 0
        #warnings.warn("The argument of the arccos is greater than 1. The Cherenkov angle will be set to 0.")
        return 0
    theta_c = np.arccos(1 / (refractive_index * beta))
    return theta_c

def calculate_cherenkov_radius(theta_c: float, refractive_index: float, MAX_RADIUS: float = 100) -> float:
    # Calculate the radius so that R_max is fixed
    R_max = MAX_RADIUS # from Fig 2.14 of CERN-THESIS-2021-215
    theta_c_max = np.arccos(1 / (refractive_index))
    R = np.tan(theta_c) * R_max / np.tan(theta_c_max)
    return R


def _read_rings(rings_ds):
    rings = []
    for idx in range(len(rings_ds)):
        try:
            r = rings_ds[idx]
            if r is not None and hasattr(r, 'dtype') and 'x' in r.dtype.names and 'y' in r.dtype.names:
                rings.append(np.stack([r['x'], r['y']], axis=1))
            else:
                rings.append(np.empty((0, 2), dtype=np.float32))
        except Exception:
            rings.append(np.empty((0, 2), dtype=np.float32))
    return rings

def read_dataset(data_file):
    """
    Read a dataset from an h5 file and return the data and labels.
    
    Parameters
    ----------
    data_file : str
        Path to the h5 file containing the dataset.
    
    Returns
    -------
    dataset: dict
        A dictionary containing the dataset in the exact same format as the one
        returned by the SCGen generate_dataset method.
    """
    dataset = []
    with h5py.File(data_file, 'r') as f:
        N_events = f.attrs['num_events'].astype(int)
        for i in range(N_events):
            event_grp = f[f'event_{i}']
            event = {
                'centers': np.array(event_grp['centers']),
                'momenta': np.array(event_grp['momenta']),
                'particle_types': np.array(event_grp['particle_types']),
                'rings': _read_rings(event_grp['rings']),
                'tracked_rings': np.array(event_grp['tracked_rings']),
                'radii': np.array(event_grp['radii'])
            }
            dataset.append(event)
    return dataset

def read_dataset_metadata(data_file):
    """
    Read the metadata of a dataset from an h5 file.
    
    Parameters
    ----------
    data_file : str
        Path to the h5 file containing the dataset.
    
    Returns
    -------
    metadata: dict
        A dictionary containing the metadata of the dataset.
    """
    with h5py.File(data_file, 'r') as f:
        metadata = {
            'refractive_index': f.attrs['refractive_index'],
            'detector_size': f.attrs['detector_size'],
            'radial_noise': f.attrs['radial_noise'],
            'N_init': f.attrs['N_init'],
            'max_cherenkov_radius': f.attrs['max_cherenkov_radius'],
            'particle_types': list(f.attrs['particle_types']),
            'masses': json.loads(f.attrs['masses']),  # Parse JSON back to dict
            'num_events': f.attrs['num_events']
        }
    return metadata

def generate_MYOLO_dataset(
    momenta_range,
    output_file = None,
    particles=[321, 211, 2212, 11, 13],  # K+, pi+, p, e-, mu-
    num_added_ring_per_event=5,
    **kwargs
):
    """
    Generates a dataset using SCGen that has a uniform distribution of momenta and center positions across the detector.
    The idea is to use the normally generated events as background (prune_centers(0)) and add rings to them with a uniform distribution of momenta
    and center positions across the detector.

    Parameters
    ----------
    momenta_range : tuple
        A tuple containing the minimum and maximum momenta for the particles in the dataset.
    output_file : str
        Path to the output h5 file where the dataset will be saved.
        If none it will return the dataset object instead of saving it
    particles : list
        A list of particle (provided as PDGIDs) types to be included in the dataset.
        Please note that this should match the particle types used in the SCGen configuration.
        Default is [321, 211, 2212, 11, 13] which corresponds to K+, pi+, p, e-, mu-.
    num_added_ring_per_event : int
        Number of rings to be added per event. Default is 5.
    **kwargs : dict
        Additional keyword arguments to be passed to the SCGen initialization and generate_dataset methods.
        If not provided, default values will be used.
    """
    import sys
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(this_file_dir))
    from Utility.utils import load_kde
    from DataGenerator.scgen import SCGen

    if 'momenta_log_distributions' not in kwargs:
        momenta_log_distributions = {}
        for ptype in particles:
            momenta_log_distributions[ptype] = load_kde(os.path.join(this_file_dir, f'../../data/distributions/log_momenta_kdes/{ptype}-kde.npz'))
    else:
        momenta_log_distributions = kwargs['momenta_log_distributions']


    scgen_params = {
        'particle_types': kwargs.get('particle_types', particles),
        'refractive_index': kwargs.get('refractive_index', 1.0014),
        'detector_size': kwargs.get('detector_size', ((-600, 600), (-600, 600))),  # Detector size in mm
        'momenta_log_distributions' : momenta_log_distributions,
        'centers_distribution': kwargs.get('centers_distribution', load_kde(os.path.join(this_file_dir, '../../data/distributions/centers_R1-kde.npz'))),
        'radial_noise': kwargs.get('radial_noise', (0, 1.5)),
        'N_init': kwargs.get('N_init', 60),
        'max_radius': kwargs.get('max_radius', 100.0),  # Maximum Cherenkov radius in mm
        'masses': kwargs.get('masses', None)
    }

    generate_dataset_params = {
        'num_events': kwargs.get('num_events', 20000),
        'num_particles_per_event': kwargs.get('num_particles_per_event', (170, 180)),
        'parallel': kwargs.get('parallel', True),
        'batch_size': kwargs.get('batch_size', 128),
        'progress_bar': kwargs.get('progress_bar', True)
    }

    # Generate background events using SCGen
    scgen = SCGen(**scgen_params)
    scgen.generate_dataset(**generate_dataset_params)
    scgen.prune_centers(0) # Consider all events as background

    # Add "signal" rings to the background events
    total_num_rings = generate_dataset_params['num_events'] * num_added_ring_per_event
    ptypes = np.random.choice(particles, size= total_num_rings)
    momenta = np.random.uniform(momenta_range[0], momenta_range[1], size=total_num_rings)
    centers = np.random.uniform(
        [scgen_params['detector_size'][0][0], scgen_params['detector_size'][1][0]],
        [scgen_params['detector_size'][0][1], scgen_params['detector_size'][1][1]],
        size=(total_num_rings, 2)
    )

    for i in range(generate_dataset_params['num_events']):
        scgen.insert_rings(event=i, centers=centers[i*num_added_ring_per_event:(i+1)*num_added_ring_per_event],
                         momenta=momenta[i*num_added_ring_per_event:(i+1)*num_added_ring_per_event],
                         particle_types=ptypes[i*num_added_ring_per_event:(i+1)*num_added_ring_per_event])
        
    if output_file is None:
        return scgen.dataset
    else:
        scgen.save_dataset(output_file)

    

def generate_MYOLO_dataset_folder(
    data_file,
    base_dir,
    image_size=128,
    max_photons=1,
    train_val_test_split=(0.8, 0.1, 0.1),
    train_data_file=None,
    annular_mask=True,
    polar_transform=True,
    seed=42,
    as_png=False,
    debug_hilight=False,
    images_per_folder=1000
):
    """
    Generate a MYOLO dataset folder from a SyntheticCherenkovDataGenerator h5 dataset file.

    The dataset is split into train, validation, and test sets based on the provided split ratios.
    The folder structure will be:

    MYOLO_dataset/
        ├── images/
        │   ├── image_0.jpg
        │   ├── image_1.jpg
        │   └── ...
        ├── train.txt
        ├── val.txt
        ├── test.txt
        └── particles.txt  (List of particles in the dataset)

    Parameters
    ----------
    data_file : str
        Path to the h5 file containing the dataset.
    base_dir : str
        Base directory where the MYOLO dataset will be created.
    image_size : int
        Size of the images to be generated (default is 128).
    max_photons : float
        Maximum number of photons in a 1mm x 1mm area (default is 20).
        It is used to compute the normalization factor for the images (based on their size).
    train_val_test_split : tuple
        Tuple containing the split ratios for train, validation, and test sets.
        Default is (0.8, 0.1, 0.1) which means 80% for training, 10% for validation, and 10% for testing.
    train_data_file : str, optional
        Path to a file containing the training data. If provided, it will be used to generate the training set.
        This will override the `train_val_test_split` train split, if using it you should set it to (0.0, val_split, test_split);
        ensuring that val_split + test_split = 1.0.
        Please note that if using a train_data_file, it's **metadata should match the one in data_file except for the number of events**.
    annular_mask : bool
        If True, applies an annular mask to the rings based on the max and min hypothetical Cherenkov radii.
    polar_transform : bool
        If True, applies a polar transformation to the rings, converting them to polar coordinates.
    seed : int
        Random seed for reproducibility.
    as_png : bool
        If True, saves the images as PNG files, otherwise saves them as numpy arrays (default is False).
    debug_hilight : bool
        If True, highlights the pixel of the main ring for each image.
    images_per_folder : int
        Number of images to save in each folder. If the number of images exceeds this value, it will create subfolders.
        This is useful to avoid having too many files in one folder which can cause performance issues on some filesystems.
        If set to None, no sharding will be applied.

    Raises
    ------
    ValueError
        If the sum of `train_val_test_split` is not equal to 1.0.
    ValueError
        If the `image_size` is not a positive integer.
    ValueError
        If `max_photons` is not a positive float.
    """
    # Validate input parameters
    if not isinstance(train_val_test_split, (tuple, list)) or len(train_val_test_split) != 3:
        raise ValueError("train_val_test_split must be a tuple or list of three values (train, val, test).")
    if not np.isclose(sum(train_val_test_split), 1.0):
        raise ValueError("The sum of train_val_test_split must be equal to 1.0.")
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer.")
    if not isinstance(max_photons, (int, float)) or max_photons <= 0:
        raise ValueError("max_photons must be a positive float.")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if not os.path.exists(data_file):
        raise ValueError(f"The data file {data_file} does not exist.")


    dataset = read_dataset(data_file)
    metadata = read_dataset_metadata(data_file)
    np.random.seed(seed)

    # Check if train_data_file is provided 
    if train_data_file is not None:
        train_dataset = read_dataset(train_data_file)
        train_metadata = read_dataset_metadata(train_data_file)
        # Compare the metadata of the train_data_file with the data_file excluding the number of events
        match, mismatches = _compare_metadata(train_metadata, metadata)
        if not match:
            raise ValueError(f"The metadata of the train_data_file does not match the metadata of the data_file. Mismatches: {mismatches}")
        
        # Update the train_val_test_split so that the train set is the size of the train_data_file

        num_rings_valtest = sum(len(event['tracked_rings']) for event in dataset)
        num_rings_train = sum(len(event['tracked_rings']) for event in train_dataset)
        tot_num_rings = num_rings_train + num_rings_valtest
        train_size = int(num_rings_train)
        val_size = int(num_rings_valtest * train_val_test_split[1])
        test_size = int(num_rings_valtest * train_val_test_split[2])
        # print(f'[DEBUG] train_size: {train_size}, val_size: {val_size}, test_size: {test_size}')
        # Append the train dataset in front of the dataset
        dataset = train_dataset + dataset
    else:
        tot_num_rings = sum(len(event['tracked_rings']) for event in dataset)
        train_size = int(tot_num_rings * train_val_test_split[0])
        val_size = int(tot_num_rings * train_val_test_split[1])
        test_size = tot_num_rings - train_size - val_size

    os.makedirs(base_dir, exist_ok=True)
    images_dir = os.path.join(base_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # If images_per_folder is set to None, we will not shard the images
    images_per_folder = tot_num_rings + 1

    # Check if sharding is needed
    if tot_num_rings > images_per_folder:
        sharding_idxs = np.arange(0, tot_num_rings, images_per_folder)
        # Create subfolders for sharding
        for folder_idx in range(len(sharding_idxs)):
            subfolder = os.path.join(base_dir, "images", f'shard-{folder_idx}')
            os.makedirs(subfolder, exist_ok=True)
    else:
        sharding_idxs = None

    max_radius = metadata['max_cherenkov_radius'] + 4 * metadata['radial_noise'][1]
    max_distance = max_radius * 2  # Diameter of the ring
    pixel_size = max_distance / image_size
    normalization_factor = (pixel_size ** 2) * max_photons

    scale_factor = (image_size - 1) / max_distance

    ptypes = metadata['particle_types']
    masses = metadata['masses']
    masses_list = masses.values()

    all_images = []
    all_labels = []
    all_momenta = []
    saved_images = 0
    shard = 0 # Keep track of the current shard (not used if sharding is None)
    for event_idx, event in enumerate(tqdm(dataset, desc="Generating images")):
        centers = event['centers']
        momenta = event['momenta']
        rings = event['rings']
        images = []
        labels = []
        for center_idx in range(len(centers)):
            center = centers[center_idx]
            # this_ring = rings[center_idx]
            # if this_ring.shape[0] == 0:
            #     # If there are no rings for this center, skip it
            #     continue
            in_range_idxs = np.where(np.linalg.norm(centers - center, axis=1) <= max_radius)[0]
            points   = []
            ring_id  = []
            for ridx in in_range_idxs:
                pts = rings[ridx].copy()
                pts[:, 0] -= center[0]
                pts[:, 1] -= center[1]
                points.append(pts)
                ring_id.extend([ridx] * len(pts))
            if points:
                in_range_rings = np.concatenate(points, axis=0)
                ring_id = np.asarray(ring_id, dtype=np.int32)
            else:
                in_range_rings = np.empty((0, 2), dtype=np.float32)
                ring_id = np.empty((0,), dtype=np.int32)
            if annular_mask:
                distances = np.linalg.norm(in_range_rings, axis=1)
                min_hyp_ring = calculate_cherenkov_radius(
                    calculate_cherenkov_angle(
                        momenta[center_idx], max(masses_list), metadata['refractive_index']
                    ),
                    metadata['refractive_index'],
                    max_radius
                ) + 4 * metadata['radial_noise'][1]
                max_hyp_ring = calculate_cherenkov_radius(
                    calculate_cherenkov_angle(
                        momenta[center_idx], min(masses_list), metadata['refractive_index']
                    ),
                    metadata['refractive_index'],
                    max_radius
                ) + 4 * metadata['radial_noise'][1]
                mask = (distances >= min_hyp_ring) & (distances <= max_hyp_ring)
                in_range_rings = in_range_rings[mask]
                ring_id = ring_id[mask]

            if polar_transform:
                r = np.linalg.norm(in_range_rings, axis=1)
                theta = np.arctan2(in_range_rings[:, 1], in_range_rings[:, 0])
                theta = (theta + 2 * np.pi) % (2 * np.pi)
                theta_scale = (image_size - 1) / (2 * np.pi)
                radius_scale = (image_size - 1) / max_radius
                x_pix = np.rint(theta * theta_scale).astype(int)
                y_pix = np.rint(r * radius_scale).astype(int)
            else:
                x_pix = np.rint((in_range_rings[:, 0] + max_radius) * scale_factor).astype(int)
                y_pix = np.rint((max_radius - in_range_rings[:, 1]) * scale_factor).astype(int)

            # Ensure pixel coordinates are within bounds
            mask = (x_pix >= 0) & (x_pix < image_size) & (y_pix >= 0) & (y_pix < image_size)
            ring_id = ring_id[mask]
            x_pix   = x_pix[mask]
            y_pix   = y_pix[mask]

            if debug_hilight:
                image = np.zeros((image_size, image_size, 3), dtype=np.float32)
                main_mask = (ring_id == center_idx)
                image[y_pix[main_mask],     x_pix[main_mask]]     = (1, 0, 0)  # red ­- main ring
                image[y_pix[~main_mask],    x_pix[~main_mask]]    = (1, 1, 1)  # white – others
            else:
                image = np.zeros((image_size, image_size), dtype=np.float32)
                image[y_pix, x_pix] += 1

            image = image / normalization_factor
            image = np.clip(image, 0, 1)
            images.append(image)
            labels.append(event['particle_types'][center_idx])

        if as_png:
            for idx, img in enumerate(images):
                img_name = f'event-{event_idx}_ridx-{idx}_type-{labels[idx]}_mom-{momenta[idx]}.png'
                if sharding_idxs is not None:
                    img_name = os.path.join(f'shard-{shard}', img_name)
                img_path = os.path.join(images_dir, img_name)
                from PIL import Image
                if debug_hilight:
                    Image.fromarray((img * 255).astype(np.uint8), mode='RGB').save(img_path)
                else:
                    Image.fromarray((img * 255).astype(np.uint8), mode='L').save(img_path)
                all_images.append(img_name)
                all_labels.append(f"{labels[idx]}")
                all_momenta.append(momenta[idx])
                saved_images += 1
                if sharding_idxs is not None:
                    if saved_images % images_per_folder == 0:
                        shard += 1
        else:
            for idx, img in enumerate(images):
                image_name = f'event-{event_idx}_ridx-{idx}_type-{labels[idx]}_mom-{momenta[idx]}.npy'
                if sharding_idxs is not None:
                    image_name = os.path.join(f'shard-{shard}', image_name)
                img_path = os.path.join(images_dir, image_name)
                np.save(img_path, img)
                all_images.append(image_name)
                all_labels.append(f"{labels[idx]}")
                all_momenta.append(momenta[idx])
                saved_images += 1
                if sharding_idxs is not None:
                    if saved_images % images_per_folder == 0:
                        shard += 1

    train_file = os.path.join(base_dir, 'train.txt')
    val_file = os.path.join(base_dir, 'val.txt')
    test_file = os.path.join(base_dir, 'test.txt')
    particles_file = os.path.join(base_dir, 'particles.txt')

    combined_img_labels = list(zip(all_images, all_labels, all_momenta))
    if train_data_file is not None:
        # We shuffle only the validation and test sets, the training set is already provided in train_data_file
        # print(f"[DEBUG] train size: {train_size}, val size: {val_size}, test size: {test_size}")
        train_combined_img_labels = combined_img_labels[:train_size]
        valtest_combined_img_labels = combined_img_labels[train_size:]
        np.random.shuffle(valtest_combined_img_labels)
        combined_img_labels = train_combined_img_labels + valtest_combined_img_labels
    else:
        np.random.shuffle(combined_img_labels)
    with open(train_file, 'w') as f:
        for img, label, momentum in combined_img_labels[:train_size]:
            f.write(f"{img} {label} {momentum}\n")
    with open(val_file, 'w') as f:
        for img, label, momentum in combined_img_labels[train_size:train_size + val_size]:
            f.write(f"{img} {label} {momentum}\n")
    with open(test_file, 'w') as f:
        for img, label, momentum in combined_img_labels[train_size + val_size:]:
            f.write(f"{img} {label} {momentum}\n")

    with open(particles_file, 'w') as f:
        for ptype in ptypes:
            f.write(f"{ptype}\n")

    print(f"Generated {len(all_images)} images in {images_dir}")
    print(f"Train set: {train_size} images")
    print(f"Validation set: {val_size} images")
    print(f"Test set: {test_size} images")


# UTILITY FUNCTION (yes a bit akward to have utility section in utils.py, but that is how it is)
def _to_hashable(x):
    if isinstance(x, np.ndarray):
        return tuple(_to_hashable(i) for i in x.tolist())
    elif isinstance(x, list):
        return tuple(_to_hashable(i) for i in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, _to_hashable(v)) for k, v in x.items()))
    else:
        return x

def _compare_metadata(meta1, meta2):
    mismatches = []
    keys1 = set(meta1.keys()) - {'num_events'}
    keys2 = set(meta2.keys()) - {'num_events'}
    if keys1 != keys2:
        missing1 = keys2 - keys1
        missing2 = keys1 - keys2
        if missing1:
            mismatches.append(f"Missing in train_metadata: {missing1}")
        if missing2:
            mismatches.append(f"Missing in metadata: {missing2}")
        return False, mismatches
    for k in keys1:
        v1 = meta1[k]
        v2 = meta2[k]
        # Compare numpy arrays
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            if set(_to_hashable(v1)) != set(_to_hashable(v2)):
                mismatches.append(f"Key '{k}': arrays differ (as sets): {v1} vs {v2}")
        # Compare lists
        elif isinstance(v1, list) and isinstance(v2, list):
            if set(_to_hashable(v1)) != set(_to_hashable(v2)):
                mismatches.append(f"Key '{k}': lists differ (as sets): {v1} vs {v2}")
        # Compare dicts
        elif isinstance(v1, dict) and isinstance(v2, dict):
            if set(v1.keys()) != set(v2.keys()):
                mismatches.append(f"Key '{k}': dict keys differ: {v1.keys()} vs {v2.keys()}")
            else:
                for kk in v1:
                    vv1 = v1[kk]
                    vv2 = v2[kk]
                    # If values are arrays or lists, compare as sets
                    if isinstance(vv1, np.ndarray) and isinstance(vv2, np.ndarray):
                        if set(_to_hashable(vv1)) != set(_to_hashable(vv2)):
                            mismatches.append(f"Key '{k}[{kk}]': arrays differ (as sets): {vv1} vs {vv2}")
                    elif isinstance(vv1, list) and isinstance(vv2, list):
                        if set(_to_hashable(vv1)) != set(_to_hashable(vv2)):
                            mismatches.append(f"Key '{k}[{kk}]': lists differ (as sets): {vv1} vs {vv2}")
                    else:
                        if vv1 != vv2:
                            mismatches.append(f"Key '{k}[{kk}]': values differ: {vv1} vs {vv2}")
        # Compare floats/ints
        elif isinstance(v1, (float, int)) and isinstance(v2, (float, int)):
            if isinstance(v1, float) or isinstance(v2, float):
                if not np.isclose(v1, v2):
                    mismatches.append(f"Key '{k}': floats differ: {v1} vs {v2}")
            else:
                if v1 != v2:
                    mismatches.append(f"Key '{k}': ints differ: {v1} vs {v2}")
        # Compare tuples
        elif isinstance(v1, tuple) and isinstance(v2, tuple):
            if len(v1) != len(v2):
                mismatches.append(f"Key '{k}': tuple lengths differ: {v1} vs {v2}")
            else:
                for i, (a, b) in enumerate(zip(v1, v2)):
                    if isinstance(a, float) and isinstance(b, float):
                        if not np.isclose(a, b):
                            mismatches.append(f"Key '{k}[{i}]': floats in tuple differ: {a} vs {b}")
                    else:
                        if a != b:
                            mismatches.append(f"Key '{k}[{i}]': tuple values differ: {a} vs {b}")
        else:
            if v1 != v2:
                mismatches.append(f"Key '{k}': values differ: {v1} vs {v2}")
    return len(mismatches) == 0, mismatches


if __name__ == "__main__":
    # Example usage
    data_file = 'D:/uni/YOLO RICH PID/data/datasets/MYOLO_valtest_dataset.h5'
    train_data_file = 'D:/uni/YOLO RICH PID/data/datasets/MYOLO_train_dataset.h5'
    base_dir = 'D:/uni/YOLO RICH PID/data/datasets/MYOLO_dataset'
    generate_MYOLO_dataset_folder(
        data_file=data_file,
        base_dir=base_dir,
        as_png=False,
        annular_mask=True,
        polar_transform=True,
        debug_hilight=False,
        train_val_test_split=(0.0, 0.5, 0.5),  # Using a train_data_file, so set train split to 0.0
        train_data_file=train_data_file
    )
    print(f"MYOLO dataset generated at {base_dir}")
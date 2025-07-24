from src.DataGenerator.RingsGenerator import generateRings
import numpy as np
import math
from typing import Any, Dict, List, Optional, Tuple
import random
import pickle as pkl
import os
from matplotlib import pyplot as plt
import colorsys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor 
from tqdm.auto import tqdm
from PIL import Image

def generate_single_event_global(args):
    """
    Global helper function to generate a single event.
    Expects a tuple with:
      (generator, N_rings_per_event, radial_noise, min_radius, only_above_treshold, min_above_threshold, index)
    """
    generator, N_rings_per_event, radial_noise, min_radius, only_above_treshold, min_above_threshold, _ = args
    # Generate a random number of rings for this event
    N_rings = random.randint(N_rings_per_event[0], N_rings_per_event[1])
    return generator.generate_event(
        N_rings,
        radial_noise,
        min_radius=min_radius,
        only_above_treshold=only_above_treshold,
        min_above_threshold=min_above_threshold
    )

def process_event_images(args):
    """
    Helper function to process image creation for a single event.
    Expects a tuple containing:
      (event_idx, event, image_size, save_folder, original_save_folder,
       train_val_test_split, event_assignment, all_same_folder, radial_noise,
       MAX_RADIUS, polar_transform, image_noise, use_annular_mask,
       calculate_thetaC, calculate_radius, get_masses, particle_types_map,
       highligh_ring_pixels)
    """
    # Unpack the arguments
    (event_idx,
     event,
     image_size,
     save_folder,
     original_save_folder,
     train_val_test_split,
     event_assignment,
     all_same_folder,
     radial_noise,
     MAX_RADIUS,
     polar_transform,
     image_noise,
     use_annular_mask,
     calculate_thetaC,
     calculate_radius,
     get_masses,
     particle_types_map,
     highligh_ring_pixels) = args

    # Determine the local save folder based on train/val/test split
    if train_val_test_split and not all_same_folder:
        folder_choice = ['train', 'val', 'test'][event_assignment[event_idx]]
        save_folder_local = os.path.join(original_save_folder, folder_choice)
    else:
        save_folder_local = save_folder

    # Retrieve masses and build the reverse mapping for particle types
    masses = get_masses()
    id_to_particle_type = {v: k for k, v in particle_types_map.items()}

    # Convert event points and labels to numpy arrays for fast indexing
    x_points = np.array(event['x_points'])
    y_points = np.array(event['y_points'])
    labels_all = np.array(event['labels'])
    x_centers = event['x_centers']
    y_centers = event['y_centers']
    momentums = event['momentums']
    particle_types_labels = event['particle_types']

    # Process each ring within the event
    for ring_idx, (x_center, y_center, momentum, particle_type_label) in enumerate(
            zip(x_centers, y_centers, momentums, particle_types_labels)):
        # Compute physical image dimensions
        image_phys_width = (MAX_RADIUS + 2 * radial_noise[1]) * 2
        image_phys_height = (MAX_RADIUS + 2 * radial_noise[1]) * 2
        pixel_width = image_phys_width / image_size[1]
        pixel_height = image_phys_height / image_size[0]

        # Compute the physical bounds of the image (centered at the ring)
        x_min = x_center - (image_phys_width / 2)
        x_max = x_center + (image_phys_width / 2)
        y_min = y_center - (image_phys_height / 2)
        y_max = y_center + (image_phys_height / 2)

        # Select points within the square region and grab their labels
        within_square = ((x_points >= x_min) & (x_points <= x_max) &
                         (y_points >= y_min) & (y_points <= y_max))
        x_points_in_square = x_points[within_square]
        y_points_in_square = y_points[within_square]
        labels_in_square = labels_all[within_square]

        # Compute distances from the ring center
        distances = np.sqrt((x_points_in_square - x_center)**2 +
                            (y_points_in_square - y_center)**2)

        # Compute possible ring radii for all particle masses at the given momentum
        possible_radii = []
        for mass in masses.values():
            thetaC = calculate_thetaC(momentum, mass)
            R_val = calculate_radius(thetaC)
            if R_val > 0:
                possible_radii.append(R_val)
        if not possible_radii:
            continue  # Skip if no valid radii (e.g. below threshold)
        min_radius_possible = min(possible_radii)
        max_radius_possible = max(possible_radii)
        radial_padding = 2 * radial_noise[1]

        # Apply the annular mask if enabled
        if use_annular_mask:
            within_annulus = ((distances >= min_radius_possible - radial_padding) &
                              (distances <= max_radius_possible + radial_padding))
            x_points_in_annulus = x_points_in_square[within_annulus]
            y_points_in_annulus = y_points_in_square[within_annulus]
            labels_in_annulus = labels_in_square[within_annulus]
        else:
            x_points_in_annulus = x_points_in_square
            y_points_in_annulus = y_points_in_square
            labels_in_annulus = labels_in_square

        # Transform points to pixel coordinates
        if polar_transform:
            # Shift points relative to the ring center
            x_shifted = x_points_in_annulus - x_center
            y_shifted = y_points_in_annulus - y_center
            # Convert to polar coordinates
            radii_pts = np.sqrt(x_shifted**2 + y_shifted**2)
            angles = np.arctan2(y_shifted, x_shifted)
            # Define mapping ranges
            r_min_range = min_radius_possible - radial_padding
            r_max_range = max_radius_possible + radial_padding
            theta_min = -np.pi
            theta_max = np.pi
            i_coords = ((angles - theta_min) / (theta_max - theta_min) * (image_size[1] - 1)).astype(int)
            j_coords = ((radii_pts - r_min_range) / (r_max_range - r_min_range) * (image_size[0] - 1)).astype(int)
        else:
            # Map the Cartesian coordinates to pixel positions
            i_coords = ((x_points_in_annulus - x_min) / pixel_width).astype(int)
            j_coords = ((y_points_in_annulus - y_min) / pixel_height).astype(int)

        # Filter out any points that fall outside the image bounds
        valid_mask = ((i_coords >= 0) & (i_coords < image_size[1]) &
                      (j_coords >= 0) & (j_coords < image_size[0]))
        i_coords = i_coords[valid_mask]
        j_coords = j_coords[valid_mask]
        labels_final = labels_in_annulus[valid_mask]  # these are the labels for the hits in this image

        if highligh_ring_pixels:
            # --- Highlight branch: build an RGB image preserving summed intensities for non-ring hits ---
            # Create an array to sum non-highlight hits (exactly as in the original summing behavior)
            non_highlight_count = np.zeros(image_size, dtype=np.float32)
            # Boolean mask for pixels that have at least one hit from the ring in question
            ring_hit_mask = np.zeros(image_size, dtype=bool)

            for idx in range(len(i_coords)):
                r_pix = j_coords[idx]
                c_pix = i_coords[idx]
                if labels_final[idx] == ring_idx:
                    ring_hit_mask[r_pix, c_pix] = True
                else:
                    non_highlight_count[r_pix, c_pix] += 1

            # Build a grayscale image from the summed counts.
            # This mimics the original behavior: pixel value = count * 255 cast to uint8.
            # (Note: If counts exceed 1, the values will wrap around, as in the original code.)
            gray_intensity = (non_highlight_count * 255).astype(np.uint8)
            rgb_image = np.stack([gray_intensity, gray_intensity, gray_intensity], axis=-1)

            # Override pixels with ring hits using the highlight color (#669BBC → (102, 155, 188))
            rgb_image[ring_hit_mask] = (102, 155, 188)

            # Optionally add Gaussian noise per channel
            if image_noise > 0.0:
                noise = np.random.normal(0, image_noise, rgb_image.shape)
                rgb_image = np.clip(rgb_image + noise, 0, 255).astype(np.uint8)

            # Convert to PIL image and enlarge 5× for presentation purposes
            im = Image.fromarray(rgb_image)
            new_size = (image_size[1] * 5, image_size[0] * 5)  # (width, height)
            im = im.resize(new_size, resample=Image.NEAREST)
        else:
            # --- Original branch: create a grayscale image as before ---
            image = np.zeros(image_size, dtype=np.float32)
            # Sum the hits in each pixel
            for idx in range(len(i_coords)):
                image[j_coords[idx], i_coords[idx]] += 1
            if image_noise > 0.0:
                noise = np.random.normal(0, image_noise, image.shape)
                image += noise
            im = Image.fromarray((image * 255).astype(np.uint8))

        # Build the filename based on whether images are saved in the same folder
        particle_type = id_to_particle_type[particle_type_label]
        if all_same_folder:
            filename = os.path.join(save_folder_local, f"image_{event_idx}-{ring_idx}_{momentum:.2f}_{particle_type}.png")
        else:
            filename = os.path.join(save_folder_local, str(particle_type),
                                    f"image_{event_idx}-{ring_idx}_{momentum:.2f}_{particle_type}.png")
        # Save the image
        im.save(filename)

    return event_idx




class SyntheticCherenkovDataGenerator:
    def __init__(self,
                 refractive_index: float,
                 distributions: Dict[str, Dict[str, Any]],
                 type_weights: Optional[Dict[str, float]] = None,
                 num_hits_prop_radius: Optional[int] = 0,
                 num_hits_physical: Optional[bool] = True,
                 fixed_detector_size: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None) -> None:
        """
        Initialize the generator with precomputed distributions.

        :param refractive_index: Refractive index of the medium.
        :param distributions: Dictionary containing distributions per particle type and parameter. It should also contain the particle mass in eV/c^2.
                             Format:
                             {
                                 'particle_type': {
                                     'mass': float,
                                     'center': 2D kde_object,
                                     'N_points': kde_object or fixed value or fixed range (tuple),
                                     'momentum': kde_object (kde of the log of the log of the momentum -> use the log because the momentum distribution is skewed). Can actualy also be a tuple for uniform distribution.
                                 },
                                 ...
                             }
        :param type_weights: Optional dictionary specifying the probability of each particle type.
                                If None, uniform probabilities are assumed.
                                Format:
                                {
                                    'particle_type': weight,
                                    ...
                                }
        :param num_hits_prop_radius: If it's 0 then it won't be used, otherwise it will be used to calculate the number of hits in the ring based on the radius of the ring. (as aN/R where a is this parameter)
        :param num_hits_physical: bool to switch on or off the calculation of the number of hits based on the expected photons yield for Cherenkov light.
        :param fixed_detector_size: If it's None then it won't be used, otherwise it will be used to generate the rings in a fixed detector size. all points outside the detector will be discarded. and if a center is outside the detector, the ring will be discarded.
        """
        # Constants from experiment
        self.MAX_RADIUS = 100 # from Fig 2.14 of CERN-THESIS-2021-215

        self.refractive_index = refractive_index
        self.distributions = distributions
        self.particle_types = list(distributions.keys())
        self.num_hits_prop_radius = num_hits_prop_radius
        self.num_hits_physical = num_hits_physical
        self.fixed_detector_size = fixed_detector_size

        # Encode the particle types as integers (labels)
        self.particle_types_map = {ptype: i for i, ptype in enumerate(self.particle_types)}

        if type_weights:
            if set(type_weights.keys()) != set(self.particle_types): # Check if the keys of type_weights are the same as the keys of distributions
                raise ValueError("type_weights must have the same keys as distributions.")
            self.type_weights = np.array([type_weights[ptype] for ptype in self.particle_types])
            if not np.isclose(self.type_weights.sum(), 1): # Check if the sum of the weights is 1 (or close to it) else normalize
                self.type_weights = self.type_weights / self.type_weights.sum()
            # Covert to numpy array 
            self.type_weights = np.array([type_weights[ptype] for ptype in self.particle_types])
        else:
            # Uniform distribution if no weights provided
            self.type_weights = np.array([1.0 / len(self.particle_types)] * len(self.particle_types))

    def calculate_thetaC(self, momentum: float, mass: float) -> float:
        beta = momentum / math.sqrt(momentum**2 + mass**2)
        arg = 1 / (self.refractive_index * beta)
        if arg > 1: # If arguement of arccos is greater than 1, then we are below the threshold for Cherenkov radiation, set radius to 0
            #warnings.warn("The argument of the arccos is greater than 1. The Cherenkov angle will be set to 0.")
            return 0
        theta_c = math.acos(1 / (self.refractive_index * beta))
        return theta_c

    def calculate_radius(self, theta_c: float) -> float:
        # Calculate the radius so that R_max is fixed
        R_max = self.MAX_RADIUS # from Fig 2.14 of CERN-THESIS-2021-215
        theta_c_max = math.acos(1 / (self.refractive_index))
        R = math.tan(theta_c) * R_max / math.tan(theta_c_max)
        return R


    def sample_parameters(self, particle_type: str, momenta_dist = None) -> Tuple[float, float, int, float, float]:
        """
        Sample parameters based on precomputed distributions for the given particle type.

        :param particle_type: The type of particle to sample parameters for.
        :param momenta_dist: To specify custom distribution for the momentum (for example for YOLO training image generation). If None the one used in the initialization will be used.
        :return: Tuple containing (x, y, radius, N_points, momentum) [The radius is calculated from the momentum and particle mass]. If the momentum is under the threshold for Cherenkov radiation, the radius is set to 0.
        """
        dist = self.distributions[particle_type]

        # Sample continuous parameters
        is_inside_detector = False
        # Sample center of the ring and make sure it's inside the detector (if fixed_detector_size is not None)
        while not is_inside_detector:
            resampled_center = dist['center'].resample(1)
            x, y = resampled_center[0].item(), resampled_center[1].item()
            if self.fixed_detector_size:
                if x < self.fixed_detector_size[0][0] or x > self.fixed_detector_size[0][1] or y < self.fixed_detector_size[1][0] or y > self.fixed_detector_size[1][1]:
                    continue
            is_inside_detector = True

        # Sample discrete parameter
        if isinstance(dist['N_points'], int):
            N_points = dist['N_points']
        elif isinstance(dist['N_points'], tuple):
            N_points = np.random.randint(dist['N_points'][0], dist['N_points'][1])
        else:
            N_points = np.round(dist['N_points'].resample(1)).astype(int)

        # Sample momentum
        if momenta_dist:
            dist['momentum'] = momenta_dist

        if isinstance(dist['momentum'], tuple):
            momentum = np.random.uniform(dist['momentum'][0], dist['momentum'][1])
        else:
            momentum_log = float(dist['momentum'].resample(1))
            momentum = math.exp(momentum_log) # Un-log the momentum

        # Calculate radius from momentum and particle mass
        mass = dist['mass']
        thetaC = self.calculate_thetaC(momentum, mass)
        R = self.calculate_radius(thetaC)

        # Update the number of hits based on the radius of the ring (if num_hits_prop_radius is not 0)
        if self.num_hits_prop_radius != 0:
            if R != 0:
                N_points = N_points * R / self.num_hits_prop_radius

        #Update the number of hits based on the expected photons yield for Cherenkov light
        if self.num_hits_physical:
            if R != 0:
                theta_max = math.acos(1 / (self.refractive_index))
                N_points = N_points * math.sin(thetaC)**2 / (math.sin(theta_max)**2)
            else:
                N_points = 0
            
        # sample N_points from poisson distribution
        N_points = np.random.poisson(N_points)

        return x, y, R, N_points, momentum
    
    def generate_event(self,
                       N_rings: int,
                       radial_noise: Tuple[float, float] = (0.0, 4),
                       min_radius: float = 0,
                       only_above_treshold: bool = False,
                       min_above_threshold: int = 0) -> Dict[str, Any]:
        """
        Generate a single synthetic event.

        :param N_rings: Number of rings in the event.
        :param radial_noise: Tuple (mean, std) for radial noise. mean and std of gaussian distribution.
        :param min_radius: Minimum radius for the rings. If the radius is below this value, the ring is discarded.
        :param only_above_treshold: If True, only rings above the Cerenkov threshold will be generated.
        :param min_above_threshold: Minimum number of rings above the Cerenkov threshold to be generated; this parameter is used only if only_above_treshold=False. (note that if an event has no rings above the threshold, it will cause problem with the graph constructor)

        :return: A dictionary representing the event with keys:
                 - 'x_points', 'y_points': Coordinates of hits
                 - 'labels': Ring indices -- for each hit (aka each point) the index of the ring it belongs to
                 - 'x_centers', 'y_centers': Centers of rings
                 - 'momentums': Momentums of particles (if available)
                 - 'radii': Radii of rings
                 - 'particle_types': id of particle type for each ring
        """
        self.radial_noise = radial_noise # Save the radial noise for the create_images function

        rings = []
        particle_type_list = []
        num_above_threshold = 0
        for _ in range(N_rings):
            # Choose particle type based on type_weights
            if only_above_treshold:
                is_above_threshold = False
                while not is_above_threshold:
                    particle_type = np.random.choice(self.particle_types, p=self.type_weights)
                    x, y, R, N_points, momentum = self.sample_parameters(particle_type)
                    if R > min_radius: # if min_radius is not spacified, this condition will just check that the momentum is above the Cerenkov threshold (R > 0)
                        is_above_threshold = True      
                rings.append( (x, y, R, N_points, momentum) )
                particle_type_list.append(self.particle_types_map[particle_type])
            else:
                # Frist generate rings above the threshold
                if num_above_threshold < min_above_threshold:
                    is_above_threshold = False
                    while not is_above_threshold:
                        particle_type = np.random.choice(self.particle_types, p=self.type_weights)
                        x, y, R, N_points, momentum = self.sample_parameters(particle_type)
                        if R > min_radius:
                            is_above_threshold = True
                            num_above_threshold += 1
                    rings.append( (x, y, R, N_points, momentum) )
                    particle_type_list.append(self.particle_types_map[particle_type])
                else:
                    # Generate the rest of the rings
                    particle_type = np.random.choice(self.particle_types, p=self.type_weights)
                    x, y, R, N_points, momentum = self.sample_parameters(particle_type)
                    rings.append( (x, y, R, N_points, momentum) )
                    particle_type_list.append(self.particle_types_map[particle_type])  

        # Generate ring points using the generateRings function
        ring_points = generateRings(rings, radial_noise=radial_noise)

        # Discard points outside the plane
        if self.fixed_detector_size:
            x_points, y_points, labels = [], [], []
            for i in range(len(ring_points[0])):
                if self.fixed_detector_size[0][0] <= ring_points[0][i] <= self.fixed_detector_size[0][1] and self.fixed_detector_size[1][0] <= ring_points[1][i] <= self.fixed_detector_size[1][1]:
                    x_points.append(ring_points[0][i])
                    y_points.append(ring_points[1][i])
                    labels.append(ring_points[2][i])
        else:
            x_points, y_points, labels = ring_points[0], ring_points[1], ring_points[2]

        event = {
            'x_points': x_points,
            'y_points': y_points,
            'labels': labels,
            'x_centers': ring_points[4],
            'y_centers': ring_points[5],
            'momentums': ring_points[6],
            'radii': ring_points[3],
            'particle_types': particle_type_list
        }

        return event
    
    def generate_dataset(self,
                         num_events: int,
                         N_rings_per_event: Tuple[int, int],
                         radial_noise: Tuple[float, float] = (0.0, 0.01),
                         min_radius: float = 0,
                         only_above_treshold: bool = False,
                         min_above_threshold: int = 0) -> List[Dict[str, Any]]:
        """
        Generate a synthetic dataset in parallel.

        :param num_events: Number of synthetic events to generate.
        :param N_rings_per_event: Tuple (min, max) specifying the number of rings per event.
        :param radial_noise: Tuple (mean, std) for radial noise.
        :param min_radius: Minimum radius for the rings.
        :param only_above_treshold: If True, only rings above the Cherenkov threshold will be generated.
        :param min_above_threshold: Minimum number of rings above the Cherenkov threshold.
        :return: List of synthetic events.
        """
        # save radial noise
        self.radial_noise = radial_noise

        # Build argument list for each event
        args_list = [
            (self, N_rings_per_event, radial_noise, min_radius, only_above_treshold, min_above_threshold, i)
            for i in range(num_events)
        ]
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            synthetic_dataset = list(tqdm(executor.map(generate_single_event_global, args_list),
                                            total=num_events, desc="Generating events"))
        executor.shutdown(wait=True)  # Explicitly shut down the executor

        self.dataset = synthetic_dataset

    def get_dataset(self, train_val_test_split = False) -> List[Dict[str, Any]]:
        """
        Get the generated synthetic dataset.

        :param train_val_test_split: If False, return the whole dataset. If (train_percent, val_percent, test_percent) is provided, return the dataset split into training, validation, and test sets.

        :return: List of synthetic events.
        """
        if train_val_test_split:
            if not isinstance(train_val_test_split, tuple):
                raise ValueError("train_val_test_split must be a tuple of (train_percent, val_percent, test_percent).")
            if not np.isclose(sum(train_val_test_split), 1):
                raise ValueError("The sum of train_val_test_split must be 1.")
            train_percent, val_percent, test_percent = train_val_test_split
            if not all([0 <= x <= 1 for x in train_val_test_split]):
                raise ValueError("train_val_test_split values must be between 0 and 1.")
            num_events = len(self.dataset)
            train_num = int(train_percent * num_events)
            val_num = int(val_percent * num_events)
            test_num = num_events - train_num - val_num
            dataset = self.dataset.copy()
            random.shuffle(dataset)
            return dataset[:train_num], dataset[train_num:train_num+val_num], dataset[-test_num:]
        return self.dataset
    
    def get_type_map(self) -> Dict[str, int]:
        """
        Get the mapping of particle types to integer labels.

        :return: Dictionary mapping particle types to integer labels.
        """
        return self.particle_types_map
    
    def get_masses(self) -> Dict[str, float]:
        """
        Get the masses of the particles.

        :return: Dictionary mapping particle types to masses.
        """
        return {ptype: self.distributions[ptype]['mass'] for ptype in self.particle_types}
    
    def prune_centers(self, keep_probability: float) -> None:
        """
        Prune the centers of the rings based on a keep_probability.

        :param keep_probability: Probability of keeping a center (0-1).
        """
        for event in self.dataset:
            new_x_centers, new_y_centers, new_momentums, new_radii, new_particle_types = [], [], [], [], []
            new_idx_counter = 0
            old_idx_counter = 0
            old_label_new_label = {}
            for x, y, momentum, radius, p_type in zip(event['x_centers'], event['y_centers'], event['momentums'], event['radii'], event['particle_types']):
                if random.random() < keep_probability:
                    new_x_centers.append(x)
                    new_y_centers.append(y)
                    new_momentums.append(momentum)
                    new_radii.append(radius)
                    new_particle_types.append(p_type)
                    old_label_new_label[old_idx_counter] = new_idx_counter
                    new_idx_counter += 1
                else:
                    old_label_new_label[old_idx_counter] = -1
                old_idx_counter += 1

            new_labels = []
            for label in event['labels']:
                new_labels.append(old_label_new_label[label])

            event['x_centers'] = new_x_centers
            event['y_centers'] = new_y_centers
            event['momentums'] = new_momentums
            event['radii'] = new_radii
            event['labels'] = new_labels
            event['particle_types'] = new_particle_types

    def make_events_background(self, momenta_bin: tuple, rings_to_add_per_event: int) -> None:
        """
        Use current events as backgrounds and add new rings within the specified momenta bin.
        """
        for event in self.dataset:
            # Mark existing hits as background
            event['labels'] = [-1] * len(event['labels'])
            # Generate new rings
            rings = []
            new_particle_types = []
            for _ in range(rings_to_add_per_event):
                uniform_probs = np.ones(len(self.particle_types)) / len(self.particle_types)
                ptype = np.random.choice(self.particle_types, p=uniform_probs)
                x, y, R, N_points, momentum = self.sample_parameters(particle_type=ptype, momenta_dist=momenta_bin)
                rings.append((x, y, R, N_points, momentum))
                new_particle_types.append(self.particle_types_map[ptype])
            # Create ring hits
            ring_points = generateRings(rings, radial_noise=self.radial_noise)
            x_new, y_new, labels_new = ring_points[0], ring_points[1], ring_points[2]
            # Update event hits and labels
            event['x_points'] = np.concatenate((event['x_points'], x_new))
            event['y_points'] = np.concatenate((event['y_points'], y_new))
            event['labels'] = np.concatenate((event['labels'], labels_new))
            # Replace ring centers, radii, momenta
            event['x_centers'] = ring_points[4]
            event['y_centers'] = ring_points[5]
            event['momentums'] = ring_points[6]
            event['radii'] = ring_points[3]
            event['particle_types'] = new_particle_types

    def create_images(self, 
                      image_size: Tuple[int, int], 
                      save_folder: str, 
                      image_noise: float = 0.0, 
                      polar_transform: bool = False, 
                      train_val_test_split = False, 
                      use_annular_mask = True,
                      all_same_folder = False,
                      highligh_ring_pixels = False,
                      ) -> None:
        """
        Create images for each ring. The image is centered at the ring center and includes all hits within a square region extending to MAX_RADIUS.
        If use_annular_mask is True an annular mask is then applied that only keeps pixels between the lowest and highest radius hypotheses (for the various particle types).
        Each image is saved as: save_folder + '/image_' + <event_index> + '-' + <counter_for_event> + '_' + <momentum> + '_' + 'label' + '.png'
        If polar_transform is True, each image is transformed to polar coordinates with origin at the center of the ring.

        **NOTE**: The train-val-test split is refered to events, not single images.

        :param image_size: Tuple (height, width) of the image in pixels.
        :param save_folder: Folder to save the images.
        :param image_noise: Standard deviation of Gaussian noise to add to the image.
        :param polar_transform: If True, transform the image to polar coordinates.
        :param train_val_test_split: If False, save images for the whole dataset. If (train_percent, val_percent, test_percent) is provided, save images in subfolders for training, validation, and test sets.
        :param use_annular_mask: If True, apply an annular mask to the image (min_radius, max_radius). Otherwise, just crop beond maximum radius.
        :param all_same_folder: If True, all the images will be saved in the same folder, regardless of particle type. Also the train_val_test_split will be ignored.
        :param highligh_ring_pixels: If True, the pixels that belong to the ring will be colored in #669BBC.
        """
        # Ensure the main save folder exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # If train/val/test split is enabled (and not saving all in one folder), create subfolders
        if train_val_test_split and not all_same_folder:
            train, val, test = train_val_test_split
            for folder in ['train', 'val', 'test']:
                folder_path = os.path.join(save_folder, folder)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            original_save_folder = save_folder
        else:
            original_save_folder = save_folder

        # Create subfolders per particle type if needed
        if not all_same_folder:
            if not train_val_test_split:
                for ptype in self.particle_types:
                    ptype_folder = os.path.join(save_folder, str(ptype))
                    if not os.path.exists(ptype_folder):
                        os.makedirs(ptype_folder)
            else:
                for ptype in self.particle_types:
                    for folder in ['train', 'val', 'test']:
                        ptype_folder = os.path.join(original_save_folder, folder, str(ptype))
                        if not os.path.exists(ptype_folder):
                            os.makedirs(ptype_folder)

        # Precompute train/val/test assignment if needed
        num_events = len(self.dataset)
        if train_val_test_split and not all_same_folder:
            train, val, test = train_val_test_split
            event_assignment = ([0] * int(train * num_events) +
                                [1] * int(val * num_events) +
                                [2] * (num_events - int(train * num_events) - int(val * num_events)))
            random.shuffle(event_assignment)
        else:
            event_assignment = [None] * num_events

        # Build an argument list for each event
        args_list = []
        for event_idx, event in enumerate(self.dataset):
            args_list.append((
                event_idx,
                event,
                image_size,
                save_folder,
                original_save_folder,
                train_val_test_split,
                event_assignment,
                all_same_folder,
                self.radial_noise,
                self.MAX_RADIUS,
                polar_transform,
                image_noise,
                use_annular_mask,
                self.calculate_thetaC,
                self.calculate_radius,
                self.get_masses,
                self.particle_types_map,
                highligh_ring_pixels
            ))

        # Process the events in parallel using a ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # The map returns an iterator; wrap it with list() to ensure completion.
            list(tqdm(executor.map(process_event_images, args_list),
                    total=len(args_list), desc="Creating images"))

    def event_to_image(self, event_idx: int, image_size: Tuple[int, int]) -> np.ndarray:
        """
        Renders the whole event as a black and white image.
        
        The function maps the event's (x, y) hit coordinates onto a pixel grid.
        If a hit falls into a pixel, that pixel is set to white (255), and the background remains black (0).
        
        If fixed_detector_size is provided, its bounds are used for the mapping;
        otherwise, the bounds are computed from the event's hit coordinates.
        
        :param event_idx: Index of the event to render.
        :param image_size: Tuple (height, width) of the image in pixels.
        :return: A numpy array representing the rendered binary image.
        """
        event = self.dataset[event_idx]
        # Create an empty image (background = 0)
        image = np.zeros(image_size, dtype=np.uint8)

        # Convert hit coordinates to numpy arrays
        x_points = np.array(event['x_points'])
        y_points = np.array(event['y_points'])

        # Determine coordinate bounds:
        if self.fixed_detector_size is not None:
            x_min, x_max = self.fixed_detector_size[0]
            y_min, y_max = self.fixed_detector_size[1]
        else:
            # Compute bounds from the event hits; add a safeguard against zero range
            x_min, x_max = x_points.min(), x_points.max()
            y_min, y_max = y_points.min(), y_points.max()
            if x_max - x_min == 0:
                x_max = x_min + 1
            if y_max - y_min == 0:
                y_max = y_min + 1

        height, width = image_size

        # Map the (x, y) coordinates to pixel indices.
        # Here, x maps to columns and y maps to rows.
        cols = ((x_points - x_min) / (x_max - x_min) * (width - 1)).astype(int)
        rows = ((y_points - y_min) / (y_max - y_min) * (height - 1)).astype(int)

        # Ensure indices are within image bounds
        cols = np.clip(cols, 0, width - 1)
        rows = np.clip(rows, 0, height - 1)

        hit_positions = np.stack((rows, cols), axis=-1)

        # Get number of hits in each pixel
        unique_lit_cols, n_hits = np.unique(hit_positions, return_counts=True, axis=0)

        # Mark pixels with the number of hits
        image[unique_lit_cols[:, 0], unique_lit_cols[:, 1]] = n_hits

        # Flip the image vertically to match the original coordinate system
        image = np.flipud(image)

        return np.array(image)


        

    def get_pararms(self) -> Tuple[float, Dict[str, Dict[str, Any]], Dict[str, float], int, Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Get the parameters used to generate the dataset.

        :return: Tuple containing (refractive_index, distributions, type_weights, num_hits_prop_radius, fixed_detector_size).
        """
        return self.refractive_index, self.distributions, self.type_weights, self.num_hits_prop_radius, self.fixed_detector_size


    def visualize_event(self, event_idx: int, save_path: Optional[str] = None) -> None:
        """
        Visualize a single event.
    
        :param event_idx: Index of the event to visualize.
        :param save_path: Path to save the visualization. If None, the visualization is displayed.
        """
        event = self.dataset[event_idx]
        plt.figure(figsize=(10, 10))

        # Assign colors to each ring
        type_map = self.get_type_map()
        colors = plt.cm.tab20.colors
        type_colors = {ptype: colors[i % len(colors)] for ptype, i in type_map.items()}
        label_to_ptype = {v: k for k, v in type_map.items()}
        
        # Build a color map for each ring
        ring_color_map = {}
        for ring_idx, ptype_label in enumerate(event['particle_types']):
            ptype = label_to_ptype[ptype_label]
            base_rgb = type_colors[ptype]
            h, l, s = colorsys.rgb_to_hls(*base_rgb)
            l_shift = random.uniform(-0.1, 0.1)
            l_new = min(max(l + l_shift, 0), 1)
            new_rgb = colorsys.hls_to_rgb(h, l_new, s)
            ring_color_map[ring_idx]  = new_rgb

        # Assign colors to each point based on its ring
        default_color = (0.5, 0.5, 0.5)  # Gray color for points without a ring
        scatter_colors = [ring_color_map.get(label, default_color) for label in event['labels']]

        # Plot the points
        scatter = plt.scatter(event['x_points'], event['y_points'], c=scatter_colors, s=10)

        # Plot ring centers
        id_to_particle_type = {v: k for k, v in type_map.items()}
        center_colors = [type_colors[id_to_particle_type[event['particle_types'][idx]]] for idx in range(len(event['x_centers']))]
        plt.scatter(event['x_centers'], event['y_centers'], c=center_colors, marker='x', s=50)

        plt.title(f'Event {event_idx}')
        
        # Set fixed detector size if specified
        if self.fixed_detector_size is not None:
            plt.xlim(self.fixed_detector_size[0])
            plt.ylim(self.fixed_detector_size[1])
    
        # Set aspect ratio to be equal
        plt.gca().set_aspect('equal', adjustable='box')
    
        # Create a legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=ptype, markerfacecolor=type_colors[ptype], markersize=10) for ptype in type_map.keys()]
        plt.legend(handles=legend_elements, title='Particle Type', loc='upper right')
    
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def save_dataset(self, path: str) -> None:
        """
        Save the generated synthetic dataset as an HDF5 file.

        :param path: Path to save the dataset. Do not include the extension.
        """
        import h5py
        
        with h5py.File(f"{path}.h5", 'w') as f:
            # Save metadata
            meta = f.create_group('metadata')
            meta.attrs['refractive_index'] = self.refractive_index
            meta.attrs['num_events'] = len(self.dataset)
            meta.attrs['num_hits_prop_radius'] = self.num_hits_prop_radius
            meta.attrs['MAX_RADIUS'] = float(self.MAX_RADIUS)
            
            # Save particle types and weights - fix string encoding
            particle_types_ascii = [str(ptype).encode('ascii') for ptype in self.particle_types]
            meta.create_dataset('particle_types', data=np.array(particle_types_ascii, dtype=h5py.special_dtype(vlen=bytes)))
            meta.create_dataset('type_weights', data=self.type_weights)
            
            # Save particle types map with proper string encoding
            particle_types_map_data = []
            for k, v in self.particle_types_map.items():
                particle_types_map_data.append((str(k).encode('ascii'), v))
            meta.create_dataset('particle_types_map', data=np.array(particle_types_map_data, 
                              dtype=[('name', h5py.special_dtype(vlen=bytes)), ('id', 'i4')]))
            
            # Save particle masses with proper string encoding
            masses_data = []
            for ptype in self.particle_types:
                masses_data.append((str(ptype).encode('ascii'), float(self.distributions[ptype]['mass'])))
            meta.create_dataset('particle_masses', data=np.array(masses_data,
                             dtype=[('name', h5py.special_dtype(vlen=bytes)), ('mass', 'f8')]))
            
            # Save fixed detector size if it exists
            if self.fixed_detector_size is not None:
                meta.create_dataset('fixed_detector_size', data=np.array(self.fixed_detector_size))
            
            # Save radial noise if it exists
            if hasattr(self, 'radial_noise'):
                meta.create_dataset('radial_noise', data=self.radial_noise)
            
            # Save events
            events_group = f.create_group('events')
            for i, event in enumerate(self.dataset):
                event_group = events_group.create_group(f'event_{i}')
                for key, value in event.items():
                    if isinstance(value, list):
                        event_group.create_dataset(key, data=np.array(value))
                    else:
                        event_group.create_dataset(key, data=value)
        
        print(f"Dataset saved to {path}.h5")

    @classmethod
    def load_dataset(cls, path: str) -> 'SyntheticCherenkovDataGenerator':
        """
        Load a saved dataset from an HDF5 file and return a SyntheticCherenkovDataGenerator instance.
        
        :param path: Path to the saved dataset (without extension).
        :return: A SyntheticCherenkovDataGenerator instance containing the loaded dataset
        """
        import h5py
        
        with h5py.File(f"{path}.h5", 'r') as f:
            # Load metadata
            meta = f['metadata']
            refractive_index = meta.attrs['refractive_index']
            num_hits_prop_radius = meta.attrs['num_hits_prop_radius']
            max_radius = meta.attrs['MAX_RADIUS']
            
            # Load particle types and weights with proper decoding
            particle_types = [s.decode('ascii') for s in meta['particle_types'][:]]
            type_weights = dict(zip(particle_types, meta['type_weights'][:]))
            
            # Load particle masses
            masses_data = meta['particle_masses'][:]
            particle_masses = {name.decode('ascii'): float(mass) for name, mass in masses_data}
            
            # Load fixed detector size if it exists
            fixed_detector_size = None
            if 'fixed_detector_size' in meta:
                fixed_detector_size = meta['fixed_detector_size'][:]
            
            # Create minimal distributions with just masses
            distributions = {}
            for ptype in particle_types:
                distributions[ptype] = {'mass': particle_masses[ptype]}
            
            # Create the generator instance
            generator = cls(
                refractive_index=refractive_index,
                distributions=distributions,
                type_weights=type_weights,
                num_hits_prop_radius=num_hits_prop_radius,
                fixed_detector_size=fixed_detector_size
            )
            
            # Set MAX_RADIUS to the saved value
            generator.MAX_RADIUS = max_radius
            
            # Load radial noise if it exists
            if 'radial_noise' in meta:
                generator.radial_noise = meta['radial_noise'][:]
            
            # Load events
            events = []
            events_group = f['events']
            for i in range(meta.attrs['num_events']):
                event = {}
                event_group = events_group[f'event_{i}']
                for key in event_group.keys():
                    event[key] = event_group[key][:]
                events.append(event)
            
            # Set dataset attribute
            generator.dataset = events
            
            # Load particle types map with proper string decoding
            if 'particle_types_map' in meta:
                particle_types_map_data = meta['particle_types_map'][:]
                generator.particle_types_map = {name.decode('ascii'): int(id) for name, id in particle_types_map_data}
            else:
                # Create from particle types if not available
                generator.particle_types_map = {ptype: i for i, ptype in enumerate(particle_types)}
            
            return generator


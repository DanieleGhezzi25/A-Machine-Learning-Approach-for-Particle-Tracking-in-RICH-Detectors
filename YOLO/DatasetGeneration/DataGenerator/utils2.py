import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import awkward as ak
import pickle as pkl
import seaborn as sns
from scipy.stats import  gaussian_kde
import json
import os
import torch

# ---- MATH UTILITIES ----
def shift_clip(x: torch.Tensor, shift: int, dim: int, fill_value=0):
    """
    Roll-like shift along `dim` but drop elements that would wrap around.
    Keeps autograd and works on CPU / CUDA.
    """
    if shift == 0:
        return x                      # view: no allocation
    size = x.size(dim)
    # Destination tensor already pre‑filled with the desired padding value
    out = x.new_full(x.shape, fill_value)
    if shift > 0:                     # positive => move “right / down”
        out.narrow(dim,  shift, size - shift)\
           .copy_(x.narrow(dim, 0,      size - shift))
    else:                             # negative => move “left / up”
        shift = -shift
        out.narrow(dim, 0,      size - shift)\
           .copy_(x.narrow(dim, shift,  size - shift))
    return out



# ---- DATA PROCESSING UTILITIES ----

class Batch:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.num_batches = len(data) // batch_size if len(data) % batch_size == 0 else len(data) // batch_size + 1
        self.current_batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch < self.num_batches:
            start = self.current_batch * self.batch_size
            end = (self.current_batch + 1) * self.batch_size if self.current_batch + 1 <= self.num_batches else len(self.data)
            self.current_batch += 1
            return self.data[start:end]
        else:
            raise StopIteration


# ---- PLOTTING UTILITIES ----

def Hist2D(x, y, xlab='', ylab='', bins_num=400, figsize=None, **kwargs):
    # awkward to numpy if needed
    X = x.to_numpy() if type(x) == ak.highlevel.Array else x
    Y = y.to_numpy() if type(y) == ak.highlevel.Array else y
    X = np.array(X) if type(X) == list else X
    Y = np.array(Y) if type(Y) == list else Y

    # Calculate the plot range and adjust the number of bins for square bins
    xwid = (X.max() - X.min())
    xmin = X.min() - 0.05 * xwid
    xmax = X.max() + 0.05 * xwid
    ywid = (Y.max() - Y.min())
    ymin = Y.min() - 0.05 * ywid
    ymax = Y.max() + 0.05 * ywid

    # Determine the number of bins for x and y to make them square
    aspect_ratio = (xmax - xmin) / (ymax - ymin)
    if aspect_ratio > 1:
        bins_x = bins_num
        bins_y = int(bins_num / aspect_ratio)
    else:
        bins_y = bins_num
        bins_x = int(bins_num * aspect_ratio)

    # Plot
    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()
    kwargs['norm'] = mpl.colors.Normalize(vmin=0)
    kwargs['cmap'] = 'hot'
    ax.set_aspect('equal', adjustable='box')
    h = ax.hist2d(X, Y, cmin=1, range=[[xmin, xmax], [ymin, ymax]], bins=(bins_x, bins_y), **kwargs)
    fig.colorbar(h[3], ax=ax, extend='max')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    return fig, ax

def Hist2DForImage(x, y, x_range=None, y_range=None, dpi=100, bins_num=400, **kwargs):
    # Convert awkward arrays to numpy if needed
    X = x.to_numpy() if type(x) == ak.highlevel.Array else x
    Y = y.to_numpy() if type(y) == ak.highlevel.Array else y
    X = np.array(X) if type(X) == list else X
    Y = np.array(Y) if type(Y) == list else Y

    # Determine the plot range
    if x_range is not None:
        xmin = x_range[0]
        xmax = x_range[1]
    else:
        xwid = (X.max() - X.min())
        xmin = X.min() - 0.05 * xwid
        xmax = X.max() + 0.05 * xwid
    if y_range is not None:
        ymin = y_range[0]
        ymax = y_range[1]
    else:
        ywid = (Y.max() - Y.min())
        ymin = Y.min() - 0.05 * ywid
        ymax = Y.max() + 0.05 * ywid

    # Determine the number of bins for x and y to maintain square bins
    aspect_ratio = (xmax - xmin) / (ymax - ymin)
    if aspect_ratio > 1:
        bins_x = bins_num
        bins_y = int(bins_num / aspect_ratio)
    else:
        bins_y = bins_num
        bins_x = int(bins_num * aspect_ratio)

    # Calculate the figure size in inches (each bin corresponds to one pixel)
    if not dpi == None:
        figsize_x = bins_x / dpi
        figsize_y = bins_y / dpi
    else:
        figsize_x = 10
        figsize_y = 10

    # Plot
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y), dpi=dpi)
    kwargs['norm'] = mpl.colors.Normalize(vmin=0)
    kwargs['cmap'] = 'gray'
    h = ax.hist2d(X, Y, bins=(bins_x, bins_y), range=[[xmin, xmax], [ymin, ymax]], **kwargs)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    fig.patch.set_visible(False)

    return fig, ax

def generate_histogram_array(x, y, x_range, y_range, image_size=(1024, 1024)):
    """
    Generates a 2D histogram array from x and y data, sized according to the specified image dimensions.
    
    Parameters:
    x (array-like): Input data for the x-axis.
    y (array-like): Input data for the y-axis.
    image_size (tuple): Desired image size (width, height) in pixels, which determines the resolution of the histogram.
    x_range (tuple): The range of x-axis values.
    y_range (tuple): The range of y-axis values.
    
    Returns:
    numpy.ndarray: The 2D histogram array with dimensions equal to image_size.
    """
    # Compute 2D histogram with number of bins set by image_size
    hist, xedges, yedges = np.histogram2d(x, y, bins=(image_size[0], image_size[1]), range=[x_range, y_range])
    
    # Transpose the histogram to correct the orientation
    return hist.T

# ---- MODIFIED FUNCTIONS ----
import numpy as np
from scipy.stats import gaussian_kde


class TruncatedKDE(gaussian_kde):
    """
    KDE that returns zero density and rejects samples outside
    user-supplied per-dimension bounds.

    Parameters
    ----------
    dataset : (d, N) array_like
        The data used to train the KDE (same format as gaussian_kde).
    bw_method : {None, 'scott', 'silverman', scalar} or callable, optional
        Passed straight to gaussian_kde.
    truncate_range : None | (a, b) | sequence[(a_i, b_i)]
        • None → no truncation.  
        • 1-D → (a, b) or [(a, b)].  
        • 2-D → [(a1, b1), (a2, b2)]  (length must equal `d`).
    """

    def __init__(self, dataset, bw_method=None, truncate_range=None):
        dataset = np.atleast_2d(dataset)
        super().__init__(dataset, bw_method=bw_method)

        if truncate_range is None:
            self.truncate_range = None
        else:
            # Normalise to list-of-tuples length d
            if self.d == 1 and len(truncate_range) == 2 and not isinstance(
                truncate_range[0], (list, tuple)
            ):
                truncate_range = [truncate_range]
            if len(truncate_range) != self.d:
                raise ValueError(
                    f"truncate_range must have {self.d} intervals "
                    f"(got {len(truncate_range)})"
                )
            self.truncate_range = [tuple(r) for r in truncate_range]


    def __call__(self, x, *args, **kwargs):
        """Evaluate the density; zero it outside the bounds."""
        result = super().__call__(x, *args, **kwargs)

        if self.truncate_range is not None:
            x = np.atleast_2d(x)
            mask = np.ones(x.shape[1], dtype=bool)
            for dim, (a, b) in enumerate(self.truncate_range):
                mask &= (x[dim] >= a) & (x[dim] <= b)
            result[~mask] = 0.0

        return result

    def resample(self, size=1, *, max_iter=10_000):
        """
        Draw `size` samples obeying the truncation bounds (if any).
        Returns an array with shape (d, size), like gaussian_kde.resample.
        """
        if self.truncate_range is None:
            return super().resample(size)

        accepted = []
        remaining = size
        iterations = 0

        while remaining > 0:
            iterations += 1
            if iterations > max_iter:
                raise RuntimeError(
                    "Maximum iterations exceeded while rejection-sampling. "
                    "The truncation region may be extremely small."
                )

            # Oversample a bit to cut down on loops
            batch = super().resample(int(remaining * 2))
            mask = np.ones(batch.shape[1], dtype=bool)
            for dim, (a, b) in enumerate(self.truncate_range):
                mask &= (batch[dim] >= a) & (batch[dim] <= b)

            if np.any(mask):
                accepted.append(batch[:, mask])
                remaining = size - sum(arr.shape[1] for arr in accepted)

        samples = np.hstack(accepted)[:, :size]  # Trim excess
        return samples


# ---- SAVING AND LOADING UTILITIES ----

def save_particle_distributions(distributions, filename):
    """
    Save particle distributions to a JSON file without using pickle.
    
    Args:
        distributions: Dictionary of particle distributions
        filename: Path to save the JSON file
    """
    # Create a JSON-serializable dictionary
    serialized = {}
    
    for particle_id, data in distributions.items():
        particle_dict = {
            'mass': float(data['mass'])
        }
        
        # Handle center KDE
        if 'center' in data and data['center'] is not None:
            kde_obj = data['center']
            particle_dict['center'] = {
                'type': 'gaussian_kde',
                'dataset': kde_obj.dataset.tolist(),
                'dimensionality': int(kde_obj.d),
                'factor': float(kde_obj.factor)
            }
        
        # Handle N_points
        if 'N_points' in data:
            particle_dict['N_points'] = int(data['N_points'])
        
        # Handle momentum TruncatedKDE
        if 'momentum' in data and data['momentum'] is not None:
            tkde_obj = data['momentum']
            print(tkde_obj)
            particle_dict['momentum'] = {
                'type': 'truncated_kde',
                'dataset': tkde_obj.dataset.tolist(),
                'truncate_range': list(tkde_obj.truncate_range),
                'dimensionality': int(tkde_obj.d),
                'factor': float(tkde_obj.factor),
            }
        
        serialized[str(particle_id)] = particle_dict
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(serialized, f, indent=2)
    
    print(f"Particle distributions saved to {filename}")

def load_particle_distributions_json(filename):
    """
    Load particle distributions from a JSON file without using pickle.
    
    Args:
        filename: Path to the JSON file
    """
    with open(filename, 'r') as f:
        return json.load(f)

def reconstruct_kde_from_data(kde_data):
    """
    Reconstruct a scipy.stats.gaussian_kde object from raw data.
    """
    
    dataset = np.array(kde_data['dataset'])
    dimensionality = kde_data['dimensionality']
    
    # Reshape dataset if needed
    if dimensionality > 1:
        dataset = dataset.reshape(dimensionality, -1)
    
    kde = gaussian_kde(dataset)
    kde.factor = kde_data['factor']
    return kde

def reconstruct_truncated_kde_from_data(tkde_data):
    """
    Reconstruct a TruncatedKDE object from raw data.
    """
    
    dataset = np.array(tkde_data['dataset'])
    truncate_range = tuple(tkde_data['truncate_range'])
    dimensionality = tkde_data['dimensionality']
    
    if dimensionality > 1:
        dataset = dataset.reshape(dimensionality, -1)
    
    kde = TruncatedKDE(dataset)
    kde.truncate_range = truncate_range
    kde.factor = tkde_data['factor']
    kde.d = dimensionality
    return kde

def reconstruct_particle_distributions(raw_data):
    """
    Reconstruct full particle distributions from raw data.
    """
    result = {}
    
    for particle_id_str, data in raw_data.items():
        particle_id = int(particle_id_str)
        particle_dict = {
            'mass': data['mass']
        }
        
        # Reconstruct center KDE if present
        if 'center' in data and data['center'] is not None:
            particle_dict['center'] = reconstruct_kde_from_data(data['center'])
        
        # Copy N_points if present
        if 'N_points' in data:
            particle_dict['N_points'] = data['N_points']
        
        # Reconstruct momentum TruncatedKDE if present
        if 'momentum' in data and data['momentum'] is not None:
            particle_dict['momentum'] = reconstruct_truncated_kde_from_data(data['momentum'])
        
        result[particle_id] = particle_dict
    
    return result

def load_particle_distributions(filename):
    """
    Load particle distributions from a JSON file and reconstruct them.
    
    Args:
        filename: Path to the JSON file
    """
    raw_data = load_particle_distributions_json(filename)
    return reconstruct_particle_distributions(raw_data)


# OLD (USING PICKLE) -------------------------------
# Save and load graphs
def save_graph(fig, filename):
    with open(filename, 'wb') as f:
        pkl.dump(fig, f)

def load_graph(filename):
    try:
        with open(filename, 'rb') as f:
            fig = pkl.load(f)
        return fig
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None


def save_kde(kde, filename):
    """
    Save a gaussian_kde or TruncatedKDE instance to an .npz file.

    Parameters
    ----------
    kde : gaussian_kde or TruncatedKDE
        The KDE object to save.
    filename : str
        Path (including “.npz”) where the object will be written.
    """
    # Determine if this is a TruncatedKDE (has attribute `truncate_range`)
    is_truncated = hasattr(kde, "truncate_range") and (kde.truncate_range is not None)

    # Extract the original dataset (shape: (d, N))
    dataset = kde.dataset

    # Extract a numeric bandwidth factor so that reconstruction is identical.
    bw = float(kde.covariance_factor())

    # Prepare the “is_truncated” flag as 0/1
    flag = np.array([1], dtype=np.uint8) if is_truncated else np.array([0], dtype=np.uint8)

    if is_truncated:
        # Convert truncate_range (list of tuples) into a 2D array of shape (d, 2)
        tr = np.asarray(kde.truncate_range, dtype=float)  # shape = (d, 2)
        np.savez(
            filename,
            dataset=dataset,
            bw_method=bw,
            is_truncated=flag,
            truncate_range=tr
        )
    else:
        # No truncation: still save an empty placeholder for truncate_range
        tr = np.empty((0, 2), dtype=float)
        np.savez(
            filename,
            dataset=dataset,
            bw_method=bw,
            is_truncated=flag,
            truncate_range=tr
        )


def load_kde(filename):
    """
    Load a gaussian_kde or TruncatedKDE instance from an .npz file written by save_kde().

    Parameters
    ----------
    filename : str
        Path to the .npz file created by save_kde().

    Returns
    -------
    kde : gaussian_kde or TruncatedKDE
        A reconstructed KDE object, exactly matching the saved one.
    """
    data = np.load(filename, allow_pickle=False)

    dataset = data["dataset"]              # shape = (d, N)
    bw = float(data["bw_method"])          # scalar bandwidth factor
    is_trunc = bool(int(data["is_truncated"][0]))

    if is_trunc:
        tr_array = data["truncate_range"]   # shape = (d, 2)
        # Convert back to a list of (a, b) tuples
        tr_list = [tuple(row) for row in tr_array]
        kde = TruncatedKDE(dataset, bw_method=bw, truncate_range=tr_list)
    else:
        kde = gaussian_kde(dataset, bw_method=bw)

    return kde

# Save generic
def save_object(obj, filename):
    with open(filename, 'wb') as f:
        pkl.dump(obj, f)

def load_object(filename):
    try:
        with open(filename, 'rb') as f:
            obj = pkl.load(f)
        return obj
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None

# Datasets (not the object but the pure set)
def save_dataset_values(dataset, filename):
    with open(filename, 'wb') as f:
        pkl.dump(dataset, f)

def load_dataset_values(filename):
    try:
        with open(filename, 'rb') as f:
            dataset = pkl.load(f)
        return dataset
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
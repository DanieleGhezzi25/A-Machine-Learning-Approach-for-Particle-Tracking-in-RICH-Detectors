import numpy as np
import math

def generateRings(rings, radial_noise=None):
    """
    Generates Cherenkov rings given a list of tuples (x, y, R, N_points) where:
        - x, y: coordinates of the center of the ring
        - R: radius of the ring
        - N_points: number of points to generate in the ring

    The output are three lists: one with the x coordinates, one with the y coordinates of the points in the rings and the two labels for each point: the ring index and the radius of the ring.

    The generator also takes into account noise parameters:
        - radial noise: adds a random value in the radial direction to the points in the ring. The noise parameter is a tuple (mean, std), for normal distribution.
    """
    x_points = []
    y_points = []
    labels = []
    radial_labels = []
    x_centers = []
    y_centers = []
    momentums = []

    for j, ring in enumerate(rings):
        if len(ring) == 5:
            x, y, R, N_points, momentum = ring
        else:
            x, y, R, N_points = ring

        if R > 0:
            for i in range(N_points):
                # angular coordinate in ring is random
                theta = np.random.random() * 2 * math.pi

                x_i = x + R * math.cos(theta)
                y_i = y + R * math.sin(theta)
                if radial_noise is not None:
                    x_i += np.random.normal(radial_noise[0], radial_noise[1])
                    y_i += np.random.normal(radial_noise[0], radial_noise[1])
                x_points.append(x_i)
                y_points.append(y_i)
                labels.append(j)
        
        radial_labels.append(R)
        x_centers.append(x)
        y_centers.append(y)
        if len(ring) == 5:
            momentums.append(momentum)

    if momentums != []:
        return x_points, y_points, labels, radial_labels, x_centers, y_centers, momentums
    return x_points, y_points, labels, radial_labels, x_centers, y_centers


# ----------- Utility functions for dataset generation -----------

def generateRingsInPlane(n_rings, x_min, x_max, y_min, y_max, R, n_points_per_ring = 50, radial_noise=None, sample_distribution=None):
    """
    Generates a number (n_rings) of rings in a plane with the following parameters:
        - x_min, x_max: range of the x coordinate of the center of the rings
        - y_min, y_max: range of the y coordinate of the center of the rings
        - R_min, R_max: range of the radius of the rings
        - n_points_per_ring: number of points to generate in each ring. It can be an integer or a tuple (min, max) for a random number of points in each ring between min and max.
        - radial_noise: tuple (mean, std) for the radial noise
        - sample_distribution: custom distribution function to sample x and y from. It should return a tuple (x, y) with the coordinates of the center of the ring.

    Note: the points that are generated outside the plane are discarded.
    """
    if type(n_points_per_ring) == int:
        N_points = [n_points_per_ring] * n_rings
    else:
        if len(n_points_per_ring) == 2:
            N_points = np.random.randint(n_points_per_ring[0], n_points_per_ring[1], n_rings)
        else:
            raise ValueError("n_points_per_ring should be an integer or a tuple of two integers")
        
    use_discrete_R = False
    if type(R) == int:
        R_min = R
        R_max = R
    elif len(R) == 2 and type(R) == tuple:
        R_min = R[0]
        R_max = R[1]
    elif len(R) > 2:
        use_discrete_R = True

    rings = []
    for i in range(n_rings):
        if sample_distribution == None:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            if use_discrete_R: # sample R from the list of radii
                R_value = np.random.choice(R)
            else:
                R_value = np.random.uniform(R_min, R_max)
        else: # sample x and y from the custom distribution function
            x, y = sample_distribution()
            if use_discrete_R:
                R_value = np.random.choice(R)
            else:
                R_value = np.random.uniform(R_min, R_max)

        rings.append((x, y, R_value, N_points[i]))

    generated_rings = generateRings(rings, radial_noise)
    # Discard points outside the plane
    x_points, y_points, labels, radial_labels, x_centers, y_centers = [], [], [], [], [], []
    for i in range(len(generated_rings[0])):
        if x_min <= generated_rings[0][i] <= x_max and y_min <= generated_rings[1][i] <= y_max:
            x_points.append(generated_rings[0][i])
            y_points.append(generated_rings[1][i])
            labels.append(generated_rings[2][i])

    for i in range(len(generated_rings[4])):
        x_centers.append(generated_rings[4][i])
        y_centers.append(generated_rings[5][i])
        radial_labels.append(generated_rings[3][i])


    return x_points, y_points, labels, radial_labels, x_centers, y_centers


#momentums can either be a list of len n_rings or a tuple (min, max) for random momentum values sampled from a uniform distribution
def generateRingsInPlaneMomentum(n_rings, x_min, x_max, y_min, y_max, R, momentums, n_points_per_ring = 50, radial_noise=None, sample_distribution=None):
    if type(n_points_per_ring) == int:
        N_points = [n_points_per_ring] * n_rings
    else:
        if len(n_points_per_ring) == 2:
            N_points = np.random.randint(n_points_per_ring[0], n_points_per_ring[1], n_rings)
        else:
            raise ValueError("n_points_per_ring should be an integer or a tuple of two integers")
        
    use_discrete_R = False
    if type(R) == int:
        R_min = R
        R_max = R
    elif len(R) == 2 and type(R) == tuple:
        R_min = R[0]
        R_max = R[1]
    elif len(R) > 2:
        use_discrete_R = True

    use_momnentum_discrete = False
    if type(momentums) == int:
        momentum_min = momentums
        momentum_max = momentums
    elif len(momentums) == 2 and type(momentums) == tuple:
        momentum_min = momentums[0]
        momentum_max = momentums[1]
    elif len(momentums) > 2:
        use_momnentum_discrete = True

    rings = []
    for i in range(n_rings):
        if sample_distribution == None:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)

        else: # sample x and y from the custom distribution function
            x, y = sample_distribution()
        
        if use_discrete_R: # sample R from the list of radii
            R_value = np.random.choice(R)
        else:
            R_value = np.random.uniform(R_min, R_max)
        if use_momnentum_discrete:
            momentum = np.random.choice(momentums)
        else:
            momentum = np.random.uniform(momentum_min, momentum_max)

        rings.append((x, y, R_value, N_points[i], momentum))

    generated_rings = generateRings(rings, radial_noise)

    # Discard points outside the plane
    x_points, y_points, labels, radial_labels, x_centers, y_centers, momentums = [], [], [], [], [], [], []
    for i in range(len(generated_rings[0])):
        if x_min <= generated_rings[0][i] <= x_max and y_min <= generated_rings[1][i] <= y_max:
            x_points.append(generated_rings[0][i])
            y_points.append(generated_rings[1][i])
            labels.append(generated_rings[2][i])

    for i in range(len(generated_rings[4])):
        radial_labels.append(generated_rings[3][i])
        x_centers.append(generated_rings[4][i])
        y_centers.append(generated_rings[5][i])
        momentums.append(generated_rings[6][i])


    return x_points, y_points, labels, radial_labels, x_centers, y_centers, momentums
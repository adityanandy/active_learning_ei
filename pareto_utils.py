import numpy as np


def is_pareto(points):
    """
    Find the Pareto-efficient points.

    Parameters
    ----------
    points : np.array
        An (n_points, n_costs) array of points, from which the Pareto-efficient points are identified.

    Returns
    -------
    pareto_inds : np.array
        The indices of the input that correspond to Pareto-efficient points.
    pareto_points : np.array
        The Pareto-efficient points.

    """    
    is_efficient = np.ones(points.shape[0], dtype=bool)
    # Start off with every point considered as Pareto-efficient.
    # Loop through all points. For each point X that is still considered Pareto-efficient, check all other points to see if they are better than point X in any target property.
    # If another point is not better than point X in any target property, it is no longer considered Pareto-efficient.
    for i, c in enumerate(points):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(points[is_efficient] < c, axis=1)  # Keep any point with a lower cost. axis=1 means this is done over the columns, so running this on a 3x2 array will lead to a 3 entry result
            is_efficient[i] = True
    pareto_inds = np.where(is_efficient == True)[0]
    pareto_points = points[pareto_inds]
    args = pareto_points[:, 0].argsort() # Ordering for sorting. Only considers the first cost
    pareto_inds = pareto_inds[args] # Sorting the Pareto indices
    pareto_points = pareto_points[args] # Sorting the Pareto points by cost
    return pareto_inds, pareto_points


def pareto_nearest(point, paretoset):
    """
    Calculate the nearest point in the Pareto set to the provided point.

    Parameters
    ----------
    point : list or np.array
        The coordinates of the point of interest. The nearest point in the Pareto set to this point will be found.
    paretoset : list of np.array objects
        The points of the Pareto set.

    Returns
    -------
    min_point : np.array
        The point in the Pareto set that is closest to the provided point.

    """
    min_point = False
    min_distance = False
    for ps in paretoset:
        dist = np.linalg.norm(np.array(point) - ps)
        if min_distance == False or dist < min_distance:
            min_point = ps
            min_distance = dist
    return min_point


def area_under_pareto(pareto_points, global_pareto):
    """
    # Calculate the area under the Pareto front.
    # This is somewhat related to equation 13 of https://arc.aiaa.org/doi/pdf/10.2514/1.16875. See also Figure 7 of that paper.

    Parameters
    ----------
    pareto_points : list of lists
        Each inner list is a Pareto point.
    global_pareto : list of lists
        The global Pareto points. These would not be known a priori for a new problem. Each inner list is a Pareto point.

    Returns
    -------
    area : float
        The area under the Pareto front.

    """
    p_last = global_pareto[-1]
    area = (pareto_points[0][0] - global_pareto[0][0]) * (global_pareto[0][1] - p_last[1])
    
    # Area for the edges.
    for ii in range(pareto_points.shape[0] - 1):
        p = pareto_points[ii]
        p_ip1 = pareto_points[ii + 1]
        area += (p_ip1[0] - p[0]) * (p[1] - p_last[1])
    
    area += (-pareto_points[-1][0] + global_pareto[-1][0]) * (pareto_points[-1][1] - p_last[1])
    return area

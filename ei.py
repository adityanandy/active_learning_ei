import numpy as np
from scipy.stats import norm
from perato_utils import pareto_nearest
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans
from pyclustering.cluster.kmedoids import kmedoids


def getEiVec2D_aug(df, approx_pareto_list):
    """
    Inserts EI information into a DataFrame.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame to append to. Columns as hat_yi: predicted yi, sigma_yi: uncertainty std of y1.
    approx_pareto_list : np.array
        A (n_points, n_costs) array.

    Returns
    -------
    df : Pandas DataFrame
        The updated DataFrame, with EI, PI, y1a, and y2a information.

    """
    ei_vec, pi_vec, y1aug_vec, y2aug_vec = [], [], [], []
    for idx, row in df.iterrows(): # Iterate over the rows of the DataFrame.
        if row['known']:
            y1aug_vec.append(0)
            y2aug_vec.append(0)
            ei_vec.append(0)
            pi_vec.append(0)
        else:
            pI, yaug1, yaug2, ei = get_ei_related(mu1=row['hat_y1'], s1=row['sigma_y1'],
                                                  mu2=row['hat_y2'], s2=row['sigma_y2'],
                                                  paretoset=approx_pareto_list)
            pi_vec.append(pI)
            y1aug_vec.append(yaug1)
            y2aug_vec.append(yaug2)
            ei_vec.append(ei)
    df['ei'] = ei_vec
    df['pi'] = pi_vec
    df['y1a'] = y1aug_vec
    df['y2a'] = y2aug_vec
    return df


def getPiVec2D_aug(df, approx_pareto_list):
    """
    Inserts PI information into a DataFrame.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame to append to. Columns as hat_yi: predicted yi, sigma_yi: uncertainty std of y1.
    approx_pareto_list : np.array
        A (n_points, n_costs) array.

    Returns
    -------
    df : Pandas DataFrame
        The updated DataFrame, with PI information.

    """
    pi_vec = []
    for idx, row in df.iterrows(): # Iterating over rows
        if row['known']:
            pi_vec.append(0)
        else:
            pi_vec.append(pi_2Daug_keane(mu1=row['hat_y1'], s1=row['sigma_y1'],
                                         mu2=row['hat_y2'], s2=row['sigma_y2'],
                                         paretoset=approx_pareto_list))
    df['pi'] = pi_vec
    return df


def get_ei_related(mu1, s1, mu2, s2, paretoset):
    """
    Get the PI (probability of improvement) and EI (expectation of improvement) of a 2D point described by mu1, s1, mu2, and s2. 
    The 2D point thus has a mean and standard deviation in each of the two dimensions.

    The PI calculation follows equation 14 of https://arc.aiaa.org/doi/pdf/10.2514/1.16875; also the function pi_2Daug_keane.
    The centroid calculation follows equation 18 of https://arc.aiaa.org/doi/pdf/10.2514/1.16875

    Parameters
    ----------
    mu1 : float
        The mean of the new point in the first dimension.
    s1 : float
        The standard deviation (uncertainty) of the new point in the first dimension.
    mu2 : float
        The mean of the new point in the second dimension.
    s2 : float
        The standard deviation (uncertainty) of the new point in the second dimension.
    paretoset : np.array
        The 2-dimensional points that make up the paretoset. Its shape should be (number_of_pareto_points, 2).

    Returns
    -------
    pI : float
        The probability of improvement of the 2D point described by mu1, s1, mu2, and s2.
    yaug1 : float
        The centroid of the P[I] integral in the first dimension.
    yaug2 : float
        The centroid of the P[I] integral in the second dimension.
    eI : float
        The expectation of improvement of the 2D point described by mu1, s1, mu2, and s2.

    """

    imp1s = (paretoset[0, 0] - mu1) / s1 # imp stands for improvement
    imp2s = (paretoset[0, 1] - mu2) / s2
    pI = norm.cdf(imp1s) # The cumulative distribution function of the normal distribution.
    yaug1 = mu1 * norm.cdf(imp1s) - s1 * norm.pdf(imp1s) # yaug stands for y_augmented
    yaug2 = mu2 * norm.cdf(imp1s)

    for i in range(paretoset.shape[0] - 1): # This loop occurs number_of_pareto_points times.
        point = paretoset[i] # Grab one point from the Pareto set.
        point_p = paretoset[i + 1] # Grab the subsequent point from the Pareto set.

        imp1s_ip1 = (point_p[0] - mu1) / s1
        imp2s_ip1 = (point_p[1] - mu2) / s2
        imp1s_i = (point[0] - mu1) / s1
        imp2s_i = (point[1] - mu2) / s2
        pI += (norm.cdf(imp1s_ip1) - norm.cdf(imp1s_i)) * norm.cdf(imp2s_i)
        yaug1 += ((mu1 * norm.cdf(imp1s_ip1) - s1 * norm.pdf(imp1s_ip1)) - (
                mu1 * norm.cdf(imp1s_i) - s1 * norm.pdf(imp1s_i))) * norm.cdf(imp2s_i)
        yaug2 += (mu2 * norm.cdf(imp2s_i) - s2 * norm.pdf(imp2s_i)) * (norm.cdf(imp1s_ip1) - norm.pdf(imp1s_i))

    imp1s_last = (paretoset[-1, 0] - mu1) / s1
    imp2s_last = (paretoset[-1, 1] - mu2) / s2
    pI += (1 - norm.cdf(imp1s_last)) * norm.cdf(imp2s_last)
    yaug1 += (mu1 * norm.cdf(imp1s_last) + s1 * norm.pdf(imp1s_last)) * norm.cdf(imp2s_last)
    yaug2 += (mu2 * norm.cdf(imp2s_last) - s2 * norm.pdf(imp2s_last)) * (1 - norm.cdf(imp1s_last))
    yaug1 /= max(pI, 1e-8) # I assume this is to prevent divide by zero.
    yaug2 /= max(pI, 1e-8)

    # See page 886 of https://arc.aiaa.org/doi/pdf/10.2514/1.16875
    # The set member closest to the centroid.
    nearest_pareto = pareto_nearest(point=[yaug1, yaug2], paretoset=paretoset) 

    # Equation 16 of https://arc.aiaa.org/doi/pdf/10.2514/1.16875
    ei = pI * np.sqrt((yaug1 - nearest_pareto[0]) ** 2 + (yaug2 - nearest_pareto[1]) ** 2)

    return pI, yaug1, yaug2, ei


def get_ei_samples_kmedoids(df, fnames, points_per_gen):
    """
    Get k-medoids points from the inputted DataFrame.

    Parameters
    ----------
    df : Pandas DataFrame
        The original DataFrame.
    fnames : list of str
        The names of the features used to predict for the target property.
    points_per_gen : int
        The initial number of medoids.

    Returns
    -------
    new_points : Pandas DataFrame
        The k-medoids points.

    """
    initial_medoids = np.random.choice(len(df), points_per_gen) # Randomly choosing #points_per_gen many medoids
    kmedoids_instance = kmedoids(df[fnames].values, initial_medoids)
    kmedoids_instance.process()
    tm = kmedoids_instance.get_medoids()
    medoids, count = [], 0
    for idx, row in df.iterrows(): # Iterate over the rows of the DataFrame.
        if count in tm:
            medoids.append(True)
        else:
            medoids.append(False)
        count += 1
    df['medoids'] = medoids
    new_points = df[df['medoids'] == True] # Filter the DataFrame for the new points.
    return new_points


def get_ei_samples_kmeans(df, fnames, points_per_gen):
    """
    Get k-means points from the inputted DataFrame.

    Parameters
    ----------
    df : Pandas DataFrame
        The original DataFrame.
    fnames : list of str
        The names of the features used to predict for the target property.
    points_per_gen : int
        The initial number of k-means points.

    Returns
    -------
    new_points : Pandas DataFrame
        The k-means points.

    """
    kmeans = KMeans(n_clusters=points_per_gen, random_state=0).fit(df[fnames].values)
    tree = BallTree(kmeans.cluster_centers_, leaf_size=2, metric='cityblock') # BallTree gets nearest neighbors
    dist_mat, inds = tree.query(df[fnames].values, k=1)
    dist_mat = dist_mat.squeeze()
    inds = inds.squeeze()
    _df = df.copy()
    _df['dist'] = dist_mat
    _df['ind'] = inds
    for x in range(points_per_gen):
        __df = _df[_df['ind'] == x]
        if x:
            new_points = new_points.append(__df[__df['dist'] == np.min(__df['dist'].values)])
        else:
            new_points = __df[__df['dist'] == np.min(__df['dist'].values)]
    return new_points


### Note, pi_2Daug_keane is an older implementation. The function get_ei_related calculates PI but also calculates EI. PI is much cheaper to calculate than EI.
def pi_2Daug_keane(mu1, s1, mu2, s2, paretoset):
    """
    Get the PI (probability of improvement) of a 2D point described by mu1, s1, mu2, and s2. 
    The 2D point thus has a mean and standard deviation in each of the two dimensions.

    Follows equation 14 of https://arc.aiaa.org/doi/pdf/10.2514/1.16875

    Get the probability of improvement. m: predicted value. s: std. paretoset: pareto front.

    Parameters
    ----------
    mu1 : float
        The mean of the new point in the first dimension.
    s1 : float
        The standard deviation (uncertainty) of the new point in the first dimension.
    mu2 : float
        The mean of the new point in the second dimension.
    s2 : float
        The standard deviation (uncertainty) of the new point in the second dimension.
    paretoset : np.array
        The 2-dimensional points that make up the paretoset. Its shape should be (number_of_pareto_points, 2).

    Returns
    -------
    pI : float
        The probability of improvement of the 2D point described by mu1, s1, mu2, and s2.

    """

    # Dealing with the first point in the Pareto set.
    imp1s = (paretoset[0, 0] - mu1) / s1 # imp stands for improvement
    imp2s = (paretoset[0, 1] - mu2) / s2
    pI = norm.cdf(imp1s) # The cumulative distribution function of the normal distribution.

    # The summation part of the equation 14 of https://arc.aiaa.org/doi/pdf/10.2514/1.16875
    for i in range(paretoset.shape[0] - 1): # This loop occurs number_of_pareto_points times.
        point = paretoset[i] # Grab one point from the Pareto set.
        point_p = paretoset[i + 1] # Grab the subsequent point from the Pareto set.

        imp1s_ip1 = (point_p[0] - mu1) / s1
        imp2s_ip1 = (point_p[1] - mu2) / s2
        imp1s_i = (point[0] - mu1) / s1
        imp2s_i = (point[1] - mu2) / s2
        pI += (norm.cdf(imp1s_ip1) - norm.cdf(imp1s_i)) * norm.cdf(imp2s_i)

    # Dealing with the last point in the Pareto set.
    imp1s_last = (paretoset[-1, 0] - mu1) / s1
    imp2s_last = (paretoset[-1, 1] - mu2) / s2
    pI += (1 - norm.cdf(imp1s_last)) * norm.cdf(imp2s_last)
    return pI

# The functions ybar1_2Daug_keane and ybar2_2Daug_keane combine with ei_2Daug_keane to calculate EI.
# But they do it less efficiently than the function get_ei_related.

def ybar1_2Daug_keane(mu1, s1, mu2, s2, paretoset):
    # Follows equation 18 of https://arc.aiaa.org/doi/pdf/10.2514/1.16875
    imp1s = (paretoset[0, 0] - mu1) / s1
    imp2s = (paretoset[0, 1] - mu2) / s2
    yaug = mu1 * norm.cdf(imp1s) - s1 * norm.pdf(imp1s)

    for i in range(paretoset.shape[0] - 1):
        point = paretoset[i]
        point_p = paretoset[i + 1]

        imp1s_ip1 = (point_p[0] - mu1) / s1
        imp2s_ip1 = (point_p[1] - mu2) / s2
        imp1s_i = (point[0] - mu1) / s1
        imp2s_i = (point[1] - mu2) / s2
        yaug += ((mu1 * norm.cdf(imp1s_ip1) - s1 * norm.pdf(imp1s_ip1)) - (
                mu1 * norm.cdf(imp1s_i) - s1 * norm.pdf(imp1s_i))) * norm.cdf(imp2s_i)

    imp1s_last = (paretoset[-1, 0] - mu1) / s1
    imp2s_last = (paretoset[-1, 1] - mu2) / s2
    yaug += (mu1 * norm.cdf(imp1s_last) + s1 * norm.pdf(imp1s_last)) * norm.cdf(imp2s_last)
    pI = pi_2Daug_keane(mu1, s1, mu2, s2, paretoset)
    pI = max(pI, 1e-8)
    yaug = yaug / pI
    return yaug


def ybar2_2Daug_keane(mu1, s1, mu2, s2, paretoset):
    # Follows equation 18 of https://arc.aiaa.org/doi/pdf/10.2514/1.16875
    imp1s = (paretoset[0, 0] - mu1) / s1
    imp2s = (paretoset[0, 1] - mu2) / s2
    yaug = mu2 * norm.cdf(imp2s) - s2 * norm.pdf(imp2s)

    for i in range(paretoset.shape[0] - 1):
        point = paretoset[i]
        point_p = paretoset[i + 1]

        imp1s_ip1 = (point_p[0] - mu1) / s1
        imp2s_ip1 = (point_p[1] - mu2) / s2
        imp1s_i = (point[0] - mu1) / s1
        imp2s_i = (point[1] - mu2) / s2
        yaug += ((mu2 * norm.cdf(imp2s_ip1) - s2 * norm.pdf(imp2s_ip1)) - (
                mu2 * norm.cdf(imp2s_i) - s2 * norm.pdf(imp2s_i))) * norm.cdf(imp1s_i)

    imp1s_last = (paretoset[-1, 0] - mu1) / s1
    imp2s_last = (paretoset[-1, 1] - mu2) / s2
    yaug += (mu2 * norm.cdf(imp2s_last) + s2 * norm.pdf(imp2s_last)) * norm.cdf(imp1s_last)
    pI = pi_2Daug_keane(mu1, s1, mu2, s2, paretoset)
    pI = max(pI, 1e-8)
    yaug = yaug / pI
    return yaug


def ei_2Daug_keane(mu1, s1, mu2, s2, paretoset, y1aug, y2aug):
    # Follows equation 16 of https://arc.aiaa.org/doi/pdf/10.2514/1.16875
    nearest_pareto = pareto_nearest(point=[y1aug, y2aug], paretoset=paretoset)
    ei = pi_2Daug_keane(mu1, s1, mu2, s2, paretoset) * np.sqrt(
        (ybar1_2Daug_keane(mu1, s1, mu2, s2, paretoset) - nearest_pareto[0]) ** 2
        + (ybar2_2Daug_keane(mu1, s1, mu2, s2, paretoset) - nearest_pareto[1]) ** 2)
    return ei
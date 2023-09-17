from typing import List

import cv2
import numpy as np

def homogenize(cartesian: np.ndarray) -> np.ndarray:
    """Transform n points (2, n) into their homogeneous coordinate 
    form of (3, n).
    """
    if len(cartesian.shape) == 1:
        return np.append(cartesian, 1.0).astype(np.float32)
    else:
        homogenized = np.ones((3, cartesian.shape[1]), dtype=np.float32)
        homogenized[:2] = cartesian
        return homogenized

def point_to_point(
    p1: np.ndarray, 
    p2: np.ndarray,
    ) -> np.ndarray:
    """Calculate a point-to-point task for p1 and p2:
        T_{pp} = p_2 - p_1
    """
    return p2 - p1

def parallel_line(
    p1: np.ndarray, 
    p2: np.ndarray,
    p3: np.ndarray, 
    p4: np.ndarray,
    ):
    """Calculate a parallel-line task as:
        E_{par} = (p_1 x p_2) x (p_3 x p_4)
    """
    # To homogeneous coords
    p1_hom = homogenize(p1)
    p2_hom = homogenize(p2)
    p3_hom = homogenize(p3)
    p4_hom = homogenize(p4)

    l12 = np.cross(p1_hom, p2_hom)
    l34 = np.cross(p3_hom, p4_hom)
    epar = np.cross(l12, l34)
    return np.expand_dims(epar[-1], 0)

def PCA(
    mask,
    scale_axis1: float = 0.01,
    scale_axis2: float = 0.5,
    ):
    """Perform PCA over binary image.
    """
    points = cv2.findNonZero(mask).sum(axis=1).astype(np.float32)

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean)

    # Store the center of the object
    cntr = np.array([int(mean[0, 0]), int(mean[0, 1])])

    # Major
    p1 = (cntr[0] + scale_axis1 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + scale_axis1 * eigenvectors[0,1] * eigenvalues[0,0])

    # Minor
    p2 = (cntr[0] - scale_axis2 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - scale_axis2 * eigenvectors[1,1] * eigenvalues[1,0])
    
    return cntr.astype(np.float32), np.array(p1).astype(np.float32), np.array(p2).astype(np.float32)

def pca_geometric_constraint_from_mask_simple(
    mask: np.ndarray,
    constraint: str,
    ):
    h, w = mask.shape[:2]

    # Simple PCA
    f_point, f_line, _ = PCA(mask)

    e = None
    kps = []
    if constraint == "p2p":
        p1 = np.float32(np.array([w//2, 4*h//5]))
        e = point_to_point(p1, f_point)
        kps = [p1, f_point]

    elif constraint == "par":
        p1 = np.float32(np.array([w//2, 5*h//8]))
        p2 = np.float32(np.array([w//2, 6*h//8]))
        e = parallel_line(p1, p2, f_point, f_line)
        kps = [p1, p2, f_point, f_line]

    else:
        raise NotImplementedError

    return np.round(e, 5), kps
        
def visualize_constraint_plt(
    kps: List[np.ndarray],
    constraint: str,
    ax,
    ):
    if "p2p" in constraint:
        color = (50/255,205/255,50/255)
        p1, p2 = kps[0], kps[1]

        ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], color=color, marker='.', s=100, edgecolor=color, linewidth=1.25, label="Point-to-point")

    elif "p2l" in constraint:
        color = (0, 1, 1)
        p1, p2, p3 = kps[0].astype(int), kps[1].astype(int), kps[2].astype(int)

        ax.scatter([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], color=color, marker='.', s=100, edgecolor=color, linewidth=1.25, label="Point-to-line")
        ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color=color)

    elif "par" in constraint:
        color = (255/255, 144/255, 30/255)
        p1, p2, p3, p4 = kps[0].astype(int), kps[1].astype(int), kps[2].astype(int), kps[3].astype(int)

        ax.scatter([p1[0], p2[0], p3[0], p4[0]], [p1[1], p2[1], p3[1], p4[1]], color=color, marker='.', s=100, edgecolor=color, linewidth=1.25, label="Parallel-line")
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], color=color)

    elif "l2l" in constraint:
        color = (168/255, 50/255, 111/255)
        p1, p2, p3, p4 = kps[0].astype(int), kps[1].astype(int), kps[2].astype(int), kps[3].astype(int)

        ax.scatter([p1[0], p2[0], p3[0], p4[0]], [p1[1], p2[1], p3[1], p4[1]], color=color, marker='.', s=100, edgecolor=color, linewidth=1.25, label="Line-to-line")
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], color=color)
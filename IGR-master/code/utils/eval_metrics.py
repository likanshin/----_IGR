import numpy as np
import trimesh
from scipy.spatial import cKDTree


def sample_points_from_mesh(mesh, num_samples=100000):
    points, face_indices = mesh.sample(num_samples, return_index=True)
    normals = mesh.face_normals[face_indices]
    return points, normals


def compute_chamfer_distance(points1, points2):
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    dist1, _ = tree2.query(points1, k=1)
    dist2, _ = tree1.query(points2, k=1)
    
    chamfer_dist = (np.mean(dist1) + np.mean(dist2)) / 2.0
    return chamfer_dist


def compute_f_score(points1, points2, threshold=0.01):
    tree2 = cKDTree(points2)
    tree1 = cKDTree(points1)
    
    dist1, _ = tree2.query(points1, k=1)
    dist2, _ = tree1.query(points2, k=1)
    
    precision = np.mean(dist1 < threshold)
    recall = np.mean(dist2 < threshold)
    
    if precision + recall < 1e-8:
        return 0.0
    
    f_score = 2 * precision * recall / (precision + recall)
    return f_score


def compute_normal_consistency(points1, normals1, points2, normals2):
    tree2 = cKDTree(points2)
    _, nn_indices = tree2.query(points1, k=1)
    
    normals2_nn = normals2[nn_indices]
    
    cos_sim = np.abs(np.sum(normals1 * normals2_nn, axis=1))
    
    return np.mean(cos_sim)


def compute_all_metrics(pred_mesh_path, gt_mesh_path, num_samples=100000, f_threshold=0.01):
    pred_mesh = trimesh.load(pred_mesh_path, process=False)
    gt_mesh = trimesh.load(gt_mesh_path, process=False)
    
    pred_points, pred_normals = sample_points_from_mesh(pred_mesh, num_samples)
    gt_points, gt_normals = sample_points_from_mesh(gt_mesh, num_samples)
    
    cd = compute_chamfer_distance(pred_points, gt_points)
    
    f_score = compute_f_score(pred_points, gt_points, threshold=f_threshold)
    
    nc = compute_normal_consistency(pred_points, pred_normals, gt_points, gt_normals)
    
    metrics = {
        'chamfer_distance': float(cd),
        'f_score': float(f_score),
        'normal_consistency': float(nc),
        'num_samples': num_samples,
        'f_threshold': f_threshold
    }
    
    return metrics

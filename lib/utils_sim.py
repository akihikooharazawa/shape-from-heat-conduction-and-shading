from tqdm import tqdm
from pathlib import Path

import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

import igl
import trimesh
from skimage.restoration import inpaint
from scipy.ndimage import median_filter
from scipy.optimize import minimize
import numpy as np

device = torch.device("cuda:0")

import os
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
import sys
sys.path.append(workspace_path)

#  -------------- Load the mesh, center it and scale such max abs coordinate is unity -------------- 
def create_reg_tri_mesh_from_camera(width: int, height: int, pixel_pitch: float = None, depth_map: np.ndarray = None):
    """
    Creates a regular triangular mesh.

    Returns: (v, f)
        v: #V by 3 matrix of vertex locations
        f: #F by 3 matrix of face indices into v
    """

    assert depth_map is None, "Depth map is not supported yet"

    if not pixel_pitch:
        pixel_pitch = 1 / max(width, height)
    x_coords = np.linspace(-(width-1) * pixel_pitch, (width-1) * pixel_pitch, width)
    y_coords = np.linspace(-(height-1) * pixel_pitch, (height-1) * pixel_pitch, height)
    uu, vv = np.meshgrid(x_coords, y_coords, indexing='xy')
    if not depth_map:
        dd = np.zeros_like(uu)
    else:
        assert depth_map.shape == uu.shape, f"Given depth map has shape {depth_map.shape}, while width and height are given as {(width, height)}"
        dd = depth_map
    # vertices of the regular triangular mesh
    v = np.stack((uu.reshape(-1), vv.reshape(-1), dd.reshape(-1)), axis=1)

    # construct triangular faces
    def _ij_to_k(i, j):
        return j * width + i

    f = set()
    for j in range(height):
        for i in range(width):
            triangles = []
            if j > 0 and i < width - 1:
                tri_A = (_ij_to_k(i, j), _ij_to_k(i, j-1), _ij_to_k(i+1,j))
                triangles.append(tuple(sorted(tri_A)))
            if j < height - 1 and i < width - 1:
                tri_B = (_ij_to_k(i, j), _ij_to_k(i+1, j), _ij_to_k(i+1,j+1))
                triangles.append(tuple(sorted(tri_B)))
            if j < height - 1 and i < width - 1:
                tri_C = (_ij_to_k(i, j), _ij_to_k(i+1, j+1), _ij_to_k(i,j+1))
                triangles.append(tuple(sorted(tri_C)))
            if j < height - 1 and i > 0:
                tri_D = (_ij_to_k(i, j), _ij_to_k(i, j+1), _ij_to_k(i-1,j))
                triangles.append(tuple(sorted(tri_D)))
            if j > 0 and i > 0:
                tri_E = (_ij_to_k(i, j), _ij_to_k(i-1, j), _ij_to_k(i-1,j-1))
                triangles.append(tuple(sorted(tri_E)))
            if j > 0 and i > 0:
                tri_F = (_ij_to_k(i, j), _ij_to_k(i-1, j-1), _ij_to_k(i,j-1))
                triangles.append(tuple(sorted(tri_F)))
            f.update(triangles)
    
    f = np.asarray(list(f))
    f, _ = igl.bfs_orient(f)

    # pixel coords
    jj, ii = np.mgrid[0:height, 0:width]
    return v, f, ii.reshape(-1), jj.reshape(-1)

def get_vf_for_obj(obj_name, root_folder=None, norm=True, scale=1.0, padding=0):
    if root_folder is None:
        root_folder = os.path.join(workspace_path, "objs")
    
    filepath = Path(root_folder) / f'{obj_name}.obj'
    
    if not filepath.exists():
        objs_path = Path(root_folder)
        if objs_path.exists():
            available_files = [f.stem for f in objs_path.glob("*.obj")]
            error_msg = f"File not found for '{obj_name}'. Available files: {available_files}"
        else:
            error_msg = f"Directory not found: {root_folder}"
        raise FileNotFoundError(error_msg)
    
    verts, faces = igl.read_triangle_mesh(str(filepath))
    # print(f"Loaded mesh from {filepath}")

    if norm is True:
        center = verts.mean(0)
        verts = verts - center
        unit_scale = np.abs(verts).max()
        verts = verts / unit_scale / (1 + padding)
        
        print(f"Scaled mesh to be within coords: Min: {verts.min(0)} and Max: {verts.max(0)}")
    
    if scale != 1.0:
        verts = verts * scale

    return verts, faces

def construct_camera_mesh(obj_name, width, height, obj_zdist, return_mask=False):
    verts_np, faces_np = get_vf_for_obj(obj_name)
    assert verts_np[..., -1].max() - verts_np[..., -1].min() < obj_zdist, f"Depth of object ({verts_np[..., -1].max() - verts_np[..., -1].min()}) is larger than given obj_zdist"
    verts_np[..., -1] += obj_zdist

    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)

    # Initialize a triangular mesh
    v, f, ii, jj = create_reg_tri_mesh_from_camera(width, height)

    ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh, False)
    ray_origins = v
    ray_dirs = np.zeros_like(v)
    ray_dirs[:, -1] = 1
    locations, index_ray, index_tri = ray_mesh_intersector.intersects_location(ray_origins, ray_dirs, multiple_hits=False)
    depth_map = np.zeros(v.shape[0], float)
    depth_map[index_ray] = locations[:, 2]
    mask = np.ones_like(depth_map)
    mask[index_ray] = 0
    depth_map = depth_map.reshape((height, width))
    mask = mask.reshape((height, width))
    depth_map_filled = inpaint.inpaint_biharmonic(depth_map, mask)

    v_new = v.copy()
    v_new[:, 2] = depth_map_filled.reshape(-1)

    obj_filename = os.path.join(workspace_path, f'objs/{obj_name}_camera_{width}x{height}_{obj_zdist}.obj')
    igl.write_triangle_mesh(obj_filename, v_new, f)
    print(f"Constructed a camera mesh for {obj_name} and saved it for future reuse at {obj_filename}.")
    
    if return_mask:
        return v_new, f, mask.reshape(-1)
    else:
        return v_new, f

def compute_finite_difference_normals(verts_np, width, height, mode="fwd"):
    verts_np_ij = verts_np.reshape((height, width, 3))

    vnormals = np.zeros_like(verts_np_ij)

    if mode == "fwd":
        vnormals[:, :-1, 0] = (verts_np_ij[:, 1:, 2] - verts_np_ij[:, :-1, 2]) / (verts_np_ij[:, 1:, 0] - verts_np_ij[:, :-1, 0])
        vnormals[:, -1, 0] = vnormals[:, -2, 0]
        vnormals[:-1, :, 1] = (verts_np_ij[1:, :, 2] - verts_np_ij[:-1, :, 2]) / (verts_np_ij[1:, :, 1] - verts_np_ij[:-1, :, 1])
        vnormals[-1, :, 1] = vnormals[-2, :, 1]
        vnormals[..., 2] = -1
    elif mode == "bak":
        vnormals[:, 1:, 0] = (verts_np_ij[:, 1:, 2] - verts_np_ij[:, :-1, 2]) / (verts_np_ij[:, 1:, 0] - verts_np_ij[:, :-1, 0])
        vnormals[1:, :, 1] = (verts_np_ij[1:, :, 2] - verts_np_ij[:-1, :, 2]) / (verts_np_ij[1:, :, 1] - verts_np_ij[:-1, :, 1])
        vnormals[..., 2] = -1
    elif mode == "ctr":
        vnormals[:, 1:-1, 0] = (verts_np_ij[:, 2:, 2] - verts_np_ij[:, :-2, 2]) / (verts_np_ij[:, 2:, 0] - verts_np_ij[:, :-2, 0])
        vnormals[1:-1, :, 1] = (verts_np_ij[2:, :, 2] - verts_np_ij[:-2, :, 2]) / (verts_np_ij[2:, :, 1] - verts_np_ij[:-2, :, 1])
        vnormals[..., 2] = -1
    else:
        raise ValueError(f"Expected one of fwd, bak, or ctr. Got {mode}")
    
    # reshape and normalize
    vnormals = vnormals.reshape(-1, 3)
    vnormals = vnormals / np.linalg.norm(vnormals, axis=1, keepdims=True)
    return vnormals


# ----------------------- Shading Equation (p,q) ----------------------- 
def compute_shading_value(p, q, ps, qs):
    denom_n = np.sqrt(p**2 + q**2 + 1.0)
    denom_s = np.sqrt(ps**2 + qs**2 + 1.0)
    eta = (p*ps + q*qs + 1.0) / (denom_n * denom_s)
    eta = np.maximum(eta, 0.0)
    return eta


# ----------------------- Compute t ----------------------- 
def rotation_matrix_z_to_vec(s_norm):

    eps = 1e-8
    
    # Construct reference basis in the plane perpendicular to s_norm
    # If s_norm is nearly parallel to z-axis, use x-axis as reference instead
    ref = np.array([0., 1., 0.])
    if abs(np.dot(s_norm, ref)) > eps:
        ref = np.array([0., 0., 1.])

    # Create orthogonal basis {u, v, s_norm} using Gram-Schmidt
    # u = ref × s_norm, v = s_norm × u forms right-handed orthogonal basis
    u = np.cross(ref, s_norm)
    u = u / np.linalg.norm(u)
    v = np.cross(s_norm, u)
    
    # Construct rotation matrix R = [u, v, s_norm]
    R = np.column_stack([u, v, s_norm])
    
    return R

# p,q to t
def unit_vector_to_t(unit_vector, eta, s_vec):
    s_norm = s_vec / np.linalg.norm(s_vec)
    R = rotation_matrix_z_to_vec(s_norm)
    # Project the unit vector onto the plane orthogonal to the reference vector
    projection = unit_vector - eta * s_norm
    projection /= np.sqrt(1-eta**2)

    # Transform projection to the standard coordinate system
    # where s_norm becomes z-axis and {u,v} become {x,y} axes
    transformed_proj = R.T @ projection
    
    # Compute azimuthal angle in the XY plane
    t = math.atan2(transformed_proj[1], transformed_proj[0])
    
    # Normalize t to the range [0, 2π)
    if t < 0:
        t += 2 * np.pi
    return t

# t to p,q
def normal_components(t, eta, s_vec):
    s_norm = s_vec / np.linalg.norm(s_vec)
    
    # Construct rotation matrix as in unit_vector_to_t
    R = rotation_matrix_z_to_vec(s_norm)
    
    # Compute a point on the xy-plane in the standard coordinate system from angle t
    xy_component = np.array([np.cos(t), np.sin(t), 0.0])
    
    # Transform to the actual coordinate system
    circle_component = R @ xy_component
    
    # Compute the final normal vector
    normal = eta * s_norm + np.sqrt(1 - eta**2) * circle_component
    
    # Compute p, q
    eps = 1e-6
    if abs(normal[2]) < eps:
        return 0.0, 0.0
    
    p = normal[0] / normal[2] 
    q = normal[1] / normal[2]
    
    return p, q

## Compute t (torch)
def rotation_matrix_z_to_vec_torch(s_norm):

    eps = 1e-8
    
    # Construct reference basis in the plane perpendicular to s_norm
    # If s_norm is nearly parallel to z-axis, use x-axis as reference instead
    ref = torch.tensor([0., 1., 0.], dtype=s_norm.dtype, device=s_norm.device)
    if abs(torch.dot(s_norm, ref)) > eps:
        ref = torch.tensor([0., 0., 1.], dtype=s_norm.dtype, device=s_norm.device)

    # Create orthogonal basis {u, v, s_norm} using Gram-Schmidt
    # u = ref × s_norm, v = s_norm × u forms right-handed orthogonal basis
    u = torch.cross(ref, s_norm)
    u = u / u.norm()
    v = torch.cross(s_norm, u)
    
    # Construct rotation matrix R = [u, v, s_norm]
    R = torch.stack([u, v, s_norm], dim=1)
    
    return R

# t to p,q (torch)
def normal_components_torch(t, eta, s_vec):
    s_norm = s_vec / s_vec.norm()
    
    # Construct rotation matrix
    R = rotation_matrix_z_to_vec_torch(s_norm)
    
    # Compute points on the xy-plane in the standard coordinate system from angle t
    xy_component = torch.stack([torch.cos(t), torch.sin(t), torch.zeros_like(t)], dim=1)
    
    # Transform to the actual coordinate system
    # circle_component: shape=(N, 3)
    circle_component = torch.matmul(xy_component, R.T)  # (N,3) @ (3,3) -> (N,3)
    
    # Compute the final normal vector
    # normal: shape=(N, 3)
    eta_expanded = eta.unsqueeze(1)  # shape=(N, 1)
    s_norm_expanded = s_norm.unsqueeze(0)  # shape=(1, 3)
    sqrt_term = torch.sqrt(1 - eta**2).unsqueeze(1)  # shape=(N, 1)
    
    normal = eta_expanded * s_norm_expanded + sqrt_term * circle_component
    
    # Compute p, q with overflow protection
    eps = 1e-6
    z_component = normal[:, 2].clamp(min=eps)
    p = normal[:, 0] / z_component
    q = normal[:, 1] / z_component
    
    return p, q


# ----------------------- Cotan Laplacian -----------------------
def get_ngbr_indices(width, height):
    jj, ii = np.mgrid[1:height-1, 1:width-1]
    vertex = (jj  , ii  )
    ngbr_a = (jj-1, ii  )
    ngbr_b = (jj  , ii+1)
    ngbr_c = (jj+1, ii+1)
    ngbr_d = (jj+1, ii  )
    ngbr_e = (jj  , ii-1)
    ngbr_f = (jj-1, ii-1)
    ngbr_g = (jj-1, ii+1)
    ngbr_h = (jj+1, ii-1)
    ind_v = np.ravel_multi_index(vertex, dims=(height, width), mode='raise').reshape(-1)
    ind_a = np.ravel_multi_index(ngbr_a, dims=(height, width), mode='raise').reshape(-1)
    ind_b = np.ravel_multi_index(ngbr_b, dims=(height, width), mode='raise').reshape(-1)
    ind_c = np.ravel_multi_index(ngbr_c, dims=(height, width), mode='raise').reshape(-1)
    ind_d = np.ravel_multi_index(ngbr_d, dims=(height, width), mode='raise').reshape(-1)
    ind_e = np.ravel_multi_index(ngbr_e, dims=(height, width), mode='raise').reshape(-1)
    ind_f = np.ravel_multi_index(ngbr_f, dims=(height, width), mode='raise').reshape(-1)
    ind_g = np.ravel_multi_index(ngbr_g, dims=(height, width), mode='raise').reshape(-1)
    ind_h = np.ravel_multi_index(ngbr_h, dims=(height, width), mode='raise').reshape(-1)
    return ind_v, ind_a, ind_b, ind_c, ind_d, ind_e, ind_f, ind_g, ind_h

def cotan_laplacians_from_pq_np(pq_np, width, height):
    ind_v, ind_a, ind_b, ind_c, ind_d, ind_e, ind_f, ind_g, ind_h = get_ngbr_indices(width, height)

    # compute the cotan laplacian
    cotan_alpha_j1j2i_n = (pq_np[ind_v, 0] * (pq_np[ind_v, 0] + pq_np[ind_a, 1]) + 1) / np.sqrt(1 + pq_np[ind_a, 1]**2 + pq_np[ind_v, 0]**2)
    cotan_alpha_j2j1i_n = (pq_np[ind_a, 1] * (pq_np[ind_v, 0] + pq_np[ind_a, 1]) + 1) / np.sqrt(1 + pq_np[ind_a, 1]**2 + pq_np[ind_v, 0]**2)

    cotan_alpha_j2j3i_n = (pq_np[ind_b, 1] * (pq_np[ind_v, 0] + pq_np[ind_b, 1]) + 1) / np.sqrt(1 + pq_np[ind_b, 1]**2 + pq_np[ind_v, 0]**2)
    cotan_alpha_j3j2i_n = -pq_np[ind_b, 1] *  pq_np[ind_v, 0]                         / np.sqrt(1 + pq_np[ind_b, 1]**2 + pq_np[ind_v, 0]**2)

    cotan_alpha_j3j4i_n = -pq_np[ind_d, 0] *  pq_np[ind_v, 1]                         / np.sqrt(1 + pq_np[ind_d, 0]**2 + pq_np[ind_v, 1]**2)
    cotan_alpha_j4j3i_n = (pq_np[ind_d, 0] * (pq_np[ind_v, 1] + pq_np[ind_d, 0]) + 1) / np.sqrt(1 + pq_np[ind_d, 0]**2 + pq_np[ind_v, 1]**2)

    cotan_alpha_j4j5i_n = (pq_np[ind_e, 0] * (pq_np[ind_v, 1] + pq_np[ind_e, 0]) + 1) / np.sqrt(1 + pq_np[ind_e, 0]**2 + pq_np[ind_v, 1]**2)
    cotan_alpha_j5j4i_n = (pq_np[ind_v, 1] * (pq_np[ind_v, 1] + pq_np[ind_e, 0]) + 1) / np.sqrt(1 + pq_np[ind_e, 0]**2 + pq_np[ind_v, 1]**2)

    cotan_alpha_j5j6i_n = (pq_np[ind_f, 1] * (pq_np[ind_e, 0] + pq_np[ind_f, 1]) + 1) / np.sqrt(1 + pq_np[ind_e, 0]**2 + pq_np[ind_f, 1]**2)
    cotan_alpha_j6j5i_n = -pq_np[ind_e, 0] *  pq_np[ind_f, 1]                         / np.sqrt(1 + pq_np[ind_e, 0]**2 + pq_np[ind_f, 1]**2)

    cotan_alpha_j6j1i_n = -pq_np[ind_f, 0] *  pq_np[ind_a, 1]                         / np.sqrt(1 + pq_np[ind_a, 1]**2 + pq_np[ind_f, 0]**2)
    cotan_alpha_j1j6i_n = (pq_np[ind_f, 0] * (pq_np[ind_f, 0] + pq_np[ind_a, 1]) + 1) / np.sqrt(1 + pq_np[ind_a, 1]**2 + pq_np[ind_f, 0]**2)

    weight_i0j1 = 0.5 * (cotan_alpha_j1j6i_n + cotan_alpha_j1j2i_n)
    weight_i0j2 = 0.5 * (cotan_alpha_j2j1i_n + cotan_alpha_j2j3i_n)
    weight_i0j3 = 0.5 * (cotan_alpha_j3j2i_n + cotan_alpha_j3j4i_n)
    weight_i0j4 = 0.5 * (cotan_alpha_j4j3i_n + cotan_alpha_j4j5i_n)
    weight_i0j5 = 0.5 * (cotan_alpha_j5j4i_n + cotan_alpha_j5j6i_n)
    weight_i0j6 = 0.5 * (cotan_alpha_j6j5i_n + cotan_alpha_j6j1i_n)

    vertex_cotan_laplacians = np.stack([weight_i0j1, weight_i0j2, weight_i0j3, weight_i0j4, weight_i0j5, weight_i0j6], axis=1)
    return vertex_cotan_laplacians

def cotan_laplacians_from_pq_torch(pq_torch, width, height, device=device):
    ind_v, ind_a, ind_b, ind_c, ind_d, ind_e, ind_f, ind_g, ind_h = get_ngbr_indices(width, height)

    ind_v_torch = torch.tensor(ind_v, dtype=torch.long, device=device)
    ind_a_torch = torch.tensor(ind_a, dtype=torch.long, device=device)
    ind_b_torch = torch.tensor(ind_b, dtype=torch.long, device=device)
    ind_c_torch = torch.tensor(ind_c, dtype=torch.long, device=device)
    ind_d_torch = torch.tensor(ind_d, dtype=torch.long, device=device)
    ind_e_torch = torch.tensor(ind_e, dtype=torch.long, device=device)
    ind_f_torch = torch.tensor(ind_f, dtype=torch.long, device=device)

    p_v  = pq_torch[ind_v_torch, 0]
    q_v  = pq_torch[ind_v_torch, 1]
    p_a  = pq_torch[ind_a_torch, 0]
    q_a  = pq_torch[ind_a_torch, 1]
    p_b  = pq_torch[ind_b_torch, 0]
    q_b  = pq_torch[ind_b_torch, 1]
    p_c  = pq_torch[ind_c_torch, 0]
    q_c  = pq_torch[ind_c_torch, 1]
    p_d  = pq_torch[ind_d_torch, 0]
    q_d  = pq_torch[ind_d_torch, 1]
    p_e  = pq_torch[ind_e_torch, 0]
    q_e  = pq_torch[ind_e_torch, 1]
    p_f  = pq_torch[ind_f_torch, 0]
    q_f  = pq_torch[ind_f_torch, 1]

    cotan_alpha_j1j2i_n = (p_v * (p_v + q_a) + 1) / torch.sqrt(1 + q_a**2 + p_v**2)
    cotan_alpha_j2j1i_n = (q_a * (p_v + q_a) + 1) / torch.sqrt(1 + q_a**2 + p_v**2)

    cotan_alpha_j2j3i_n = (q_b * (p_v + q_b) + 1) / torch.sqrt(1 + q_b**2 + p_v**2)
    cotan_alpha_j3j2i_n = -(q_b * p_v)            / torch.sqrt(1 + q_b**2 + p_v**2)

    cotan_alpha_j3j4i_n = -(p_d * q_v)            / torch.sqrt(1 + p_d**2 + q_v**2)
    cotan_alpha_j4j3i_n = (p_d * (q_v + p_d) + 1) / torch.sqrt(1 + p_d**2 + q_v**2)

    cotan_alpha_j4j5i_n = (p_e * (q_v + p_e) + 1) / torch.sqrt(1 + p_e**2 + q_v**2)
    cotan_alpha_j5j4i_n = (q_v * (q_v + p_e) + 1) / torch.sqrt(1 + p_e**2 + q_v**2)

    cotan_alpha_j5j6i_n = (q_f * (p_e + q_f) + 1) / torch.sqrt(1 + p_e**2 + q_f**2)
    cotan_alpha_j6j5i_n = -(p_e * q_f)            / torch.sqrt(1 + p_e**2 + q_f**2)

    cotan_alpha_j6j1i_n = -(p_f * q_a)            / torch.sqrt(1 + q_a**2 + p_f**2)
    cotan_alpha_j1j6i_n = (p_f * (p_f + q_a) + 1) / torch.sqrt(1 + q_a**2 + p_f**2)

    weight_i0j1 = 0.5 * (cotan_alpha_j1j6i_n + cotan_alpha_j1j2i_n)
    weight_i0j2 = 0.5 * (cotan_alpha_j2j1i_n + cotan_alpha_j2j3i_n)
    weight_i0j3 = 0.5 * (cotan_alpha_j3j2i_n + cotan_alpha_j3j4i_n)
    weight_i0j4 = 0.5 * (cotan_alpha_j4j3i_n + cotan_alpha_j4j5i_n)
    weight_i0j5 = 0.5 * (cotan_alpha_j5j4i_n + cotan_alpha_j5j6i_n)
    weight_i0j6 = 0.5 * (cotan_alpha_j6j5i_n + cotan_alpha_j6j1i_n)

    vertex_cotan_laplacians_torch = torch.stack([weight_i0j1, weight_i0j2, weight_i0j3, weight_i0j4, weight_i0j5, weight_i0j6], dim=1)
    return vertex_cotan_laplacians_torch


# ----------------------- Planar Initialization ----------------------- 
# Find flat regions based on the cotan laplacian
def find_flat_regions(L):
    # Calculate sum of absolute values for each row
    sum_L = np.sum(np.abs(L), axis=1)

    # Compute diagonal differences
    diff_wij14 = (L[:, 0] - L[:, 3])**2  # (wi0j1 - wi0j4)^2
    diff_wij25 = (L[:, 1] - L[:, 4])**2  # (wi0j2 - wi0j5)^2
    diff_wij36 = (L[:, 2] - L[:, 5])**2  # (wi0j3 - wi0j6)^2
    
    # Normalize by the sum of squared values in each diagonal pair
    norm_14 = L[:, 0]**2 + L[:, 3]**2
    norm_25 = L[:, 1]**2 + L[:, 4]**2
    norm_36 = L[:, 2]**2 + L[:, 5]**2
    
    # Avoid division by zero
    norm_14 = np.maximum(norm_14, 1e-12)
    norm_25 = np.maximum(norm_25, 1e-12)
    norm_36 = np.maximum(norm_36, 1e-12)
    
    normalized_diff_14 = diff_wij14 / norm_14
    normalized_diff_25 = diff_wij25 / norm_25
    normalized_diff_36 = diff_wij36 / norm_36
    
    sum_w = normalized_diff_14 + normalized_diff_25 + normalized_diff_36
    
    # Calculate metric
    metric = sum_w / sum_L
    
    # Sort based on metric
    sorted_idx = np.argsort(metric)
    sorted_L = L[sorted_idx]
    
    return sorted_idx, sorted_L

def compute_p_q_from_w(w1, w2, w3):

    D = w1 + w2 + 2*w3
    discriminant = D**2 - 4
    
    if discriminant < 0:
        return 0, 0, 1
    
    x1 = (D + np.sqrt(discriminant)) / 2
    x2 = (D - np.sqrt(discriminant)) / 2
    x = max(x1, x2)
    
    if x <= 0:
        return 0, 0, 1

    p_squared = (w1 + w3) * x - 1
    q_squared = (w2 + w3) * x - 1
    
    if p_squared < 0 or q_squared < 0:
        return 0, 0, 1

    p = np.sqrt(p_squared)
    q = np.sqrt(q_squared)

    if w3 * x < 0:
        return -p, q, x
    else:
        return p, q, x

def compute_x0_from_w_common_sol(w, x0_prev):

    w1 = (w[0]+w[3])/2.0
    w2 = (w[1]+w[4])/2.0
    w3 = (w[2]+w[5])/2.0

    p, q, x = compute_p_q_from_w(w1, w2, w3)
    
    if p is None or q is None:
        x0 = np.random.uniform(-1, 1, 8)
        scale = 1
        return x0, scale
    
    x0_new = np.array([p, q, q, p, p, q, q, p])
    return x0_new, None

def extract_candidates(id_planar, L_planar, eta_2d, width, TOP_K=None):
    x0_prev = np.zeros(8, dtype=float)
    subset_init = []
    scale = None
    selected_coords = []

    for rank in range(len(id_planar)):
        if (TOP_K is not None) and (len(subset_init) >= TOP_K):
            break

        i = id_planar[rank]
        row_in = (i // (width - 2)) + 1
        col_in = (i %  (width - 2)) + 1

        # 1) Omit shading pixels
        eta_in = eta_2d[row_in, col_in]
        if eta_in <= 1e-3:
            continue

        # 2) Check if the pixel is too close to already selected pixels (5x5 neighborhood)
        skip = False
        for (r_sel, c_sel) in selected_coords:
            if abs(row_in - r_sel) <= 2 and abs(col_in - c_sel) <= 2:
                skip = True
                break
        if skip:
            continue

        # 3) Compute candidate parameters from w_vals
        l_in = L_planar[rank]
        x0, scale = compute_x0_from_w_common_sol(l_in, x0_prev)
        p, q = x0[4], x0[5]

        # 4) Ensure that the subset is not flat
        subset_init_pq = (i, p, q, p, q, p, q, p, q, p, q, p, q)
        if not any(val != 0 for val in subset_init_pq[1:]):
            continue

        # Register the pixel as selected
        subset_init.append(subset_init_pq)
        selected_coords.append((row_in, col_in))
        x0_prev = x0

    return subset_init, scale

# ADDITIONAL VERSION THAT RETURNS ALL SOLUTIONS
def compute_p_q_from_w_v2(w):

    w1 = (w[0]+w[3])/2.0
    w2 = (w[1]+w[4])/2.0
    w3 = (w[2]+w[5])/2.0

    D = w1 + w2 + 2*w3
    disc = D**2 - 4
    
    if disc < 0:
        return []

    x1 = (D + np.sqrt(disc)) / 2
    x2 = (D - np.sqrt(disc)) / 2
    x = max(x1, x2)
    
    p_sq = (w1 + w3) * x - 1
    q_sq = (w2 + w3) * x - 1
    
    if p_sq < 0 or q_sq < 0:
        return []
    
    p = np.sqrt(p_sq)
    q = np.sqrt(q_sq)

    solutions = []
    if w3 < 0:
        solutions = [(p, q), (-p, -q)]
    else:
        solutions = [(p, -q), (-p, q)]
    return solutions

def compute_p_q_from_w_v3(w):

    W1, W2, W3, W4, W5, W6 = w.T

    # Planar approximation
    w1 = (W1 + W4) / 2.0
    w2 = (W2 + W5) / 2.0
    w3 = (W3 + W6) / 2.0

    D = w1 + w2 + 2.0 * w3 # D should be >= 2
    disc = D**2 - 4.0 # Disc should be >= 0 (some regions have neg. disc due to numerical errors)
    
    N = w.shape[0]
    p1, q1, p2, q2 = (np.full(N, np.nan) for _ in range(4))

    # Check valid discriminant
    valid = (disc >= 0.0) & (D >= 2.0)
    if not np.any(valid):
        return (p1, q1), (p2, q2)

    sqrt_disc = np.sqrt(disc[valid])
    Dv = D[valid]

    x1 = (Dv + sqrt_disc) / 2.0
    x2 = (Dv - sqrt_disc) / 2.0
    x  = np.maximum(x1, x2)

    w1v, w2v, w3v = w1[valid], w2[valid], w3[valid]
    p_sq = (w1v + w3v) * x - 1.0
    q_sq = (w2v + w3v) * x - 1.0

    valid2 = (p_sq >= 0.0) & (q_sq >= 0.0)
    if not np.any(valid2):
        return (p1, q1), (p2, q2)

    idx = np.where(valid)[0][valid2]

    P = np.sqrt(p_sq[valid2])
    Q = np.sqrt(q_sq[valid2])

    sgn_w3 = np.where(w3v[valid2] >= 0.0, 1.0, -1.0)

    p1[idx], p2[idx] =  P, -P
    q1[idx], q2[idx] = -sgn_w3 * Q,  sgn_w3 * Q

    # # Debug prints
    # print("disc min/max:", disc.min(), disc.max())
    # print("valid count:", np.count_nonzero(valid), "/", valid.size)

    # # p_sq, q_sq
    # print("p_sq min/max:", p_sq.min(), p_sq.max())
    # print("q_sq min/max:", q_sq.min(), q_sq.max())
    # print("valid2 count:", np.count_nonzero(valid2), "/", valid2.size)

    # # Intermediate values
    # print("w1+w3 min/max:", (w1v+w3v).min(), (w1v+w3v).max())
    # print("w2+w3 min/max:", (w2v+w3v).min(), (w2v+w3v).max())
    # print("x min/max:", x.min(), x.max())

    return (p1, q1), (p2, q2)


# -------------- RPS: Reverse Photometric Stereo -------------- 
def reverse_photometric_stereo(pq_array, eta_array):

    eps=1e-6
    denom = np.sqrt(pq_array[:,0]**2 + pq_array[:,1]**2 + 1.0)

    A = np.column_stack([pq_array[:,0],
                         pq_array[:,1],
                         np.ones_like(denom)])
    B = eta_array * denom
    X, *_ = np.linalg.lstsq(A, B, rcond=None)
    ps_est, qs_est, rs_est = X
    ps_est = ps_est / (rs_est + eps)
    qs_est = qs_est / (rs_est + eps)

    residual = np.linalg.norm(B - A.dot(X))**2

    return ps_est, qs_est, residual


# -------------- Sign correction --------------
def determine_pq_signs_per_point_improved(pq_array, eta_array):

    p = pq_array[:, 0]
    q = pq_array[:, 1]
    
    p_signs = np.ones(pq_array.shape[0])
    q_signs = np.ones(pq_array.shape[0])
    
    for i in range(pq_array.shape[0]):

        # 1) ps, qs = 1, 0
        eta_p_A = compute_shading_value( p[i], q[i], 1, 0)
        eta_p_B = compute_shading_value(-p[i], q[i], 1, 0)
        err_p_A = np.abs(eta_p_A - eta_array[i])
        err_p_B = np.abs(eta_p_B - eta_array[i])
        p_sign = 1 if err_p_A <= err_p_B else -1

        # 2) ps, qs = 0, 1
        eta_q_A = compute_shading_value(p[i], -q[i], 0, 1)
        eta_q_B = compute_shading_value(p[i],  q[i], 0, 1)
        err_q_A = np.abs(eta_q_A - eta_array[i])
        err_q_B = np.abs(eta_q_B - eta_array[i])
        q_sign = 1 if err_q_A <= err_q_B else -1
        
        p_signs[i] = p_sign
        q_signs[i] = q_sign
    
    return p_signs, q_signs


# -------------- Frankot-Chelappa --------------
# src: https://mywavepy.readthedocs.io/en/latest/_modules/wavepy/surface_from_grad.html#frankotchellappa
def frankotchellappa(del_f_del_x, del_f_del_y, reflec_pad=True):
    """

    The simplest method is the so-called Frankot-Chelappa method. The idea
    behind this method is to search (calculate) an integrable gradient field
    that best fits the data. Luckly, Frankot Chelappa were able in they article
    to find a simple single (non interective) equation for that. We are
    even luckier since this equation makes use of FFT, which is very
    computationally efficient.

    Considering a signal :math:`s(x,y)`
    with differential signal given by
    :math:`s_x = s_x(x,y) = \\frac{\\partial s (x, y)}{\\partial x}` and
    :math:`s_y = s_y(x,y) = \\frac{\\partial s (x, y)}{\\partial y}`. The
    Fourier Transform of :math:`s_x` and :math:`s_y` are given by

    .. math::
            \\mathcal{F} \\left [ s_x \\right ] =
            \\mathcal{F} \\left [ s_x \\right ] (f_x, f_y) =
            \\mathcal{F} \\left [ \\frac{\\partial s (x, y)}
            {\\partial x} \\right ](f_x, f_y), \\quad
            \\mathcal{F} \\left [ s_y \\right ] =
            \\mathcal{F} \\left [ s_y \\right ] (f_x, f_y) =
            \\mathcal{F} \\left [ \\frac{\\partial s (x, y)}
            {\\partial y} \\right ](f_x, f_y).



    Finally, Frankot-Chellappa method is base in solving the following
    equation:


    .. math::
            \\mathcal{F} \\left [ s \\right ] = \\frac{-i f_x \\mathcal{F}
            \\left [ s_x \\right ] - i f_y \\mathcal{F} \\left [ s_y
            \\right ]}{2 \\pi (f_x^2 + f_y^2 )}

    where

    .. math::
            \\mathcal{F} \\left [ s \\right ] = \\mathcal{F} \\left[
            s(x, y) \\right ] (f_x, f_y)

    is the Fourier Transform of :math:`s`.

    To avoid the singularity in the denominator, it is added
    :py:func:`numpy.finfo(float).eps`, the smallest float number in the
    machine.

    Keep in mind that Frankot-Chelappa is not the best method. More advanced
    alghorithms are available, where it is used more complex math and
    interactive methods. Unfortunatelly, these algorothims are only available
    for MATLAB.


    References
    ----------

        :cite:`Frankot88`. The padding we use here is :cite:`Huang15`.




    Parameters
    ----------

    del_f_del_x, del_f_del_y : ndarrays
        2 dimensional gradient data

    reflec_pad: bool
       This flag pad the gradient field in order to obtain a 2-dimensional
       reflected function. See more in the Notes below.

    Returns
    -------
    ndarray
        Integrated data, as provided by the Frankt-Chellappa Algorithm. Note
        that the result are complex numbers. See below


    Notes
    -----


    * Padding

        Frankt-Chellappa makes intensive use of the Discrete Fourier
        Transform (DFT), and due to the periodicity property of the DFT, the
        result of the integration will also be periodic (even though we
        only get one period of the answer). This property can result in a
        discontinuity at the edges, and Frankt-Chellappa method is badly
        affected by discontinuity,

        In this sense the idea of this padding is that by reflecting the
        function at the edges we avoid discontinuity. This was inspired by
        the code of the function
        `DfGBox
        <https://www.mathworks.com/matlabcentral/fileexchange/45269-dfgbox>`_,
        available in the MATLAB File Exchange website.

        Note that, since we only have the gradient data, we need to consider
        how a reflection at the edges will affect the partial derivatives. We
        show it below without proof (but it is easy to see).

        First lets consider the data for the :math:`x` direction derivative
        :math:`\\Delta_x = \\dfrac{\\partial f}{\\partial x}` consisting of a
        2D array of size :math:`N \\times M`. The padded matrix
        will be given by:

        .. math::
            \\left[
            \\begin{matrix}
              \\Delta_x(x, y) & -\\Delta_x(N-x, y) \\\\
              \\Delta_x(x, M-y) & -\\Delta_x(N-x, M-y)
            \\end{matrix}
            \\right]

        and for the for the y direction derivative
        :math:`\\Delta_y = \\dfrac{\\partial f}{\\partial y}` we have

        .. math::
            \\left[
            \\begin{matrix}
              \\Delta_y(x, y) & \\Delta_y(N-x, y) \\\\
              -\\Delta_y(x, M-y) & -\\Delta_y(N-x, M-y)
            \\end{matrix}
            \\right]

        Note that this padding increases the number of points from
        :math:`N \\times M` to :math:`2M \\times 2N`. However, **the function
        only returns the** :math:`N \\times M` **result**, since the
        other parts are only a repetion of the result. In other words,
        the padding is done only internally.


    * Results are Complex Numbers

        Again due to the use of DFT's, the results are complex numbers.
        In principle an ideal gradient field of real numbers results
        a real-only result. This "imaginary noise" is observerd
        even with theoretical functions, which leads to the conclusion
        that it is due to a numerical noise. It is left to the user
        to decide what to do with noise, for instance to use the
        modulus or the real part of the result. But it
        is recomended to use the real part.



    """

    from numpy.fft import fft2, ifft2, fftfreq

    if reflec_pad:
        del_f_del_x, del_f_del_y = _reflec_pad_grad_fields(del_f_del_x,
                                                           del_f_del_y)

    NN, MM = del_f_del_x.shape
    wx, wy = np.meshgrid(fftfreq(MM) * 2 * np.pi,
                         fftfreq(NN) * 2 * np.pi, indexing='xy')
    # by using fftfreq there is no need to use fftshift

    numerator = -1j * wx * fft2(del_f_del_x) - 1j * wy * fft2(del_f_del_y)

    denominator = (wx) ** 2 + (wy) ** 2 + np.finfo(float).eps

    res = ifft2(numerator / denominator)
    res -= np.mean(np.real(res))

    if reflec_pad:
        return _one_forth_of_array(res)
    else:
        return res

def _reflec_pad_grad_fields(del_func_x, del_func_y):
    """

    This fucntion pad the gradient field in order to obtain a 2-dimensional
    reflected function. The idea is that, by having an reflected function,
    we avoid discontinuity at the edges.


    This was inspired by the code of the function DfGBox, available in the
    MATLAB File Exchange website:
    https://www.mathworks.com/matlabcentral/fileexchange/45269-dfgbox

    """

    del_func_x_c1 = np.concatenate((del_func_x,
                                    del_func_x[::-1, :]), axis=0)

    del_func_x_c2 = np.concatenate((-del_func_x[:, ::-1],
                                    -del_func_x[::-1, ::-1]), axis=0)

    del_func_x = np.concatenate((del_func_x_c1, del_func_x_c2), axis=1)

    del_func_y_c1 = np.concatenate((del_func_y,
                                    -del_func_y[::-1, :]), axis=0)

    del_func_y_c2 = np.concatenate((del_func_y[:, ::-1],
                                    -del_func_y[::-1, ::-1]), axis=0)

    del_func_y = np.concatenate((del_func_y_c1, del_func_y_c2), axis=1)

    return del_func_x, del_func_y

def _one_forth_of_array(array):
    """
    Undo for the function
    :py:func:`wavepy:surface_from_grad:_reflec_pad_grad_fields`

    """

    array, _ = np.array_split(array, 2, axis=0)
    return np.array_split(array, 2, axis=1)[0]

# -------------- Refinement --------------
# Objective function for refinement
def obj_function(x, idx_i, idx_a, idx_b, idx_d, idx_e, idx_f, i_v, lap_gt, width, height):

    p_i, q_i, p_a, q_a, p_b, q_b, p_d, q_d, p_e, q_e, p_f, q_f = x

    # Laplacian error
    pq_modified = np.zeros((height*width, 2), dtype=np.float64)
    pq_modified[idx_i, :] = [p_i, q_i]
    pq_modified[idx_a, :] = [p_a, q_a]
    pq_modified[idx_b, :] = [p_b, q_b]
    pq_modified[idx_d, :] = [p_d, q_d]
    pq_modified[idx_e, :] = [p_e, q_e]
    pq_modified[idx_f, :] = [p_f, q_f]
    
    # Laplacian error
    lap_hat = cotan_laplacians_from_pq_np(pq_modified, width, height)
    diff_lap = lap_gt[i_v] - lap_hat[i_v]
    E_lap = np.mean(diff_lap**2)
    
    return E_lap

def detect_bad_pixels(image, window_size=3, threshold_factor=3):
    """
    Detect bad pixels using median-based outlier detection.
    
    Parameters:
    - image: Input image array
    - window_size: Size of neighborhood window (odd number)
    - threshold_factor: How many standard deviations to use for thresholding
    
    Returns:
    - mask: Binary mask of detected bad pixels
    """

    
    # Calculate local median
    median_img = median_filter(image, size=window_size)
    
    # Calculate absolute difference from median
    diff = np.abs(image - median_img)
    
    # Calculate threshold (can use MAD for robustness)
    mad = np.median(np.abs(diff - np.median(diff)))
    threshold = threshold_factor * 1.4826 * mad  # 1.4826 makes MAD consistent with std for normal distribution
    
    # Create bad pixel mask
    bad_pixel_mask = diff > threshold
    
    return bad_pixel_mask

def fix_outliers(image, window_size=3, threshold_factor=3):
    """
    Fix outliers in an image using median-based outlier detection and replacement.
    
    Parameters:
    - image: Input image array
    - window_size: Size of neighborhood window (odd number)
    - threshold_factor: How many standard deviations to use for thresholding
    
    Returns:
    - output_image: Image with outliers replaced by local median
    """
    
    # Detect bad pixels
    bad_pixel_mask = detect_bad_pixels(image, window_size, threshold_factor)
    
    # Infill outliers with median
    median_img = median_filter(image, size=window_size)
    output_image = image.copy()
    output_image[bad_pixel_mask] = median_img[bad_pixel_mask]

    return output_image

def fill_invalid_by_mean(array2d, mask):
    from scipy import ndimage
    filled_array = array2d.astype(float).copy()
    filled_array[mask == 1] = np.nan

    kernel = np.ones((10, 10), dtype=float)
    sum_of_neighbors = ndimage.convolve(np.nan_to_num(filled_array), kernel, mode='constant', cval=0.0)
    valid_count = ndimage.convolve((~np.isnan(filled_array)).astype(float), kernel, mode='constant', cval=0.0)
    local_mean = sum_of_neighbors / np.clip(valid_count, 1e-6, None)

    filled_array[mask == 1] = local_mean[mask == 1]
    return filled_array


# -------------- RANSAC: Source Direction Estimation --------------
def ransac_estimate_ps_qs_with_sign(
    pq, eta,
    n_iter=8000,
    subset_size=8,
    inlier_thresh=0.05,
    min_inlier_ratio=0.3,
    random_state=0,
    refine=True
):
    pq = np.asarray(pq, float)
    eta = np.asarray(eta, float).reshape(-1)
    eta = np.clip(eta, -1.0, 1.0)

    # sanitize
    ok = np.isfinite(pq).all(axis=1) & np.isfinite(eta)
    pq = pq[ok]
    eta = eta[ok]
    N = pq.shape[0]
    if N < subset_size:
        raise ValueError(f"not enough points: {N} < subset_size={subset_size}")

    rng = np.random.default_rng(random_state)
    min_inliers = max(int(np.ceil(min_inlier_ratio * N)), subset_size)

    # shading function (your compute_shading_value if exists, otherwise fallback)
    shading_fn = globals().get("compute_shading_value", None)
    if shading_fn is None:
        def shading_fn(p, q, ps, qs):
            denom_n = np.sqrt(p*p + q*q + 1.0)
            denom_s = np.sqrt(ps*ps + qs*qs + 1.0)
            return (p*ps + q*qs + 1.0) / (denom_n * denom_s)

    best = None
    best_rmse = np.inf

    for _ in range(n_iter):
        idx = rng.choice(N, subset_size, replace=False)

        # Subset estimation
        try:
            ps0, qs0, _ = reverse_photometric_stereo(pq[idx], eta[idx])
        except Exception:
            continue
        if not (np.isfinite(ps0) and np.isfinite(qs0)):
            continue

        # Evaluate all points
        pred = shading_fn(pq[:, 0], pq[:, 1], ps0, qs0)
        err_pos = np.abs(pred - eta)
        err_neg = np.abs(-pred - eta)
        use_neg = err_neg < err_pos
        sign = np.where(use_neg, -1.0, +1.0)
        res = np.minimum(err_pos, err_neg)

        inliers = res < inlier_thresh
        if inliers.sum() < min_inliers:
            continue

        # Refine with all inliers
        if refine:
            eta_eff = sign[inliers] * eta[inliers]
            try:
                ps, qs, _ = reverse_photometric_stereo(pq[inliers], eta_eff)
            except Exception:
                continue
            if not (np.isfinite(ps) and np.isfinite(qs)):
                continue

            pred2 = shading_fn(pq[:, 0], pq[:, 1], ps, qs)
            err2_pos = np.abs(pred2 - eta)
            err2_neg = np.abs(-pred2 - eta)
            res2 = np.minimum(err2_pos, err2_neg)
            inliers2 = res2 < inlier_thresh
            if inliers2.sum() < min_inliers:
                continue

            rmse = float(np.sqrt(np.mean(res2[inliers2] ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best = (ps, qs, inliers2, (err2_neg < err2_pos))
        else:
            rmse = float(np.sqrt(np.mean(res[inliers] ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best = (ps0, qs0, inliers, use_neg)

    # Fallback to all data
    if best is None:
        ps, qs, residual = reverse_photometric_stereo(pq, eta)
        inliers = np.ones(N, dtype=bool)
        use_neg = np.zeros(N, dtype=bool)
        return ps, qs, best_rmse, inliers, use_neg, True

    ps, qs, inliers, use_neg = best
    return ps, qs, best_rmse, inliers, use_neg, False


# -------------- Local Refinement of Planar Subset --------------
def run_refinement(subset_init, lap_gt, width, height, ind_v, ind_a, ind_b, ind_d, ind_e, ind_f):
    
    results_refined = []

    for _, candidate in enumerate(subset_init):
        i, p_i, q_i, p_a, q_a, p_b, q_b, p_d, q_d, p_e, q_e, p_f, q_f = candidate

        row = (i // (width-2)) + 1
        col = (i %  (width-2)) + 1

        idx_i = np.ravel_multi_index((row, col), dims=(height, width))
        i_v = np.where(ind_v == idx_i)[0][0]
        idx_a = ind_a[i_v]
        idx_b = ind_b[i_v]
        idx_d = ind_d[i_v]
        idx_e = ind_e[i_v]
        idx_f = ind_f[i_v]
        
        x0 = np.array([p_i, q_i, p_a, q_a, p_b, q_b, 
                       p_d, q_d, p_e, q_e, p_f, q_f], dtype=np.float64)
        
        def wrapped_objective(x):
            return obj_function(x, idx_i, idx_a, idx_b, idx_d, idx_e, idx_f, i_v, lap_gt, width, height)
        
        res = minimize(wrapped_objective,
                       x0,
                       method="SLSQP",
                       options={"ftol": 1e-16})
        best_x = res.x

        # Results
        p_i, q_i, p_a, q_a, p_b, q_b, p_d, q_d, p_e, q_e, p_f, q_f = best_x
        results_refined.append((i, p_i, q_i, p_a, q_a, p_b, q_b, p_d, q_d, p_e, q_e, p_f, q_f))

    return results_refined

def compute_total_obj_function(subset_init, lap_gt, width, height, ind_v, ind_a, ind_b, ind_d, ind_e, ind_f):

    total_cost = 0.0
    for (pix_id, p_i, q_i, p_a, q_a, p_b, q_b, p_d, q_d, p_e, q_e, p_f, q_f) in subset_init:
        row_in = (pix_id // (width-2)) + 1
        col_in = (pix_id %  (width-2)) + 1

        idx_i = np.ravel_multi_index((row_in, col_in), dims=(height, width))
        i_v = np.where(ind_v == idx_i)[0][0]
        idx_a = ind_a[i_v]
        idx_b = ind_b[i_v]
        idx_d = ind_d[i_v]
        idx_e = ind_e[i_v]
        idx_f = ind_f[i_v]

        x = np.array([p_i, q_i, p_a, q_a, p_b, q_b, p_d, q_d, p_e, q_e, p_f, q_f], dtype=np.float64)

        pix_obj = obj_function(x, idx_i, idx_a, idx_b, idx_e, idx_d, idx_f, i_v, lap_gt, width, height)
        total_cost += pix_obj

    return total_cost

def main_refinement(subset_init, pq_init, width, height, lap_gt, eta_array, max_iter=None):

    ind_v, ind_a, ind_b, ind_c, ind_d, ind_e, ind_f, ind_g, ind_h = get_ngbr_indices(width, height)

    cost_prev = compute_total_obj_function(subset_init, lap_gt, width, height, ind_v, ind_a, ind_b, ind_d, ind_e, ind_f)
    print(f"Initial cost = {cost_prev:.6f}")

    ps_final, qs_final, residual = np.nan, np.nan, np.inf

    ps_init, qs_init, residual_init, inliers_init, use_neg_init, fell_back_init = ransac_estimate_ps_qs_with_sign(
        pq_init, eta_array,
        n_iter=8000,
        subset_size=min(8, len(pq_init)),
        inlier_thresh=0.05,
        min_inlier_ratio=0.3,
        random_state=0,
        refine=True
    )
    print("Initial estimate (RANSAC)")
    print(f"    fell_back={fell_back_init}, inlier_ratio={float(inliers_init.mean()):.3f}, residual={residual_init:.6e}")
    print(f"    ps, qs = {ps_init:.3f}, {qs_init:.3f}")

    for iter in range(1, max_iter + 1):
        results_refined = run_refinement(subset_init, lap_gt, width, height, ind_v, ind_a, ind_b, ind_d, ind_e, ind_f)
        subset_init = results_refined

        pq_list_refined = []
        for (pix_id, p_i, q_i, p_a, q_a, p_b, q_b, p_d, q_d, p_e, q_e, p_f, q_f) in subset_init:
            pq_list_refined.append([p_i, q_i])
        pq_array_refined = np.array(pq_list_refined, dtype=np.float64)
        eta_use = np.asarray(eta_array, dtype=np.float64).reshape(-1)

        ps_final, qs_final, residual, inliers, use_neg, fell_back = ransac_estimate_ps_qs_with_sign(
            pq_array_refined, eta_use,
            n_iter=8000,
            subset_size=min(8, len(pq_array_refined)),
            inlier_thresh=0.05,
            min_inlier_ratio=0.3,
            random_state=0,
            refine=True
        )
        print(f"    light(RANSAC): fell_back={fell_back}, inlier_ratio={float(inliers.mean()):.3f}, residual={residual:.6e}")

        cost_new = compute_total_obj_function(subset_init, lap_gt, width, height,
                                              ind_v, ind_a, ind_b, ind_d, ind_e, ind_f)
        print(f"Iter {iter:02d} | Cost new = {cost_new:.6f} | "
              f"inlier_ratio={float(np.mean(inliers)):.3f} | residual={residual:.6e}")

        diff_cost = abs(cost_new - cost_prev)
        cost_prev = cost_new
        if diff_cost < 1e-16:
            break

    return subset_init, ps_final, qs_final, residual

def extract_candidates_v3(L, eta_gt_2d, height, width, ps_s, qs_s, iters=None):

    H, W = height, width
    H_in, W_in = H - 2, W - 2

    invalid_map = np.zeros((H, W), dtype=np.uint8)
    sign_map    = np.zeros((H, W), dtype=np.uint8)
    ambiguous_map = np.zeros((height, width), dtype=np.uint8)

    P = np.zeros((H, W, 2), dtype=np.float64)
    Q = np.zeros((H, W, 2), dtype=np.float64)

    # 1. Compute all candidates at once
    (p1, q1), (p2, q2) = compute_p_q_from_w_v3(L)

    valid = np.isfinite(p1) & np.isfinite(q1) & np.isfinite(p2) & np.isfinite(q2)
    invalid = ~valid

    p1_2d = p1.reshape(H_in, W_in)
    q1_2d = q1.reshape(H_in, W_in)
    p2_2d = p2.reshape(H_in, W_in)
    q2_2d = q2.reshape(H_in, W_in)
    inv_2d = invalid.reshape(H_in, W_in)

    invalid_map[1:-1, 1:-1] = inv_2d.astype(np.uint8)

    P[1:-1, 1:-1, 0] = p1_2d
    Q[1:-1, 1:-1, 0] = q1_2d
    P[1:-1, 1:-1, 1] = p2_2d
    Q[1:-1, 1:-1, 1] = q2_2d

    # 2. Handle NaNs
    np.nan_to_num(P, copy=False, nan=0.0)
    np.nan_to_num(Q, copy=False, nan=0.0)

    # 3. Initial sign assignment using eta consistency
    for rr in range(H_in):
        r = rr + 1
        for cc in range(W_in):
            c = cc + 1
            if invalid_map[r, c]:
                continue

            p_1, q_1 = P[r, c, 0], Q[r, c, 0]
            p_2, q_2 = P[r, c, 1], Q[r, c, 1]

            target_eta = eta_gt_2d[r, c]
            eta1 = compute_shading_value(p_1, q_1, ps_s, qs_s)
            eta2 = compute_shading_value(p_2, q_2, ps_s, qs_s)

            e1 = abs(eta1 - target_eta)
            e2 = abs(eta2 - target_eta)
            sign_map[r, c] = 0 if e1 < e2 else 1

            # Ambiguous if both shading errors are nearly equal
            if abs(e1 - e2) < 0.03:
                ambiguous_map[r, c] = 1

    # 4. FC refinement (Optional)
    # sign_map = refine_with_fc(P, Q, sign_map, iters)

    # 5. Make subset
    subset = []
    for r in range(1, H-1):
        for c in range(1, W-1):
            i = (r - 1) * W_in + (c - 1)

            if invalid_map[r, c]:
                subset.append((i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
                continue

            s = int(sign_map[r, c])
            p = P[r, c, s]
            q = Q[r, c, s]
            subset.append((i, p, q, p, q, p, q, p, q, p, q, p, q))

    return subset, sign_map, invalid_map, ambiguous_map

def obj_function_pq(p_param, q_param, idx_i_torch, idx_a_torch, idx_b_torch,
                    idx_d_torch, idx_e_torch, idx_f_torch, lap_torch, width, height):

    p_i = p_param[idx_i_torch]
    q_i = q_param[idx_i_torch]
    p_a = p_param[idx_a_torch]
    q_a = q_param[idx_a_torch]
    p_b = p_param[idx_b_torch]
    q_b = q_param[idx_b_torch]
    p_d = p_param[idx_d_torch]
    q_d = q_param[idx_d_torch]
    p_e = p_param[idx_e_torch]
    q_e = q_param[idx_e_torch]
    p_f = p_param[idx_f_torch]
    q_f = q_param[idx_f_torch]

    pq_modified = torch.zeros((height*width, 2), device=p_param.device)
    pq_modified[idx_i_torch, 0] = p_i
    pq_modified[idx_i_torch, 1] = q_i

    pq_modified[idx_a_torch, 0] = p_a
    pq_modified[idx_a_torch, 1] = q_a

    pq_modified[idx_b_torch, 0] = p_b
    pq_modified[idx_b_torch, 1] = q_b

    pq_modified[idx_d_torch, 0] = p_d
    pq_modified[idx_d_torch, 1] = q_d

    pq_modified[idx_e_torch, 0] = p_e
    pq_modified[idx_e_torch, 1] = q_e

    pq_modified[idx_f_torch, 0] = p_f
    pq_modified[idx_f_torch, 1] = q_f

    lap_hat = cotan_laplacians_from_pq_torch(pq_modified, width, height, device=p_param.device)
    diff = lap_torch - lap_hat
    E_lap = torch.mean(diff**2)

    r_i = (p_d - p_i) - (q_b - q_i)
    r_f = (p_e - p_f) - (q_a - q_f)
    E_int = torch.mean(r_i**2 + r_f**2)

    return E_lap + E_int

def optimize_pq(p_init, q_init, idx_i_torch, idx_a_torch, idx_b_torch, idx_d_torch, idx_e_torch, idx_f_torch,
                lap_torch, width, height, optimization_params):

    adam_params = optimization_params['adam']
    lr = adam_params['lr']
    max_iters = adam_params['max_iters']

    p_param = torch.nn.Parameter(p_init.clone())
    q_param = torch.nn.Parameter(q_init.clone())
    optimizer = optim.Adam([p_param, q_param], lr=lr)

    tol = 1e-12
    prev_loss = None

    pbar = tqdm(range(max_iters), desc="Optimization")
    best_loss = float('inf')
    best_p = p_param.clone().detach()
    best_q = q_param.clone().detach()

    for iter in pbar:
        optimizer.zero_grad()
        loss = obj_function_pq(p_param, q_param,
                               idx_i_torch, idx_a_torch, idx_b_torch,
                               idx_d_torch, idx_e_torch, idx_f_torch,
                               lap_torch, width, height)
        loss.backward()
        optimizer.step()
        current_loss = loss.item()

        if prev_loss is not None and abs(prev_loss - current_loss) < tol:
            pbar.set_description(f"Converged at iter {iter} | Loss diff {abs(prev_loss - current_loss):.2e}")
            break
        prev_loss = current_loss

        if current_loss < best_loss:
            best_loss = current_loss
            best_p = p_param.clone().detach()
            best_q = q_param.clone().detach()

        if iter % 10 == 0:
            pbar.set_description(f"Iter: {iter} | Loss: {current_loss:.6f}")

    return best_p, best_q, p_param, q_param


# ---------------------------------------------------------------------------
# Analysis-by-synthesis helpers
#
# Used by notebooks/sim_05_analysis_by_synthesis.ipynb: recover the light
# source by RANSAC over reverse photometric stereo, then fix the per-pixel
# convex/concave sign of (p, q) against the measured shading.
# ---------------------------------------------------------------------------

def xyz_to_pq_torch(x, y, z):
    eps = 1e-8
    z = torch.where(torch.abs(z) < eps, torch.full_like(z, eps), z)
    p = x / z
    q = y / z
    return p, q

def t_and_reference_to_unit_vector_torch(t, eta, s_vec):
    s_norm = s_vec / s_vec.norm(p=2)
    R = rotation_matrix_z_to_vec_torch(s_norm)
    vec_2d = torch.stack([torch.cos(t),
                          torch.sin(t),
                          torch.zeros_like(t)], dim=1)
    orthogonal_component = vec_2d @ R.T
    #r = torch.sqrt(1.0 - eta**2)
    r = torch.sqrt(torch.clamp(1.0 - eta**2, min=0.0))
    s_norm_2d = s_norm.unsqueeze(0)
    n_t = -eta.unsqueeze(1)*s_norm_2d + r.unsqueeze(1)*orthogonal_component
    return n_t

def t_to_pq_torch(t, eta, s_vec):
    n_t = t_and_reference_to_unit_vector_torch(t, eta, s_vec)
    x, y, z = n_t[:, 0], n_t[:, 1], n_t[:, 2]
    p, q = xyz_to_pq_torch(x, y, z)
    return p, q

def correct_signs_light(pq_array, eta_array, n_trials=None, random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)

    N = pq_array.shape[0]
    p = pq_array[:, 0]
    q = pq_array[:, 1]

    best_err = np.inf
    best_signs = None

    for _ in range(n_trials):
        # Hemisphere sampling
        u = np.random.rand()
        theta = np.random.rand() * 2*np.pi
        phi = np.arccos(u)

        sx = np.sin(phi)*np.cos(theta)
        sy = np.sin(phi)*np.sin(theta)
        sz = np.cos(phi)
        ps = sx/sz
        qs = sy/sz

        # Compute shading values
        eta_org = compute_shading_value( p,  q,  ps, qs)
        eta_inv = compute_shading_value(-p, -q,  ps, qs)
        err_org = np.abs(eta_org - eta_array)
        err_inv = np.abs(eta_inv - eta_array)
        mask = err_org <= err_inv
        signs = np.where(mask, 1, -1)

        # Compute total error
        total_err = np.sum(np.minimum(err_org, err_inv))

        # Update best values
        if total_err < best_err:
            best_err   = total_err
            best_signs = signs.copy()

    return best_signs, best_err

def angle_diff(a, b):
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi

def reverse_photometric_stereo_ransac(pq_array, eta_array, n_iter, subset_size):
    best_s     = None
    best_score = np.inf
    residuals = []

    for i in range(n_iter):
        subset_idx = np.random.choice(pq_array.shape[0], size=subset_size, replace=False)
        pq_subset  = pq_array[subset_idx]
        eta_subset = eta_array[subset_idx]

        ps_est, qs_est, residual = reverse_photometric_stereo(pq_subset, eta_subset)

        residuals.append(residual)

        if residual < best_score:
            best_score = residual
            best_s     = (ps_est, qs_est)
    ps_best, qs_best = best_s

    return ps_best, qs_best, best_score

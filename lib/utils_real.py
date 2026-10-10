import os
import pickle as pkl
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pytorch3d.renderer import (
    PerspectiveCameras,
)
import skimage

workspace_path = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
import sys
sys.path.append(workspace_path)

import lib.constants as constants



## Load and save mesh
def save_meshes_pkl(v, f, workspace_path, filename, exp_name, **kwargs):
    # create a folder named as filename in results folder
    # save the pkl file in that folder
    os.makedirs(os.path.join(workspace_path, 'results', filename), exist_ok=True)
    dict_to_save = dict(
        verts_seq = v,
        faces = f,
    )
    dict_to_save.update(kwargs)
    with open(os.path.join(workspace_path, 'results', filename, exp_name + '.pkl'), 'wb') as f:
        pkl.dump(dict_to_save, f)

## plot n images
def plt_plot_first_nimages(images, n=1, num_cols=4, figsize=(10, 5), title=None):
    if figsize is not None:
        plt.figure(figsize=figsize)
    if title is not None:
        plt.suptitle(title)
    print(min(images.shape[0], n))
    for i in range(min(images.shape[0], n)):
        plt.subplot(n//num_cols+1, num_cols, i+1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()



## Counts to Temp
def counts2temp(data_counts, camera):
    Emiss = constants.EPSILON
    K1 = 1 / (constants.Tau * Emiss * constants.TransmissionExtOptics)
        
    R = constants.R_const[camera]
    B = constants.B_const[camera]
    F = constants.F_const[camera]
    O = constants.O_const[camera]
    
    # Pseudo radiance of the reflected environment
    r1 = ((1-Emiss)/Emiss) * (R/(np.exp(B/constants.TRefl)-F))
    # # Pseudo radiance of the atmosphere
    r2 = ((1 - constants.Tau)/(Emiss * constants.Tau)) * (R/(np.exp(B/constants.TAtm)-F)) 
    # # Pseudo radiance of the external optics
    r3 = ((1-constants.TransmissionExtOptics) / (Emiss * constants.Tau * constants.TransmissionExtOptics)) * (R/(np.exp(B/constants.TExtOptics)-F))
            
    K2 = r1 + r2 + r3
    
    data_obj_signal = data_counts.astype(float)
    data_temp = (B / np.log(R/((K1 * data_obj_signal) - K2 - O) + F))
    # print(np.amin(data_counts), np.amax(data_counts), np.amin(data_temp), np.amax(data_temp), data_obj_signal.shape)
    
    return data_temp
## Get a perspective Camera
def get_perspective_camera(img_size, HFOV, VFOV, R, T, device):
    fx = (img_size[1]/2)/np.tan(np.deg2rad(HFOV/2))
    fy = (img_size[0]/2)/np.tan(np.deg2rad(VFOV/2))

    focal_length = torch.tensor([[fx, fy] for i in range(R.shape[0])], device=device, dtype=torch.float32) # Width, Height
    principal_point = torch.tensor([[img_size[1]/2, img_size[0]/2] for i in range(R.shape[0])], device=device) # Width, Height

    # The definition of image_size is now (height, width)
    torch_image_size = torch.tensor([[img_size[0], img_size[1]] for i in range(R.shape[0])], device=device, dtype=torch.float32) # Height, Width
    cameras = PerspectiveCameras(device=device, focal_length=focal_length, principal_point=principal_point, R=R, T=T, image_size=torch_image_size, in_ndc=False)
    return cameras

## Render Vertices as Image
def render_vertices_as_image(u_vals, pixel_coords, pix_image_num, img_size, fill_value=None, init_image=None, renderNviews=None):
    u_vals_pixels = []
    if fill_value is None:
        min_vals = u_vals.min()
    else:
        min_vals = fill_value
    if renderNviews is None:
        renderNviews = torch.count_nonzero(torch.unique(pix_image_num) != -1).cpu().numpy()
    for i in range(renderNviews):
        if init_image is not None:
            u_vals_img = init_image
        else:
            u_vals_img = torch.full((u_vals.shape[0], img_size[0], img_size[1]), min_vals, dtype=u_vals.dtype, device=u_vals.device)
        valid_verts = torch.where(pix_image_num[:,i] == i)
        img_coords = pixel_coords[valid_verts]
        u_vals_img[:, img_coords[:, 0], img_coords[:, 1]] = u_vals[:, valid_verts[0]]
        u_vals_pixels.append(u_vals_img)
    return torch.stack(u_vals_pixels, dim=0)
    
## Compute face normals (pytorch)
def compute_face_normals_torch(verts, faces, normalize=False):
    verts_faces = verts[faces]

    edge1 = verts_faces[:, 1] - verts_faces[:, 0]
    edge2 = verts_faces[:, 2] - verts_faces[:, 0]
    
    cross_prod = torch.cross(edge1, edge2, dim=-1)

    face_normals = cross_prod
    neg_idx = torch.where(face_normals[:, -1] < 0)
    face_normals[neg_idx] = -face_normals[neg_idx]
    if normalize:
        face_normals = face_normals / torch.norm(face_normals, dim=-1, keepdim=True)
        return face_normals
    else:
        return face_normals

## Compute per vertex normals (pytorch)
def compute_vertex_normals_torch(verts, faces, camera_loc=torch.tensor([0,0,100.0], device='cuda')):
    
    face_normals = compute_face_normals_torch(verts, faces)
    # camera_dir = camera_loc[None, :] - verts

    # # Make sure the face normals point in the same direction
    # pos_norms = torch.sum(cross_prod * camera_dir[faces[:, 0]], dim=-1, keepdim=True)
    # neg_norms = torch.sum(-cross_prod * camera_dir[faces[:,0]], dim=-1, keepdim=True)
    # face_normals = torch.where(pos_norms >= neg_norms, cross_prod, -cross_prod)

    # face_normals = torch.where(torch.sum(cross_prod * verts_faces[:, 0], dim=-1, keepdim=True) >= 0, cross_prod, -cross_prod)

    # Compute the vertex normals by averaging the face normals based on triangle area
    face_normals_stacked = torch.stack([face_normals, face_normals, face_normals], dim=1)

    zero_idx = torch.zeros_like(faces)
    idx = torch.stack([faces, zero_idx])

    normals_x = torch.sparse_coo_tensor(idx.reshape(2,-1), face_normals_stacked[..., 0].reshape(-1), size=(verts.shape[0], 1)).coalesce().to_dense()
    normals_y = torch.sparse_coo_tensor(idx.reshape(2,-1), face_normals_stacked[..., 1].reshape(-1), size=(verts.shape[0], 1)).coalesce().to_dense()
    normals_z = torch.sparse_coo_tensor(idx.reshape(2,-1), face_normals_stacked[..., 2].reshape(-1), size=(verts.shape[0], 1)).coalesce().to_dense()

    normals = torch.cat([normals_x, normals_y, normals_z], dim=-1)
    normals = normals / torch.norm(normals, dim=-1, keepdim=True)

    return normals


device = torch.device("cuda:0")
## Increase image border
def increase_image_border(images, border_width=1, bg_value=100):
    
    mask_img = np.zeros_like(images, dtype=bool)
    mask_img[images != bg_value] = True
    dimg = images.copy()
    dimg[dimg == bg_value] = np.NaN
    xdiff = np.diff(dimg, axis=-1)
    ydiff = np.diff(dimg, axis=-2)
    xdiff[np.isnan(xdiff)] = -0.01
    ydiff[np.isnan(ydiff)] = -0.01
    xdiff = np.pad(xdiff, ((0,0), (0,0), (1,0)), mode='constant', constant_values=0)
    ydiff = np.pad(ydiff, ((0,0), (1,0), (0,0)), mode='constant', constant_values=0)

    # blur xdiff and ydiff
    xdiff = skimage.filters.gaussian(xdiff, sigma=5.8, preserve_range=True, truncate=2.0, cval=0.0)
    ydiff = skimage.filters.gaussian(ydiff, sigma=5.8, preserve_range=True, truncate=2.0, cval=0.0)

    # print(xdiff.shape, ydiff.shape, mask_img.shape)
    for k in range(border_width):
        s1 = np.roll(mask_img, 1, axis=-1) & ~mask_img
        s2 = np.roll(mask_img, -1, axis=-1) & ~mask_img
        s3 = np.roll(mask_img, 1, axis=-2) & ~mask_img
        s4 = np.roll(mask_img, -1, axis=-2) & ~mask_img
        
        # Roll the diff image and add it to the original image
        xdiff[np.where(s1)] = np.roll(xdiff, 1, axis=-1)[np.where(s1)]
        xdiff[np.where(s2)] = np.roll(xdiff, -1, axis=-1)[np.where(s2)]
        
        ydiff[np.where(s3)] = np.roll(ydiff, 1, axis=-2)[np.where(s3)]
        ydiff[np.where(s4)] = np.roll(ydiff, -1, axis=-2)[np.where(s4)]

        images[np.where(s1)] = np.roll(images, 1, axis=-1)[np.where(s1)] - xdiff[np.where(s1)]
        images[np.where(s2)] = np.roll(images, -1, axis=-1)[np.where(s2)] - xdiff[np.where(s2)]
        images[np.where(s3)] = np.roll(images, 1, axis=-2)[np.where(s3)] - ydiff[np.where(s3)]
        images[np.where(s4)] = np.roll(images, -1, axis=-2)[np.where(s4)] - ydiff[np.where(s4)]
        
        mask_img[np.where(s1)] = 1
        mask_img[np.where(s2)] = 1
        mask_img[np.where(s3)] = 1
        mask_img[np.where(s4)] = 1

    images[images == bg_value] = 0.0
    return images

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

def compute_shading_value(p, q, ps, qs):
    denom_n = np.sqrt(p**2 + q**2 + 1.0)
    denom_s = np.sqrt(ps**2 + qs**2 + 1.0)
    eta = (p*ps + q*qs + 1.0) / (denom_n * denom_s)
    eta = np.maximum(eta, 0.0)
    return eta

def get_largest_closed_mask(frames_mask, method=cv2.RETR_TREE):
    contours, hierarchy = cv2.findContours(frames_mask.astype(np.uint8), method, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_mask = np.zeros_like(frames_mask, dtype=np.uint8)
    cv2.drawContours(largest_mask, [largest_contour], -1, (1), thickness=cv2.FILLED)
    child_contours = [contours[i] for i, h in enumerate(hierarchy[0]) if h[3] >= 0]
    cv2.drawContours(largest_mask, child_contours, -1, (0), thickness=cv2.FILLED)
    return largest_mask

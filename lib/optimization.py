import numpy as np
import scipy as sp
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
import matplotlib as mpl
import pytorch3d as p3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform,
                                RasterizationSettings,
                                MeshRenderer,
                                MeshRasterizer,
                                MeshRendererWithFragments)
from pytorch3d.loss import (mesh_edge_loss,
                            mesh_laplacian_smoothing,
                            mesh_normal_consistency)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from tqdm import tqdm
import gpytoolbox

workspace_path = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))

import sys
sys.path.append(workspace_path)
import meshplot as mp

import largesteps
from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix, massmatrix_fast
from largesteps.optimize import AdamUniform

from lib.utils_real import *
import lib.constants as constants

class BaseOptimizationClass():

    def __init__(self, params, plot=False):
        self.params = params
        self.plot = plot
        self.device = params.device
        
        # Just initializing some optimizable variables to None
        self.K_opti = None
        self.optimized_heat_vals = None
        self.heat_source_opti = None
        self.heat_capacity_opti = None
        self.convection_coeff_opti = None
        self.alpha_opti = None
        self.HFOV = params.HFOV if hasattr(params, 'HFOV') else constants.HFOV
        self.VFOV = params.VFOV if hasattr(params, 'VFOV') else constants.VFOV
        pass


    def prepare_renderers(self, R, T, img_size):
        self.R, self.T, self.img_size = R, T, img_size
        self.cameras = get_perspective_camera(img_size, self.HFOV, self.VFOV, R, T, self.device)
        
        raster_settings = RasterizationSettings(
            image_size=(img_size[0], img_size[1]),
            blur_radius=0,
            faces_per_pixel=1,
            clip_barycentric_coords=True,
        )

        sigma = 1e-4
        raster_settings_sh = RasterizationSettings(
            image_size=(img_size[0], img_size[1]),
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
            faces_per_pixel=50,
            clip_barycentric_coords=True,
        )

        self.renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=p3d.renderer.mesh.shader.HardDepthShader(
                device=self.device,
                cameras=self.cameras,
            )
        )

        self.renderer_sh = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings_sh
            ),
            shader=p3d.renderer.mesh.shader.SoftSilhouetteShader()
        )


    def prepare_candidate_mesh(self, mask=None, depth=None):
        self.get_mesh_from_mask(mask, depth)
        boundary_vertices = gpytoolbox.boundary_vertices(self.faces_sim_np)
        self.boundary_vertices = torch.from_numpy(boundary_vertices).to(self.device).long()
        self.interior_vertices = torch.from_numpy(np.setdiff1d(np.arange(self.verts_sim_pt.shape[0]), boundary_vertices)).to(self.device).long()
        self.compute_boundary_normals(mask)


    def get_mesh_from_mask(self, mask, depth = None):
        valid_pixel_locs = np.stack(np.where(mask), axis=-1)
        self.cat_corres_pixel_coords_init = torch.tensor(valid_pixel_locs.copy(), device=self.device, dtype=torch.long)
        self.cat_points_image_num_init = torch.full([valid_pixel_locs.shape[0], self.params.num_files], 0, device=self.device, dtype=torch.long)
        for i in range(self.cat_points_image_num_init.shape[1]):
            self.cat_points_image_num_init[:, i] = i

        # Do a simple triangulation of a 2D mask to get face indices
        triangulation = mpl.tri.Triangulation(valid_pixel_locs[:,0], valid_pixel_locs[:,1])
        triangles = triangulation.triangles
        xtri = valid_pixel_locs[:, 0][triangles] - np.roll(valid_pixel_locs[:, 0][triangles], 1, axis=1)
        ytri = valid_pixel_locs[:, 1][triangles] - np.roll(valid_pixel_locs[:, 1][triangles], 1, axis=1)

        edge_lengths = np.sqrt(xtri**2 + ytri**2)
        mean_edge_length = np.mean(edge_lengths)
        std = np.std(edge_lengths)
        trimask = edge_lengths < (mean_edge_length + 2*std)
        trimask = np.all(trimask, axis=-1)
        faces_sim_np = triangulation.triangles[trimask.astype(bool)]
        
        xy_vals_np = valid_pixel_locs[:, ::-1].copy() + 0.5 # Values related to Width first and then Height
        xy_vals_np[:, 1] = mask.shape[0] - xy_vals_np[:, 1]
        xy_vals_np[:, 0] = mask.shape[1] - xy_vals_np[:, 0]
        init_xy_vals = torch.from_numpy(xy_vals_np).to(device=self.device)

        if depth is not None:
            depth_vals = depth[valid_pixel_locs[:, 0], valid_pixel_locs[:, 1]]
            depth_vals_pt = torch.from_numpy(depth_vals[:, np.newaxis]).to(device=self.device)
        else:
            depth_vals_pt = torch.full([valid_pixel_locs.shape[0], 1], self.params.obj_dist, device=self.device)
        
        init_xy_depth = torch.cat([init_xy_vals, depth_vals_pt], dim=-1).float()
        init_xyz = self.cameras.unproject_points(init_xy_depth.float(), world_coordinates=True)
        verts_sim_np = init_xyz.cpu().numpy()
        
        self.verts_sim_pt = torch.tensor(verts_sim_np, device=self.device, dtype=torch.float32)
        self.faces_sim_pt = torch.tensor(faces_sim_np, device=self.device, dtype=torch.long)

        self.verts_sim_np = verts_sim_np
        self.faces_sim_np = faces_sim_np
        self.init_xy_depth = init_xy_depth
        self.obj_masks_pt = torch.from_numpy(mask).float().to(self.device)

        if self.plot:
            mp.plot(self.verts_sim_np, self.faces_sim_np, shading={"wireframe": True})


    def compute_boundary_normals(self, mask):
        mask_img = np.zeros_like(mask)
        mask_img[np.where(mask)] = 1.0

        # Pad mask Image
        num_pad = 1
        mask_img = np.pad(mask_img, num_pad, mode='constant', constant_values=0.0)

        # Get image gradients on mask
        zx = sp.ndimage.sobel(mask_img, axis=0)
        zy = sp.ndimage.sobel(mask_img, axis=1)

        # plot the vector field
        plt.figure()
        plt.imshow(mask_img)
        plt.quiver(zx, zy)
        plt.show()

        # Compute normals
        edge_map = np.stack([zx, zy, np.zeros_like(zx)], axis=-1)
        edge_map = edge_map/(np.linalg.norm(edge_map, axis=-1, keepdims=True) + 1e-8)
        camera_dir = np.array([0.0,0.0,1.0])
        mask_normal_map = np.cross(edge_map, camera_dir[None, None, :])
        mask_normal_map = mask_normal_map/(np.linalg.norm(mask_normal_map, axis=-1, keepdims=True) + 1e-8)
        mask_normal_map[..., :2] = -mask_normal_map[..., :2]
        self.mask_normal_map_pt = torch.from_numpy(mask_normal_map).float().to(self.device)
        self.boundary_normals = self.mask_normal_map_pt[self.cat_corres_pixel_coords_init[self.boundary_vertices, 0] + num_pad, self.cat_corres_pixel_coords_init[self.boundary_vertices, 1] + num_pad]
        self.boundary_normals = self.boundary_normals/torch.norm(self.boundary_normals, dim=1, keepdim=True)

        plt.imshow(mask_img)
        plt.quiver(mask_normal_map[..., 0], mask_normal_map[..., 1])
        plt.show()


    def map_heatvalues_tomesh(self, frames_temp):
        with torch.no_grad():
            self.u_real_atverts_pt = torch.full((frames_temp.shape[0], frames_temp.shape[1],self.verts_sim_pt.shape[0]),constants.OBJECT_TEMP, dtype=torch.float32, device=self.device)
            verts_wprojection_forview = torch.zeros([frames_temp.shape[0], self.verts_sim_pt.shape[0]], dtype=torch.bool, device=self.device)
            verts_wprojection = torch.zeros([self.verts_sim_pt.shape[0]], dtype=torch.bool, device=self.device)

            for i in range(frames_temp.shape[0]):
                # Changing this to have separate output along the column dimension.
                valid_indx_init = torch.where(self.cat_points_image_num_init[:, i] == i)[0]
                for j in range(frames_temp.shape[1]):
                    if frames_temp[i][j].shape[0] != self.img_size[0] or frames_temp[i][j].shape[1] != self.img_size[1]:
                        tmp_frame = cv2.resize(frames_temp[i][j], (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_CUBIC)
                    else:
                        tmp_frame = frames_temp[i][j]
                    tmp_frame_pt = torch.from_numpy(tmp_frame).float().to(self.device)
                    self.u_real_atverts_pt[i, j, valid_indx_init] = tmp_frame_pt[self.cat_corres_pixel_coords_init[valid_indx_init, 0], self.cat_corres_pixel_coords_init[valid_indx_init, 1]]
                
                verts_wprojection[valid_indx_init] = True
                verts_wprojection_forview[i, valid_indx_init] = True
            
            self.u_real_atverts_np = self.u_real_atverts_pt.detach().cpu().numpy()
            self.valid_projected_locs = torch.where(verts_wprojection)[0].cpu().numpy()
            self.invalid_projected_locs = torch.where(~verts_wprojection)[0].cpu().numpy()
            self.valid_projected_locs_forview = torch.where(verts_wprojection_forview)


    @property
    def num_vertices(self):
        return self.verts_sim_pt.shape[0]
    
    @property
    def num_faces(self):
        return self.faces_sim_pt.shape[0]

    @property
    def num_timesteps(self):
        return self.u_real_atverts_pt.shape[1]

    @property
    def dt(self):
        return self.params.NUM_FRAME_DELTA/60.0

class OptimizationWrapper(BaseOptimizationClass):

    def __init__(self, params, plot=False):
        super().__init__(params, plot)
        

    def initialize_Lopti_variables(self, opti_params):
        with torch.no_grad():
            
            ### Optimize Heat values at Unknown Locations ###
            if self.params.optim_heat_vals:
                self.optimized_heat_vals = self.u_real_atverts_pt.clone().detach().requires_grad_(True)
            else:
                self.optimized_heat_vals = self.u_real_atverts_pt.clone().detach().requires_grad_(False)
            
            # Sampling probability for the time dimension
            dT_dt = (self.optimized_heat_vals[:, 1:] - self.optimized_heat_vals[:, :-1]).detach().abs()
            dT_dt = dT_dt.permute(0, 2, 1)
            self.dT_dt_weights = dT_dt / torch.sum(dT_dt, dim=-1, keepdim=True)

            self.unknown_heatverts_mask = torch.zeros_like(self.u_real_atverts_pt, dtype=torch.bool, device=self.device)
            self.unknown_heatverts_mask[:, :, self.invalid_projected_locs] = True

            ### Optimize Heat Capacity ###
            self.heat_capacity_opti = torch.tensor(np.sqrt(self.params.SURFACE_THICKNESS*self.params.RHO*self.params.C), dtype=torch.float32, device=self.device)
            if self.params.optim_heat_capacity:
                self.heat_capacity_opti.requires_grad = True
            else:
                self.heat_capacity_opti.requires_grad = False

            ### Optimize Heat Source ###
            if self.params.optim_heat_source:
                self.heat_source_opti = torch.randn([self.u_real_atverts_pt.shape[0], 1, self.verts_sim_pt.shape[0]], device=self.device) * 0.001
                self.heat_source_opti.requires_grad = True
                self.heat_source_shape = self.heat_source_opti.shape
            if hasattr(self.params, 'use_heat_source') and self.params.use_heat_source:
                self.heat_source_opti = torch.from_numpy(np.sqrt(opti_params.heat_source).astype(np.float32)).to(self.device)
                self.heat_source_opti = self.heat_source_opti.reshape(self.u_real_atverts_pt.shape[0], 1, self.verts_sim_pt.shape[0])
                self.heat_source_opti.requires_grad = False
                self.heat_source_shape = self.heat_source_opti.shape

            ### Optimize Convection Coefficient ###
            self.convection_coeff_opti = torch.tensor(np.sqrt(constants.CONVECTION_COEFF), dtype=torch.float32, device=self.device)
            if self.params.optim_convection_coeff:
                self.convection_coeff_opti.requires_grad = True
            else:
                self.convection_coeff_opti.requires_grad = False

            if self.params.optim_thermal_cond:
                self.thermal_cond = torch.tensor(1.0, dtype=torch.float32, device=self.device)
                self.thermal_cond.requires_grad = True
            
            if self.params.optim_thermal_diff:
                self.thermal_diff = torch.tensor(1.0, dtype=torch.float32, device=self.device)
                self.thermal_diff.requires_grad = True

            ### Optimize Shape Offsets ###
            if self.params.optim_laplacian:
                L_locs = np.zeros([self.verts_sim_pt.shape[0], self.verts_sim_pt.shape[0]], dtype=bool)
                for k in range(self.faces_sim_np.shape[0]):
                    L_locs[self.faces_sim_np[k, 0], self.faces_sim_np[k, 1]] = True
                    L_locs[self.faces_sim_np[k, 1], self.faces_sim_np[k, 2]] = True
                    L_locs[self.faces_sim_np[k, 2], self.faces_sim_np[k, 0]] = True
                    L_locs[self.faces_sim_np[k, 1], self.faces_sim_np[k, 0]] = True
                    L_locs[self.faces_sim_np[k, 2], self.faces_sim_np[k, 1]] = True
                    L_locs[self.faces_sim_np[k, 0], self.faces_sim_np[k, 2]] = True

                # Make L_locs upper triangular
                L_locs = np.triu(L_locs)

                self.MinvL_locs = torch.from_numpy(np.array(np.where(L_locs))).to(device=self.device, dtype=torch.long)
                
                # Initialize L matrix values
                if hasattr(opti_params, 'L_matrix'):
                    L_matrix = opti_params.L_matrix
                    Lidxs = np.array(np.where(L_locs))
                    L_values = np.array(L_matrix[Lidxs[0], Lidxs[1]])
                    self.MinvLopti = torch.from_numpy(L_values[0]).to(self.device)
                else:
                    self.MinvLopti = torch.ones([self.MinvL_locs[0].shape[0]], device=self.device)

                # Initialize M matrix values
                if hasattr(opti_params, 'M_matrix'):
                    M_matrix = opti_params.M_matrix
                    M_diag = np.array(M_matrix.diagonal())
                    self.mass_opti = torch.from_numpy(np.sqrt(M_diag)).to(self.device)
                else:
                    self.mass_opti = torch.ones([self.verts_sim_pt.shape[0]], device=self.device)
                
                self.MinvLopti.requires_grad = True
            

    def initialize_Loptimizers(self, Lopti_params):
        optim_variables = []
        if self.params.optim_laplacian:
            optim_variables += [self.MinvLopti]
        if self.params.optim_heat_vals:
            optim_variables += [self.optimized_heat_vals]
        if self.params.optim_thermal_cond:
            optim_variables += [self.thermal_cond]
        if self.params.optim_thermal_diff:
            optim_variables += [self.thermal_diff]
        if self.params.optim_heat_source:
            optim_variables += [self.heat_source_opti]
        if self.params.optim_heat_capacity:
            optim_variables += [self.heat_capacity_opti]
        if self.params.optim_convection_coeff:
            optim_variables += [self.convection_coeff_opti]
        
        self.Loptimizer = optim.Adam(optim_variables, lr=Lopti_params.learning_rate)
        self.Llr_scheduler = torch.optim.lr_scheduler.StepLR(self.Loptimizer, step_size=Lopti_params.lr_step_size, gamma=Lopti_params.lr_gamma)


    def initialize_Loptimaztion(self, Lopti_params):

        self.initialize_Lopti_variables(Lopti_params)
        self.initialize_Loptimizers(Lopti_params)
        self.Lopti_params = Lopti_params


    def run_MinvLoptimization(self, writer=None, verts_numpy=[], faces_numpy=[], losses=None, filename='dummy', exp_name='v1', global_step=0):
        
        # Pre-process before optimization
        t = self.params.NUM_FRAME_DELTA/60.0
        V = self.verts_sim_pt.shape[0]
        faces_numpy.append(self.faces_sim_np)

        if losses is None:
            losses = []

        # Iterative roop
        for step in tqdm(range(self.Lopti_params.num_iterations)):

            # Prepare heat values at vertices
            self.Loptimizer.zero_grad()
            if not self.params.optim_heat_vals_all_vertices:
                new_u_real_atverts_pt = (self.u_real_atverts_pt.float() * ~self.unknown_heatverts_mask) + \
                    (self.optimized_heat_vals.float() * self.unknown_heatverts_mask)
            else:
                new_u_real_atverts_pt = self.optimized_heat_vals.float()
            
            # Prepare MinvL matrix
            L = torch.sparse_coo_tensor(self.MinvL_locs, (self.MinvLopti)**1, size=(V, V))
            L += L.t()
            vals = torch.sparse.sum(L, dim=-1).to_dense()
            indices = torch.arange(V, device='cuda')
            idx = torch.stack([indices, indices], dim=0)
            L = L - torch.sparse_coo_tensor(idx, vals, (V, V))
            midx = torch.stack([torch.arange(V), torch.arange(V)], dim=0)
            Minv = torch.sparse_coo_tensor(midx, (1.0/(self.mass_opti**2)), size=(V, V))
            MinvL = torch.sparse.mm(Minv, L)

            # Compute time step
            u_tplus1 = new_u_real_atverts_pt.permute(2,0,1)[:, :, 1:]
            ushape = u_tplus1.shape
            u_tplus1 = u_tplus1.reshape(ushape[0], ushape[1]*ushape[2])
            
            u_t = new_u_real_atverts_pt.permute(2,0,1)[:, :, :-1]
            u_t = u_t.reshape(ushape[0], ushape[1]*ushape[2])
                
            du = (u_tplus1 - u_t)
            du = (du).reshape(ushape[0], ushape[1], ushape[2]).permute(1,2,0)

            tMinvL = t*MinvL
            tMinvLT_t1 = torch.sparse.mm(tMinvL, u_tplus1).reshape(ushape[0], ushape[1], ushape[2]).permute(1,2,0)
            lhs_terms = du - tMinvLT_t1

            # Convection terms
            if self.params.ADD_CONVECTION or self.params.optim_convection_coeff:
                convection_coeff = (self.convection_coeff_opti)**2
                self.convection_coeff_np = convection_coeff.detach().cpu().numpy()
                u_surr_diff = (u_tplus1 - constants.AMBIENT_TEMP).reshape(ushape[0], ushape[1], ushape[2]).permute(1,2,0)
                convection_term = t*convection_coeff*u_surr_diff
                lhs_terms += convection_term

            # Heat source term
            if self.params.optim_heat_source:
                heat_source_opti = self.heat_source_opti
                heat_source_vals = (heat_source_opti)**2
                self.heat_source_vals_np = heat_source_vals.detach().cpu().numpy()
                heat_source_term = t*heat_source_vals
                lhs_terms -= heat_source_term
            
            # Radiation term
            if self.params.ADD_RADIATION:
                heat_capacity = (self.heat_capacity_opti)**2
                self.heat_capacity_np = heat_capacity.detach().cpu().numpy()
                rad_coeff = t*4*constants.SIGMA*constants.EPSILON/(heat_capacity)
                u_tplus1_pow4 = u_tplus1*(constants.AMBIENT_TEMP**3)
                surr_diff = (u_tplus1_pow4 - (constants.AMBIENT_TEMP**4))
                rad_term = (rad_coeff*surr_diff).reshape(ushape[0], ushape[1], ushape[2]).permute(1,2,0)
                lhs_terms += rad_term

            # PDE difference and loss computation
            pde_diff = lhs_terms.permute(0,2,1)
            pde_diff = pde_diff[self.valid_projected_locs_forview]
            sim_heat_loss = (pde_diff**2).mean()
            self.sim_heat_loss_np = sim_heat_loss.item()
            
            # Construct total loss
            loss = 0.0
            
            # Regularization terms
            if self.params.optim_heat_vals:
                heat_reg_loss = torch.linalg.norm(self.optimized_heat_vals - self.u_real_atverts_pt, dim=1).mean()
                self.heat_reg_loss_np = heat_reg_loss.item()
                loss += self.Lopti_params.lambda_heat_reg * heat_reg_loss

            if self.params.optim_laplacian or self.params.optim_thermal_diff:
                loss += self.Lopti_params.lambda_pde * sim_heat_loss
            
            # Backpropagation and optimization step
            loss.backward()
            self.Loptimizer.step()
            self.Llr_scheduler.step()

            # Logging
            if writer is not None:
                if self.params.optim_heat_vals:
                    writer.add_scalar(f'HeatOptim/LaplacianOpti/heat_reg_loss/{self.img_size[0]}x{self.img_size[1]}', self.heat_reg_loss_np, global_step=global_step)

                if self.params.optim_thermal_cond:
                    writer.add_scalar(f'ThermalCondOptim/LaplacianOpti/{self.img_size[0]}x{self.img_size[1]}', (self.thermal_cond**2).item(), global_step=global_step)

                if self.params.optim_laplacian or self.params.optim_thermal_diff:
                    cost = self.sim_heat_loss_np
                    writer.add_scalar(f'cost_function/LaplacianOpti/{self.img_size[0]}x{self.img_size[1]}', cost, global_step=global_step)
                    
                if self.params.optim_heat_source:
                    writer.add_scalar(f'HeatSourceOptim/LaplacianOpti/Min(Watts)/{self.img_size[0]}x{self.img_size[1]}', self.heat_source_vals_np.min(), global_step=global_step)
                    writer.add_scalar(f'HeatSourceOptim/LaplacianOpti/Max(Watts)/{self.img_size[0]}x{self.img_size[1]}', self.heat_source_vals_np.max(), global_step=global_step)

                if self.params.optim_heat_capacity:
                    writer.add_scalar(f'HeatCapacityOptim/LaplacianOpti/{self.img_size[0]}x{self.img_size[1]}', self.heat_capacity_np, global_step=global_step)
                
                if self.params.optim_convection_coeff:
                    writer.add_scalar(f'ConvectionCoeffOptim/LaplacianOpti/{self.img_size[0]}x{self.img_size[1]}', self.convection_coeff_np, global_step=global_step)

                global_step += 1

        # Convert MinvL to full matrix
        device = MinvL.device
        H, W = self.params.max_img_size
        N_full = H * W

        pix_rc = self.cat_corres_pixel_coords_init.to(device=device).long()  # (V, 2)
        rows = pix_rc[:, 0]   # (V,)
        cols = pix_rc[:, 1]   # (V,)
        flat_idx = rows * W + cols    # (V,)
        idx_valid = MinvL.indices()   # shape: (2, nnz)
        vals_valid = MinvL.values()   # shape: (nnz,)
        row_valid = idx_valid[0]              # (nnz,)
        col_valid = idx_valid[1]              # (nnz,)
        row_full = flat_idx[row_valid]        # (nnz,)
        col_full = flat_idx[col_valid]        # (nnz,)

        full_indices = torch.stack([row_full, col_full], dim=0)  # (2, nnz)
        MinvL_full = torch.sparse_coo_tensor(full_indices, vals_valid, size=(N_full, N_full), device=device, dtype=MinvL.dtype)

        return MinvL_full#, phi_k


    def get_heatsource_img(self):
        
        with torch.no_grad():
            heat_source_opti = self.heat_source_opti
            heat_source_vals = (heat_source_opti)**2
            heat_source_vals_np = heat_source_vals.detach().cpu().numpy()

        cat_points_image_num_init_np = self.cat_points_image_num_init.cpu().numpy()
        valid_indx_init = np.where(cat_points_image_num_init_np[:, 0] == 0)[0]
        heat_source_optimized_img = np.full((heat_source_vals_np.shape[0], self.img_size[0], self.img_size[1]), np.NaN)
        cat_corres_pixel_coords_init_np = self.cat_corres_pixel_coords_init.detach().cpu().numpy()
        for it in range(heat_source_vals_np.shape[0]):
            heat_source_optimized_img[it, cat_corres_pixel_coords_init_np[valid_indx_init, 0], cat_corres_pixel_coords_init_np[valid_indx_init, 1]] \
                = heat_source_vals_np[it, 0, valid_indx_init]
        return heat_source_optimized_img
    

    def get_optimization_results(self):

        optimization_results = {}
        if self.params.optim_laplacian:
            optimization_results['MinvLopti'] = self.MinvLopti.detach().cpu()
            optimization_results['MinvL_locs'] = self.MinvL_locs
            optimization_results['mass_opti'] = self.mass_opti.detach().cpu()
            V = self.verts_sim_pt.shape[0]
            L = torch.sparse_coo_tensor(self.MinvL_locs, (self.MinvLopti)**1, size=(V, V))
            L += L.t()
            vals = torch.sparse.sum(L, dim=-1).to_dense()
            indices = torch.arange(V, device='cuda')
            idx = torch.stack([indices, indices], dim=0)
            L = L - torch.sparse_coo_tensor(idx, vals, (V, V))
            optimization_results['L'] = L.detach().cpu()

        if self.params.optim_heat_vals:
            # optim_variables += [self.optimized_heat_vals]
            optimization_results['optimized_heat_vals'] = self.optimized_heat_vals.detach().cpu()
        if self.params.optim_thermal_diff:
            # optim_variables += [self.K_opti]
            optimization_results['thermal_diff'] = self.thermal_diff.detach().cpu()
        if self.params.optim_thermal_cond:
            optimization_results['thermal_cond'] = self.thermal_cond.detach().cpu()
        if self.params.optim_heat_source:
            # optim_variables += [self.heat_source_opti]
            optimization_results['heat_source_opti'] = self.heat_source_opti.detach().cpu()
        if self.params.optim_heat_capacity:
            # optim_variables += [self.heat_capacity_opti]
            optimization_results['heat_capacity_opti'] = self.heat_capacity_opti.detach().cpu()
        if self.params.optim_convection_coeff:
            # optim_variables += [self.convection_coeff_opti]
            optimization_results['convection_coeff_opti'] = self.convection_coeff_opti.detach().cpu()
        
        optimization_results['verts_sim_pt'] = self.verts_sim_pt.detach().cpu()
        optimization_results['faces_sim_pt'] = self.faces_sim_pt.detach().cpu()
        optimization_results['cat_points_image_num_init'] = self.cat_points_image_num_init.detach().cpu()
        optimization_results['cat_corres_pixel_coords_init'] = self.cat_corres_pixel_coords_init.detach().cpu()
        return optimization_results
    

    def set_optimization_results(self, optimization_results):

        if self.params.optim_laplacian:
            self.MinvLopti = optimization_results['MinvLopti'].to(self.device)
            self.MinvLopti.requires_grad = True
        if self.params.optim_heat_vals:
            self.optimized_heat_vals = optimization_results['optimized_heat_vals'].to(self.device)
            self.optimized_heat_vals.requires_grad = True
        if self.params.optim_thermal_diff:
            self.thermal_diff = optimization_results['thermal_diff'].to(self.device)
            self.thermal_diff.requires_grad = True
        if self.params.optim_heat_source:
            self.heat_source_opti = optimization_results['heat_source_opti'].to(self.device)
            self.heat_source_opti.requires_grad = True
        if self.params.optim_heat_capacity:
            self.heat_capacity_opti = optimization_results['heat_capacity_opti'].to(self.device)
            self.heat_capacity_opti.requires_grad = True
        if self.params.optim_convection_coeff:
            self.convection_coeff_opti = optimization_results['convection_coeff_opti'].to(self.device)
            self.convection_coeff_opti.requires_grad = True

    
class ShapeOptimizationWrapper(OptimizationWrapper):

    def initialize_Sopti_variables(self, opti_params):
        with torch.no_grad():
            
            self.M_lsteps = compute_matrix(self.verts_sim_pt, self.faces_sim_pt, opti_params.lambda_, cotan=True)
            
            ### Optimize Thermal Diffusivity ###
            if self.params.optim_thermal_diff:
                if self.alpha_opti is None:
                    self.alpha_opti = torch.tensor(np.sqrt(constants.TDiff[self.params.obj_material]), dtype=torch.float32, device=self.device)
                self.alpha_opti.requires_grad = True
            else:
                self.alpha_opti = torch.tensor(np.sqrt(constants.TDiff[self.params.obj_material]), dtype=torch.float32, device=self.device)

            ### Optimize Heat values at Unknown Locations ###
            if self.params.optim_heat_vals:
                if self.optimized_heat_vals is None:
                    self.optimized_heat_vals = self.u_real_atverts_pt.clone().detach()
                
                if opti_params.heat_value_opti:
                    self.optimized_heat_vals.requires_grad = True
                else:
                    self.optimized_heat_vals.requires_grad = False
            else:
                self.optimized_heat_vals = self.u_real_atverts_pt.clone().detach().requires_grad_(False)

            # Sampling probability for the time dimension
            dT_dt = (self.optimized_heat_vals[:, 1:] - self.optimized_heat_vals[:, :-1]).detach().abs()
            dT_dt = dT_dt.permute(0, 2, 1)
            self.dT_dt_weights = dT_dt / torch.sum(dT_dt, dim=-1, keepdim=True)
                
            self.unknown_heatverts_mask = torch.zeros_like(self.u_real_atverts_pt, dtype=torch.bool, device=self.device)
            self.unknown_heatverts_mask[:, :, self.invalid_projected_locs] = True

            ### Optimize Heat Capacity ###
            if self.params.optim_heat_capacity:
                if self.heat_capacity_opti  is None:
                    self.heat_capacity_opti = torch.tensor(np.sqrt(self.params.SURFACE_THICKNESS*self.params.RHO*self.params.C), dtype=torch.float32, device=self.device)
                if opti_params.heat_capacity_opti:
                    self.heat_capacity_opti.requires_grad = True
                else:
                    self.heat_capacity_opti.requires_grad = False
            else:
                self.heat_capacity_opti = torch.tensor(np.sqrt(self.params.SURFACE_THICKNESS*self.params.RHO*self.params.C), dtype=torch.float32, device=self.device)
            
            ### Optimize Convection Coefficient ###
            if self.params.optim_convection_coeff:
                if self.convection_coeff_opti is None:
                    self.convection_coeff_opti = torch.tensor(np.sqrt(constants.CONVECTION_COEFF), dtype=torch.float32, device=self.device)
                if opti_params.convection_coeff_opti:
                    self.convection_coeff_opti.requires_grad = True
                else:
                    self.convection_coeff_opti.requires_grad = False
            else:
                self.convection_coeff_opti = torch.tensor(np.sqrt(constants.CONVECTION_COEFF), dtype=torch.float32, device=self.device)

            ### Optimize Heat Source ###
            if self.params.optim_heat_source and self.heat_source_opti is None:
                self.heat_source_opti = torch.rand([self.u_real_atverts_pt.shape[0], 1, self.verts_sim_pt.shape[0]], device=self.device, requires_grad=True)
                self.heat_source_shape = self.heat_source_opti.shape
            elif self.params.optim_heat_source and not opti_params.heat_source_opti:
                self.heat_source_opti.requires_grad = False
                self.heat_source_shape = self.heat_source_opti.shape

            ### Optimize Mesh Scale Factor ###
            if self.params.optim_scale_factor:
                self.scale_factor = torch.tensor(1.0, device=self.device, requires_grad=True)
            else:
                self.scale_factor = torch.tensor(1.0, device=self.device, requires_grad=False)

            ### Optimize Shap e Offsets ###
            if self.params.optim_shape:
                self.update_dir_offset = torch.zeros([self.verts_sim_pt.shape[0]], device=self.device, requires_grad=True)
            else:
                self.update_dir_offset = torch.zeros([self.verts_sim_pt.shape[0]], device=self.device, requires_grad=False)
            self.depth_lsteps = to_differential(self.M_lsteps, self.init_xy_depth[:, -1])
            
            if self.params.use_image_symmetry_constraint:
                meanxy = torch.ceil(torch.mean(self.cat_corres_pixel_coords_init.to(dtype=torch.float32), axis=0)).to(dtype=torch.long)
                vertex_num_img = render_vertices_as_image(torch.arange(self.verts_sim_pt.shape[0])[None, :], \
                                                          self.cat_corres_pixel_coords_init, self.cat_points_image_num_init, self.img_size, fill_value=-1)[0][0]
                
                if self.params.symmetric_about_axis == 0 or self.params.symmetric_about_axis == -1:
                    vleft_img = vertex_num_img[:meanxy[0], :]
                    vright_img = vertex_num_img[meanxy[0]:, :]
                    vright_img_flipped = torch.flip(vright_img, dims=[0])
                    vleft_img_flipped = torch.flip(vleft_img, dims=[0])
                    vsymmetric_vert_img = torch.full([self.img_size[0], self.img_size[1]], -1, device=self.device, dtype=torch.long)
                    min_himg_size = min(vleft_img.shape[0], vright_img.shape[0])
                    vsymmetric_vert_img[meanxy[0] - min_himg_size:meanxy[0], :] = vright_img_flipped[-min_himg_size:, :]
                    vsymmetric_vert_img[meanxy[0]:meanxy[0] + min_himg_size, :] = vleft_img_flipped[:min_himg_size, :]
                    # Get symmetric index for vertices
                    self.vsymmetric_vert_indx = torch.arange(self.verts_sim_pt.shape[0], device=self.device, dtype=torch.long)
                    vsymmetric_vert_indx = (vsymmetric_vert_img[self.cat_corres_pixel_coords_init[:, 0], self.cat_corres_pixel_coords_init[:, 1]]).to(dtype=torch.long)
                    self.vsymmetric_vert_indx[vsymmetric_vert_indx != -1] = vsymmetric_vert_indx[vsymmetric_vert_indx != -1]

                if self.params.symmetric_about_axis == 1 or self.params.symmetric_about_axis == -1:
                    hleft_img = vertex_num_img[:, :meanxy[1]]
                    hright_img = vertex_num_img[:, meanxy[1]:]
                    hright_img_flipped = torch.flip(hright_img, dims=[1])
                    hleft_img_flipped = torch.flip(hleft_img, dims=[1])
                    hsymmetric_vert_img = torch.full([self.img_size[0], self.img_size[1]], -1, device=self.device, dtype=torch.long)
                    min_wimg_size = min(hleft_img.shape[1], hright_img.shape[1])
                    hsymmetric_vert_img[:, meanxy[1] - min_wimg_size:meanxy[1]] = hright_img_flipped[:, -min_wimg_size:]
                    hsymmetric_vert_img[:, meanxy[1]:meanxy[1] + min_wimg_size] = hleft_img_flipped[:, :min_wimg_size]
                    # Get symmetric index for vertices
                    self.hsymmetric_vert_indx = torch.arange(self.verts_sim_pt.shape[0], device=self.device, dtype=torch.long)
                    hsymmetric_vert_indx = (hsymmetric_vert_img[self.cat_corres_pixel_coords_init[:, 0], self.cat_corres_pixel_coords_init[:, 1]]).to(dtype=torch.long)
                    self.hsymmetric_vert_indx[hsymmetric_vert_indx != -1] = hsymmetric_vert_indx[hsymmetric_vert_indx != -1]


    def initialize_Soptimizers(self, opti_params):
        optim_variables = []
        if self.params.optim_shape:
            optim_variables += [self.update_dir_offset]
        if self.params.optim_heat_vals and opti_params.heat_value_opti:
            optim_variables += [self.optimized_heat_vals]
        if self.params.optim_scale_factor:
            optim_variables += [self.scale_factor]
        if self.params.optim_heat_source and opti_params.heat_source_opti:
            optim_variables += [self.heat_source_opti]
        if self.params.optim_heat_capacity and opti_params.heat_capacity_opti:
            optim_variables += [self.heat_capacity_opti]
        if self.params.optim_convection_coeff and opti_params.convection_coeff_opti:
            optim_variables += [self.convection_coeff_opti]
        if self.params.optim_thermal_cond:
            optim_variables += [self.alpha_opti]
        if self.params.optim_thermal_diff:
            optim_variables += [self.K_opti]
        self.optimizer = AdamUniform(optim_variables, opti_params.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opti_params.lr_step_size, gamma=opti_params.lr_gamma)


    def initialize_Soptimaztion(self, opti_params):

        self.initialize_Sopti_variables(opti_params)
        self.initialize_Soptimizers(opti_params)
        self.opti_params = opti_params


    def run_Soptimization(self, writer=None, verts_numpy=[], faces_numpy=[], losses=None, filename='dummy', exp_name='v1', global_step=0, p_map=None, q_map=None):

        # Pre-process before optimization
        store_mesh_every = 20
        save_mesh_pkl_every = 50
        t = self.params.NUM_FRAME_DELTA/60.0

        if losses is None:
            losses = []

        faces_numpy.append(self.faces_sim_np)
        
        # Iterative roop
        for step in tqdm(range(self.opti_params.num_iterations)):
            if not self.params.unproject_and_update:
                new_ulsteps = self.u_lsteps.clone()
                if self.params.optim_shape:
                    new_ulsteps[:, -1] = new_ulsteps[:, -1] + (self.update_dir_offset * self.update_direction[:,-1])
                verts_sim_pt = from_differential(self.M_lsteps, new_ulsteps, 'Cholesky')
                verts_sim_pt_act = (verts_sim_pt / 2.0) * constants.IMAGE_WIDTH_METRIC * (self.scale_factor**2)
            else:
                new_ulsteps = self.depth_lsteps.clone()

                if self.params.limit_zvals:
                    new_ulsteps = new_ulsteps + ((torch.tanh(self.update_dir_offset)) * (self.params.zfar - self.params.znear))
                else:
                    new_ulsteps = new_ulsteps + self.update_dir_offset
                if self.M_lsteps.dtype == torch.float64:
                    update_depth_vals = from_differential(self.M_lsteps, new_ulsteps, 'CholeskyD')
                else:
                    update_depth_vals = from_differential(self.M_lsteps, new_ulsteps, 'Cholesky')

                curr_xy_depth = self.init_xy_depth.clone()
                curr_xy_depth[:, -1] = update_depth_vals

                # Unproject and multiply by 1000 to convert things to mm from m
                verts_sim_pt = (self.cameras.unproject_points(curr_xy_depth, world_coordinates=True, from_ndc=False))
                verts_sim_pt_act = verts_sim_pt * (self.scale_factor**2) * 1000.0
            
            # Prepare heat values at vertices
            if not self.params.optim_heat_vals_all_vertices:
                new_u_real_atverts_pt = (self.u_real_atverts_pt.float() * ~self.unknown_heatverts_mask) + \
                    (self.optimized_heat_vals.float() * self.unknown_heatverts_mask)
            else:
                new_u_real_atverts_pt = self.optimized_heat_vals.float()

            # Create Mesh and Laplacian
            curr_mesh = Meshes(verts=[verts_sim_pt], faces=[self.faces_sim_pt])
            L = largesteps.geometry.laplacian_cot2(verts_sim_pt_act, self.faces_sim_pt)
            Ma = massmatrix_fast(verts_sim_pt_act, self.faces_sim_pt)
            Macoalesced = Ma.coalesce()
            M_indx = Macoalesced.indices()
            M_val = Macoalesced.values()
            Mv = torch.sparse_coo_tensor(M_indx, M_val*self.params.SURFACE_THICKNESS, size=Ma.shape)

            # Radiation and Convection terms
            ii = torch.arange(L.shape[0], device=self.device)        
            I = torch.sparse_coo_tensor(torch.stack([ii, ii]), torch.ones(L.shape[0], device=self.device), dtype=torch.float32, size=L.shape)
            ZERO_MAT = torch.sparse_coo_tensor(torch.stack([ii, ii]), torch.zeros(L.shape[0], device=self.device), dtype=torch.float32, size=L.shape)
            lrad = Mv * 4*constants.SIGMA * constants.EPSILON * (constants.AMBIENT_TEMP**3) if self.params.ADD_RADIATION else ZERO_MAT
            rrad = Mv * 4*constants.SIGMA * constants.EPSILON * (constants.AMBIENT_TEMP**4) if self.params.ADD_RADIATION else ZERO_MAT

            convection_coeff = (self.convection_coeff_opti)**2
            lconv = torch.sparse.mm(Mv, convection_coeff*I) if self.params.ADD_CONVECTION else ZERO_MAT
            rconv = torch.sparse.mm(Mv, convection_coeff*constants.AMBIENT_TEMP*I) if self.params.ADD_CONVECTION else ZERO_MAT

            # Heat Diffusion terms
            tL = t*L
            if self.params.optim_thermal_diff:
                tL_indx = tL._indices()
                tL_val = tL._values()
                alphatL = torch.sparse_coo_tensor(tL_indx, (self.alpha_opti**2)*tL_val, size=L.shape)
            else:
                alphatL = (self.alpha_opti**2) * tL

            heat_capacity = (self.heat_capacity_opti)**2
            ldiff = Mv - alphatL
            lhs_val = ldiff + t*(lrad/heat_capacity + lconv)

            # Construct time stepping lhs and rhs
            u_tplus1 = new_u_real_atverts_pt.permute(2,0,1)[:, :, 1:]
            ushape = u_tplus1.shape
            u_tplus1 = u_tplus1.reshape(ushape[0], ushape[1]*ushape[2])
            lhs_val = torch.sparse.mm(lhs_val, u_tplus1).reshape(ushape[0], ushape[1], ushape[2]).permute(1,2,0)

            rhs_rad_conv = torch.sparse.mm(t*(rrad/heat_capacity + rconv), torch.ones([ushape[0], ushape[1]*ushape[2]]))
            u_t = new_u_real_atverts_pt.permute(2,0,1)[:, :, :-1]
            u_t = u_t.reshape(ushape[0], ushape[1]*ushape[2])
            rhs_val = (torch.sparse.mm(Mv, u_t) + rhs_rad_conv).reshape(ushape[0], ushape[1], ushape[2]).permute(1, 2, 0)
            
            # Add heat source term if optimizing for it
            if self.params.optim_heat_source:
                heat_source_opti = self.heat_source_opti
                heat_source_vals = (heat_source_opti)**2
                heat_source_rhs = t*heat_source_vals
                heat_source_rhs = torch.sparse.mm(Mv, heat_source_rhs.permute(2,0,1).reshape(self.heat_source_shape[2], -1))
                heat_source_rhs = heat_source_rhs.reshape(self.heat_source_shape[2], self.heat_source_shape[0], self.heat_source_shape[1]).permute(1,2,0)
                rhs_val += heat_source_rhs
            
            # PDE difference and loss computation
            pde_diff = (lhs_val - rhs_val).permute(0,2,1)
            pde_diff = pde_diff[self.valid_projected_locs_forview].abs()
            pde_diff = pde_diff.mean(dim=-1)
            sim_heat_loss = pde_diff.mean()
            
            # Construct total loss
            loss = 0.0

            # Regularization terms
            if self.params.optim_shape:
                edge_loss = mesh_edge_loss(curr_mesh)
                normal_reg_term = mesh_normal_consistency(curr_mesh)
                laplacian_loss = mesh_laplacian_smoothing(curr_mesh, method='uniform')

                loss += self.opti_params.mesh_edge_reg*edge_loss
                if step > self.opti_params.activate_normal_reg_after_step:
                    loss += (self.opti_params.normal_reg*normal_reg_term)
                if step > self.opti_params.activate_laplacian_reg_after_step:
                    loss += (self.opti_params.laplacian_reg*laplacian_loss)

            # PDE regularization term
            if self.params.optim_shape or self.params.optim_thermal_diff:
                loss += self.opti_params.lambda_pde * sim_heat_loss

            # Symmetric vertices regularization
            symmetric_verts_loss_np = 0.0
            if self.params.use_image_symmetry_constraint and (step < self.opti_params.num_steps_for_symmetric_verts or self.opti_params.num_steps_for_symmetric_verts == -1):
                if self.params.symmetric_about_axis == 0 or self.params.symmetric_about_axis == -1:
                    vsymmetric_verts_depth = curr_xy_depth[self.vsymmetric_vert_indx, -1]
                    vsymmetric_verts_loss = torch.mean((curr_xy_depth[:, -1] - vsymmetric_verts_depth)**2)
                    loss += self.opti_params.lambda_symmetric_verts * vsymmetric_verts_loss
                    symmetric_verts_loss_np += vsymmetric_verts_loss.item()
                if self.params.symmetric_about_axis == 1 or self.params.symmetric_about_axis == -1:
                    hsymmetric_verts_depth = curr_xy_depth[self.hsymmetric_vert_indx, -1]
                    hsymmetric_verts_loss = torch.mean((curr_xy_depth[:, -1] - hsymmetric_verts_depth)**2)
                    loss += self.opti_params.lambda_symmetric_verts * hsymmetric_verts_loss
                    symmetric_verts_loss_np += hsymmetric_verts_loss.item()

            # ================= Additional Loss =================
            # Convert MinvL to full matrix
            device = L.device
            L = L.coalesce()
            H, W = self.params.max_img_size
            N_full = H * W

            pix_rc = self.cat_corres_pixel_coords_init.to(device=device).long()  # (V, 2)
            rows = pix_rc[:, 0]   # (V,)
            cols = pix_rc[:, 1]   # (V,)
            flat_idx = rows * W + cols    # (V,)
            idx_valid = L.indices()   # shape: (2, nnz)
            vals_valid = L.values()   # shape: (nnz,)
            row_valid = idx_valid[0]              # (nnz,)
            col_valid = idx_valid[1]              # (nnz,)
            row_full = flat_idx[row_valid]        # (nnz,)
            col_full = flat_idx[col_valid]        # (nnz,)
            full_indices = torch.stack([row_full, col_full], dim=0)  # (2, nnz)
            L_full = torch.sparse_coo_tensor(full_indices, vals_valid, size=(N_full, N_full), device=device, dtype=L.dtype)

            # Normal loss
            pq_loss = None
            if (p_map is not None) and (q_map is not None):
                p_map_np = np.array(p_map, copy=True)
                q_map_np = np.array(q_map, copy=True)

                # Resize p_map and q_map to match H, W
                if p_map_np.shape != (H, W):
                    p_map_np = cv2.resize(p_map_np.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                if q_map_np.shape != (H, W):
                    q_map_np = cv2.resize(q_map_np.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

                p_map_t = torch.from_numpy(p_map_np.astype(np.float32)).to(device=self.device)
                q_map_t = torch.from_numpy(q_map_np.astype(np.float32)).to(device=self.device)

                # Compute vertex normals
                vertex_normals = compute_vertex_normals_torch(verts_sim_pt, self.faces_sim_pt)  # (V,3)
                nx = vertex_normals[:, 0]
                ny = vertex_normals[:, 1]
                nz = vertex_normals[:, 2]

                eps = 1e-6
                p_curr = -nx / (nz + eps)
                q_curr = -ny / (nz + eps)

                # Sample target p,q using image coordinates (row, col) corresponding to each vertex
                pix_rc = self.cat_corres_pixel_coords_init.to(device=self.device).long()  # (V,2)
                rows = pix_rc[:, 0].clamp(0, H-1)
                cols = pix_rc[:, 1].clamp(0, W-1)

                p_tgt_v = p_map_t[rows, cols]
                q_tgt_v = q_map_t[rows, cols]

                valid_mask = torch.isfinite(p_tgt_v) & torch.isfinite(q_tgt_v)
                if valid_mask.any():
                    pq_loss = ((p_curr[valid_mask] - p_tgt_v[valid_mask])**2 +
                               (q_curr[valid_mask] - q_tgt_v[valid_mask])**2).mean()
                    loss += self.opti_params.lambda_normal * pq_loss


            # Backpropagation and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # Normal pytorch optimization step
            with torch.no_grad():
                losses.append(loss.item())
                if self.params.optim_shape:
                    if step % store_mesh_every == 0:
                        verts_numpy.append(verts_sim_pt.detach().cpu().numpy())
                    
                    if self.opti_params.reduce_diffusivity and (step +1) % 1000 == 0 and step > 2000:
                        self.alpha_opti = torch.minimum(torch.sqrt((self.alpha_opti**2) * 2.0), torch.sqrt(torch.tensor(1.0)))
                        
                    if (step+1) % save_mesh_pkl_every == 0:
                        save_meshes_pkl(verts_numpy, faces_numpy, workspace_path, filename, exp_name)

            # Logging
            if writer is not None:
                if self.params.optim_shape:
                    writer.add_scalar(f'laplacian_smoothing/{self.img_size[0]}x{self.img_size[1]}', laplacian_loss.item(), global_step=global_step)
                    writer.add_scalar(f'normal_consistency/{self.img_size[0]}x{self.img_size[1]}', normal_reg_term.item(), global_step=global_step)
                    writer.add_scalar(f'mesh_edge_loss/{self.img_size[0]}x{self.img_size[1]}', edge_loss.item(), global_step=global_step)

                if self.params.optim_shape or self.params.optim_thermal_diff:
                    cost = sim_heat_loss.detach().cpu().numpy()
                    writer.add_scalar(f'cost_function/ShapeOpti/{self.img_size[0]}x{self.img_size[1]}', cost, global_step=global_step)
                
                if self.params.optim_scale_factor:
                    writer.add_scalar(f'scale_factor/{self.img_size[0]}x{self.img_size[1]}', self.scale_factor.item(), global_step=global_step)

                if self.params.optim_heat_source:
                    writer.add_scalar(f'HeatSourceOptim/ShapeOpti/Min(Watts)/{self.img_size[0]}x{self.img_size[1]}', heat_source_vals.min().item(), global_step=global_step)
                    writer.add_scalar(f'HeatSourceOptim/ShapeOpti/Max(Watts)/{self.img_size[0]}x{self.img_size[1]}', heat_source_vals.max().item(), global_step=global_step)
                
                if self.params.optim_heat_capacity and self.opti_params.heat_capacity_opti:
                    writer.add_scalar(f'HeatCapacityOptim/ShapeOpti/{self.img_size[0]}x{self.img_size[1]}', heat_capacity.item(), global_step=global_step)

                if self.params.optim_convection_coeff and self.opti_params.convection_coeff_opti:
                    writer.add_scalar(f'ConvectionCoeffOptim/ShapeOpti/{self.img_size[0]}x{self.img_size[1]}', convection_coeff.item(), global_step=global_step)

                if self.params.optim_thermal_diff:
                    writer.add_scalar(f'AlphaOptim/ShapeOpti/{self.img_size[0]}x{self.img_size[1]}', (self.alpha_opti**2).item(), global_step=global_step)

                if self.params.use_image_symmetry_constraint:
                    writer.add_scalar(f'SymmetricVertsLoss/ShapeOpti/{self.img_size[0]}x{self.img_size[1]}', symmetric_verts_loss_np, global_step=global_step)

                global_step += 1
        
        with torch.no_grad():
            self.curr_vertex_normals = compute_vertex_normals_torch(verts_sim_pt, self.faces_sim_pt)

        return verts_numpy, faces_numpy, losses, L_full
    

    def run_SFLS(self, losses=None, shading_img=None, ps_gt=None, qs_gt=None, L=None, image_size=None, p_map=None, q_map=None):

        H, W = image_size
        
        if losses is None:
            losses = []

        # Eta
        if shading_img.ndim == 3:
            shading_img_2d = shading_img[0]
        else:
            shading_img_2d = shading_img
        if shading_img_2d.shape != (H, W):
            shading_img_2d = cv2.resize(
                shading_img_2d.astype(np.float32),
                (W, H),  # cv2 expects (width, height)
                interpolation=cv2.INTER_LINEAR)
        eta_img = np.nan_to_num(shading_img_2d.astype(np.float32), nan=0.0)
        target_eta = eta_img[1:H-1, 1:W-1]
        
        # Cotan Laplacian
        ind_v, ind_a, ind_b, ind_c, ind_d, ind_e, ind_f, ind_g, ind_h = get_ngbr_indices(W, H)  # numpy
        ngbr_inds = np.vstack((ind_a, ind_b, ind_c, ind_d, ind_e, ind_f)).T.reshape(-1)
        ver_inds = np.repeat(ind_v, 6)
        L_np = L.to_dense().detach().cpu().numpy()
        L_full = L_np[ver_inds, ngbr_inds].reshape(-1, 6)
        lap_int_reshape = L_full.reshape(H-2, W-2, 6)

        # Weights (Discrete Fundamental Forms)
        w1 = (lap_int_reshape[..., 0]+lap_int_reshape[..., 3])/2.0
        w2 = (lap_int_reshape[..., 1]+lap_int_reshape[..., 4])/2.0
        w3 = (lap_int_reshape[..., 2]+lap_int_reshape[..., 5])/2.0
        
        D = w1 + w2 + 2*w3
        disc = np.maximum(0.0, D**2 - 4)
        x1 = (D + np.sqrt(disc)) / 2
        x2 = (D - np.sqrt(disc)) / 2
        x = np.maximum(x1, x2)

        # Solve for p and q
        P = np.sqrt(np.maximum(0.0, (w1 + w3) * x - 1))
        Q = np.sqrt(np.maximum(0.0, (w2 + w3) * x - 1))
        sign_q1 = np.where(w3 < 0, 1.0, -1.0)
        p1, p2 = P, -P
        q1, q2 = sign_q1 * Q, -sign_q1 * Q

        # Choose between (p1, q1) and (p2, q2) based on shading error
        eta1 = compute_shading_value(p1, q1, ps_gt, qs_gt)
        eta2 = compute_shading_value(p2, q2, ps_gt, qs_gt)
        target_eta = eta_img[1:H-1, 1:W-1]

        e1 = np.abs(eta1 - target_eta)
        e2 = np.abs(eta2 - target_eta)
        mask_choose1 = e1 <= e2

        p_final = np.where(mask_choose1, p1, p2)
        q_final = np.where(mask_choose1, q1, q2)
        
        _, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(p_final, cmap='coolwarm')
        ax[0].set_title("p_final")
        ax[0].axis("off")
        ax[1].imshow(q_final, cmap='coolwarm')
        ax[1].set_title("q_final")
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()
    
        # Update p_map and q_map to full image size
        p_full = np.full((H, W), np.nan, dtype=np.float32)
        q_full = np.full((H, W), np.nan, dtype=np.float32)
        p_full[1:-1, 1:-1] = p_final
        q_full[1:-1, 1:-1] = q_final
        p_map = p_full
        q_map = q_full 

        return p_map, q_map


    def get_depth_image(self):
        with torch.no_grad():
            if not self.params.unproject_and_update:
                new_ulsteps = self.u_lsteps.clone()
                if self.params.optim_shape:
                    new_ulsteps[:, -1] = new_ulsteps[:, -1] + (self.update_dir_offset * self.update_direction[:,-1])

                verts_sim_pt = from_differential(self.M_lsteps, new_ulsteps, 'Cholesky')

                verts_sim_pt_act = (verts_sim_pt / 2.0) * constants.IMAGE_WIDTH_METRIC * (self.scale_factor**2)
                return None
            else:
                new_ulsteps = self.depth_lsteps.clone()

                if self.params.limit_zvals:
                    new_ulsteps = new_ulsteps + ((torch.tanh(self.update_dir_offset)) * (self.params.zfar - self.params.znear))
                else:
                    new_ulsteps = new_ulsteps + self.update_dir_offset

                if self.M_lsteps.dtype == torch.float64:
                    update_depth_vals = from_differential(self.M_lsteps, new_ulsteps, 'CholeskyD')
                else:
                    update_depth_vals = from_differential(self.M_lsteps, new_ulsteps, 'Cholesky')

                curr_xy_depth = self.init_xy_depth.clone()
                curr_xy_depth[:, -1] = update_depth_vals

                depth_images = np.full([self.params.num_views, self.img_size[0], self.img_size[1]], 100.0)
                cat_points_image_num_init_np = self.cat_points_image_num_init.cpu().numpy()
                cat_corres_pixel_coords_init_np = self.cat_corres_pixel_coords_init.cpu().numpy()
                for i in range(self.params.num_views):
                    valid_indx_init = np.where(cat_points_image_num_init_np[:, i] == i)[0]
                    depth_images[i, cat_corres_pixel_coords_init_np[valid_indx_init, 0], cat_corres_pixel_coords_init_np[valid_indx_init, 1]] \
                        = curr_xy_depth[valid_indx_init, -1].detach().cpu().numpy()
                
                return depth_images.copy()
            
            
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

    return E_lap

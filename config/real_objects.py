"""Per-object settings for the real thermal-video experiments.

Every value here was taken from the per-object notebooks that produced the
results in the paper, so that a single notebook
(``notebooks/real.ipynb``) reproduces each object instead of
keeping one near-duplicate notebook per object.

``get_config(obj, baseline=True)`` returns the comparison setting: shape from
heat conduction alone. The shading stage is switched off and the
hyper-parameters that were tuned for that setting are applied.

Recordings are referred to by the short names used for distribution
(``DATA/obj_study_raw/<name>``); ``SOURCE_RECORDINGS`` maps each one back to
the capture it came from.
"""

from types import SimpleNamespace


# Distributed file name -> name of the original capture.
SOURCE_RECORDINGS = {
    'bear-0.npz': 'bear-v34-0-left-rad1-tlin0-2023-10-16_13-14-39.npz',
    'bear-1.npz': 'bear-v34-1-right-rad1-tlin0-2023-10-16_13-07-11.npz',
    'bunny.npz': 'bunny-black-v81-incan-0-left-rad1-tlin0-2024-01-28_17-19-43.npz',
    'droplet.npz': 'thermal_video_20251219_190720_30fps_droplet-stronglight.npz',
    'heart.npz': 'thermal_video_20251231_153423_10fps_heart-stronglight.npz',
    'paintedbear.npz': 'thermal_video_20260108_170144_10fps_paintedbear-stronglight.npz',
    'soda_can-0.npz': 'soda_can-v43-0-left-incan-rad1-tlin0-2023-10-21_18-47-43.npz',
    'soda_can-1.npz': 'soda_can-v43-1-right-incan-rad1-tlin0-2023-10-21_18-50-52.npz',
    'sphere-0.npz': 'sphere-v32-0-left-rad1-tlin0-2023-10-13_13-15-54.npz',
    'sphere-1.npz': 'sphere-v32-1-right-rad1-tlin0-2023-10-13_13-18-47.npz',
}

# Identical for every object.
COMMON_PARAMS = {
    'ADD_CONVECTION': 1,
    'ADD_RADIATION': 1,
    'NUM_BG_FRAMES': 1,
    'SURFACE_THICKNESS': 3e-06,
    'add_origverts_to_initmesh': 1,
    'init_mesh_from_mask': 1,
    'light_directions': [[50.0, 0.0, 50.0]],
    'limit_zvals': 0,
    'max_img_size': (192, 240),
    'num_files': 1,
    'num_views': 1,
    'obj_dist': 0.4445,
    'obj_material': 'polystyrene',
    'obj_name': 'bear',
    'optim_convection_coeff': 1,
    'optim_heat_capacity': 1,
    'optim_heat_source': 1,
    'optim_heat_vals': 1,
    'optim_heat_vals_all_vertices': 1,
    'optim_laplacian': 1,
    'optim_scale_factor': 0,
    'optim_shape': 1,
    'optim_thermal_cond': 0,
    'optim_thermal_diff': 0,
    'optimize_MinvL': 1,
    'split_final_stage_opti': 1,
    'start_img_size': (192, 240),
    'unproject_and_update': 1,
    'zfar': 0.4845,
    'znear': 0.4045,
}

# Identical for every object.
COMMON_SHAPE_OPTI = {
    'activate_laplacian_reg_after_step': 500,
    'activate_normal_reg_after_step': 0,
    'convection_coeff_opti': 0,
    'heat_capacity_opti': 0,
    'heat_source_opti': 0,
    'heat_value_opti': 0,
    'lambda_symmetric_verts': 10000.0,
    'laplacian_reg': 0.0,
    'learning_rate': 0.01,
    'lr_gamma': 0.5,
    'lr_step_size': 7000,
    'mesh_edge_reg': 0.0,
    'num_steps_for_symmetric_verts': 1000,
    'reduce_diffusivity': 0,
}

# Per-object values: only the keys that differ from the two dicts above.
OBJECTS = {
    'bear': {
        "videos": [
            'bear-0.npz',
            'bear-1.npz',
        ],
        "params": {
            'NUM_FRAME_DELTA': 10,
            'obj_name': 'bear',
            'symmetric_about_axis': 1,
            'use_image_symmetry_constraint': 1,
        },
        "shape_opti": {
            'lambda_': 16,
            'lambda_normal': 0.0005,
            'lambda_pde': 2.5,
            'normal_reg': 0.1,
            'num_iterations': 500,
        },
        "num_ST_loops": 2,
        "baseline": {
            "videos": [
                'bear-0.npz',
            ],
            "shape_opti": {
                # lambda_normal is unused here (the shading stage never runs).
                'lambda_normal': 0.0,
                'lambda_pde': 2.0,
                'normal_reg': 0.05,
            },
            "num_ST_loops": 1,
        },
    },
    'bunny': {
        "videos": [
            'bunny.npz',
        ],
        "params": {
            'NUM_FRAME_DELTA': 10,
            'obj_name': 'bunny',
            'symmetric_about_axis': -1,
            'use_image_symmetry_constraint': 0,
        },
        "shape_opti": {
            'lambda_': 16,
            'lambda_normal': 0.0013,
            'lambda_pde': 1.0,
            'normal_reg': 0.09,
            'num_iterations': 500,
        },
        "num_ST_loops": 2,
        # No comparison run was recorded for this object.
    },
    'droplet': {
        "videos": [
            'droplet.npz',
        ],
        "params": {
            'NUM_FRAME_DELTA': 10,
            'obj_name': 'droplet',
            'symmetric_about_axis': -1,
            'use_image_symmetry_constraint': 1,
        },
        "shape_opti": {
            'lambda_': 16,
            'lambda_normal': 0.001,
            'lambda_pde': 10.0,
            'normal_reg': 0.09,
            'num_iterations': 500,
        },
        "num_ST_loops": 1,
        # No comparison run was recorded for this object.
    },
    'heart': {
        "videos": [
            'heart.npz',
        ],
        "params": {
            'NUM_FRAME_DELTA': 1,
            'obj_name': 'heart',
            'symmetric_about_axis': 0,
            'use_image_symmetry_constraint': 0,
        },
        "shape_opti": {
            'lambda_': 16,
            'lambda_normal': 0.01,
            'lambda_pde': 10.0,
            'normal_reg': 0.1,
            'num_iterations': 500,
        },
        "num_ST_loops": 1,
        "baseline": {
            "videos": [
                'heart.npz',
            ],
            "params": {
                'NUM_FRAME_DELTA': 10,
                'symmetric_about_axis': 1,
            },
            "shape_opti": {
                # lambda_normal is unused here (the shading stage never runs).
                'lambda_normal': 0.0,
            },
            "num_ST_loops": 1,
        },
    },
    'paintedbear': {
        "videos": [
            'paintedbear.npz',
        ],
        "params": {
            'NUM_FRAME_DELTA': 1,
            'obj_name': 'paintedbear',
            'symmetric_about_axis': 1,
            'use_image_symmetry_constraint': 1,
        },
        "shape_opti": {
            'lambda_': 16,
            'lambda_normal': 0.0005,
            'lambda_pde': 150.0,
            'normal_reg': 0.11,
            'num_iterations': 500,
        },
        "num_ST_loops": 2,
        "baseline": {
            "videos": [
                'paintedbear.npz',
            ],
            "params": {
                'NUM_FRAME_DELTA': 10,
            },
            "shape_opti": {
                # lambda_normal is unused here (the shading stage never runs).
                'lambda_normal': 0.0,
                'lambda_pde': 40.0,
                'normal_reg': 0.1,
            },
            "num_ST_loops": 1,
        },
    },
    'soda_can': {
        "videos": [
            'soda_can-0.npz',
            'soda_can-1.npz',
        ],
        "params": {
            'NUM_FRAME_DELTA': 10,
            'obj_name': 'soda_can',
            'symmetric_about_axis': -1,
            'use_image_symmetry_constraint': 1,
        },
        "shape_opti": {
            'lambda_': 1,
            'lambda_normal': 0.003,
            'lambda_pde': 0.1,
            'normal_reg': 3.0,
            'num_iterations': 3000,
        },
        "num_ST_loops": 1,
        "baseline": {
            "videos": [
                'soda_can-0.npz',
                'soda_can-1.npz',
            ],
            "params": {
                'symmetric_about_axis': 1,
            },
            "shape_opti": {
                # lambda_normal is unused here (the shading stage never runs).
                'lambda_': 16,
                'lambda_normal': 0.0,
                'lambda_pde': 1.0,
                'normal_reg': 0.05,
                'num_iterations': 500,
            },
            "num_ST_loops": 1,
        },
    },
    'sphere': {
        "videos": [
            'sphere-0.npz',
            'sphere-1.npz',
        ],
        "params": {
            'NUM_FRAME_DELTA': 10,
            'obj_name': 'sphere',
            'symmetric_about_axis': 1,
            'use_image_symmetry_constraint': 1,
        },
        "shape_opti": {
            'lambda_': 16,
            'lambda_normal': 0.002,
            'lambda_pde': 0.05,
            'normal_reg': 0.1,
            'num_iterations': 25000,
        },
        "num_ST_loops": 1,
        "baseline": {
            "videos": [
                'sphere-0.npz',
                'sphere-1.npz',
            ],
            "shape_opti": {
                # lambda_normal is unused here (the shading stage never runs).
                'lambda_normal': 0.0,
                'lambda_pde': 10.0,
                'num_iterations': 500,
            },
            "num_ST_loops": 1,
        },
    },
}


def available_objects(baseline=False):
    """Object names ``get_config`` accepts."""
    if baseline:
        return sorted(k for k, v in OBJECTS.items() if "baseline" in v)
    return sorted(OBJECTS)


def get_config(obj_name, baseline=False):
    """Settings for one object as a namespace.

    Attributes: ``name``, ``params``, ``shape_opti``, ``videos``,
    ``num_ST_loops``, ``use_shading`` and ``exp_name``.
    """
    if obj_name not in OBJECTS:
        raise KeyError(
            f"unknown object {obj_name!r}; choose from {available_objects()}")
    entry = OBJECTS[obj_name]

    if baseline:
        if "baseline" not in entry:
            raise KeyError(
                f"no comparison setting recorded for {obj_name!r}; "
                f"available: {available_objects(baseline=True)}")
        override = entry["baseline"]
    else:
        override = {}

    params = {**COMMON_PARAMS, **entry["params"], **override.get("params", {})}
    shape_opti = {**COMMON_SHAPE_OPTI, **entry["shape_opti"],
                  **override.get("shape_opti", {})}

    return SimpleNamespace(
        name=obj_name,
        params=params,
        shape_opti=shape_opti,
        videos=list(override.get("videos", entry["videos"])),
        num_ST_loops=override.get("num_ST_loops", entry["num_ST_loops"]),
        use_shading=not baseline,
        exp_name=f"{obj_name}-baseline" if baseline else obj_name,
    )

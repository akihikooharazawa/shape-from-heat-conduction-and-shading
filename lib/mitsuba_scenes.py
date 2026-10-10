import mitsuba as mi


def _orthographic_sensor(up_y, sample_count, width=2048, height=2048):
    return {
        'type': 'orthographic',
        'to_world': mi.ScalarTransform4f().look_at(
            origin=mi.ScalarPoint3f(0.0, 0.0, 102.1),
            target=mi.ScalarPoint3f(0.0, 0.0, 0.0),
            up=mi.ScalarPoint3f(0.0, up_y, 0.0)
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': sample_count,
        },
        'film': {
            'type': 'hdrfilm',
            'width': width,
            'height': height,
            'pixel_format': 'luminance',
        },
    }


def interref_scene_dict(ps, qs,
                        reflectance=None,
                        sample_count=128,
                        mesh_file='temp_mesh.ply'):
    """Lambertian diffuse BRDF with interreflection via path tracing."""
    if reflectance is None:
        reflectance = [0.3, 0.3, 0.3]
    return {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 128},
        'sensor': _orthographic_sensor(up_y=1.0, sample_count=sample_count),
        'shape': {
            'type': 'ply',
            'filename': mesh_file,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': reflectance},
            },
        },
        'emitter_directional': {
            'type': 'directional',
            'direction': [ps, qs, -1.0],
            'irradiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]},
        },
    }


def brdf_scene_dict(ps, qs,
                    alpha=0.4,
                    diffuse_reflectance=None,
                    irradiance=10.0,
                    sample_count=512,
                    mesh_file='temp_mesh.ply'):
    """Rough-plastic BRDF. alpha controls surface roughness (0.1–0.4)."""
    if diffuse_reflectance is None:
        diffuse_reflectance = [0.8, 0.8, 0.8]
    return {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 128},
        'sensor': _orthographic_sensor(up_y=-1.0, sample_count=sample_count),
        'shape': {
            'type': 'ply',
            'filename': mesh_file,
            'bsdf': {
                'type': 'roughplastic',
                'diffuse_reflectance': {'type': 'rgb', 'value': diffuse_reflectance},
                'alpha': alpha,
            },
        },
        'emitter_directional': {
            'type': 'directional',
            'direction': [-ps, -qs, -1.0],
            'irradiance': {'type': 'rgb', 'value': [irradiance] * 3},
        },
    }


def albedo_scene_dict(ps, qs,
                      checker_scale_u=5.0,
                      checker_scale_v=5.0,
                      c0=None,
                      c1=None,
                      sample_count=128,
                      mesh_file='temp_mesh_uv.ply'):
    """Checkerboard albedo pattern rendered with directional light."""
    if c0 is None:
        c0 = [0.20, 0.20, 0.20]
    if c1 is None:
        c1 = [0.40, 0.40, 0.40]
    return {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 128},
        'sensor': _orthographic_sensor(up_y=1.0, sample_count=sample_count),
        'shape': {
            'type': 'ply',
            'filename': mesh_file,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'checkerboard',
                    'color0': {'type': 'rgb', 'value': c0},
                    'color1': {'type': 'rgb', 'value': c1},
                    'to_uv': mi.ScalarTransform4f().scale(
                        mi.ScalarVector3f(checker_scale_u, checker_scale_v, 1.0)
                    ).translate(
                        mi.ScalarVector3f(0.0, 0.0, 0.0)
                    ),
                },
            },
        },
        'emitter_directional': {
            'type': 'directional',
            'direction': mi.ScalarVector3f(ps, qs, -1.0),
            'irradiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]},
        },
    }

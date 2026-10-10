import torch


def tri_cotmatrix_entries(verts, faces):
    """
    Compute the cotangent laplacian

    Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """


    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0
    return cot


def laplacian_cot(verts, faces):
    
    # V = sum(V_n), F = sum(F_n)
    V, F = verts.shape[0], faces.shape[0]
    cot = tri_cotmatrix_entries(verts, faces)

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)

    # L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))
    L = torch.sparse_coo_tensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # Add the diagonal indices
    vals = torch.sparse.sum(L, dim=0).to_dense()
    indices = torch.arange(V, device='cuda')
    idx = torch.stack([indices, indices], dim=0)
    # L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
    L = torch.sparse_coo_tensor(idx, vals, (V, V)) - L
    return L

def laplacian_cot2(verts, faces):
    return -laplacian_cot(verts, faces)/2.0

def laplacian_from_cotvals(cot, faces, num_verts):

    V, F = num_verts, faces.shape[0]
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    # L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))
    L = torch.sparse_coo_tensor(idx, cot.view(-1)/2.0, (V, V))
    L = L.coalesce()

    # Make it symmetric; this means we are also setting
    L += L.t()

    # Add the diagonal indices
    vals = torch.sparse.sum(L, dim=0).to_dense()
    indices = torch.arange(V, device='cuda')
    idx = torch.stack([indices, indices], dim=0)
    # L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
    L = torch.sparse_coo_tensor(idx, vals, (V, V)) - L
    return L

def laplacian_from_cotvals_dense(cot, faces, num_verts, device='cuda'):

    V, F = num_verts, faces.shape[0]
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.zeros([V, V], dtype=cot.dtype, device=device)
    L.index_put_((idx[0], idx[1]), cot.view(-1)/2.0, accumulate=True)
    L = L + L.t()
    L = torch.diag(L.sum(dim=0)) - L
    return L

def laplacian_uniform(verts, faces):
    """
    Compute the uniform laplacian

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device='cuda', dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()

def compute_matrix(verts, faces, lambda_, alpha=None, L=None, cotan=False):
    """
    Build the parameterization matrix.

    If alpha is defined, then we compute it as (1-alpha)*I + alpha*L otherwise
    as I + lambda*L as in the paper. The first definition can be slightly more
    convenient as it the scale of the resulting matrix doesn't change much
    depending on alpha.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    lambda_ : float
        Hyperparameter lambda of our method, used to compute the
        parameterization matrix as (I + lambda_ * L)
    alpha : float in [0, 1[
        Alternative hyperparameter, used to compute the parameterization matrix
        as ((1-alpha) * I + alpha * L)
    cotan : bool
        Compute the cotangent laplacian. Otherwise, compute the combinatorial one
    """
    if not (L is not None):
        if cotan:
            L = laplacian_cot(verts, faces)
        else:
            L = laplacian_uniform(verts, faces)

    idx = torch.arange(verts.shape[0], dtype=torch.long, device='cuda')
    eye = torch.sparse_coo_tensor(torch.stack((idx, idx), dim=0), torch.ones(verts.shape[0], dtype=torch.float, device='cuda'), (verts.shape[0], verts.shape[0]))
    if alpha is None:
        M = torch.add(eye, lambda_*L) # M = I + lambda_ * L
    else:
        if alpha < 0.0 or alpha >= 1.0:
            raise ValueError(f"Invalid value for alpha: {alpha} : it should take values between 0 (included) and 1 (excluded)")
        M = torch.add((1-alpha)*eye, alpha*L) # M = (1-alpha) * I + alpha * L
    return M.coalesce()


# For every vertex compute sum area of all triangles incident on that vertex and divide by 3
def compute_faceArea(v, f):
    v1 = v[f[:, 0], :]
    v2 = v[f[:, 1], :]
    v3 = v[f[:, 2], :]
    faceArea = torch.norm(torch.cross(v2-v1, v3-v1, dim=1), dim=-1)/2
    return faceArea


def massmatrix_fast(v, f):
    faceArea = compute_faceArea(v, f)
    V, F = v.shape[0], f.shape[0]
    idx = torch.stack([f, f], dim=0).reshape(2, -1)
    values = torch.stack([faceArea, faceArea, faceArea], dim=-1).reshape(-1)
    M = torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()/3.0
    return M
    
def massmatrix_from_faceArea(faceArea, f, num_verts):
    V, F = num_verts, f.shape[0]
    idx = torch.stack([f, f], dim=0).reshape(2, -1)
    values = torch.stack([faceArea, faceArea, faceArea], dim=-1).reshape(-1)
    M = torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()/3.0
    return M

def massmatrix_from_faceArea_dense(faceArea, f, num_verts):
    V, F = num_verts, f.shape[0]
    idx = torch.stack([f, f], dim=0).reshape(2, -1)
    values = torch.stack([faceArea, faceArea, faceArea], dim=-1).reshape(-1)
    # M = torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()/3.0
    M = torch.zeros([V, V], dtype=faceArea.dtype, device='cuda')
    M.index_put_((idx[0], idx[1]), values, accumulate=True)
    M = M/3.0
    return M

def massmatrix_fromvals(massvalues, num_verts, inverse=False):
    V = num_verts
    idx = torch.stack([torch.arange(V), torch.arange(V)], dim=0).reshape(2, -1)
    if not inverse:
        M = torch.sparse_coo_tensor(idx, massvalues, (V, V))
    else:
        M = torch.sparse_coo_tensor(idx, 1.0/massvalues, (V, V))
    return M
import torch

def tet_edge_lengths(verts, faces):
    """
    Compute the edge lengths of a tet mesh.

    Parameters
    ----------
    verts : torch.Tensor
        (V, 3) array of vertex positions
    faces : torch.Tensor
        (F, 4) array of tet faces.

    Returns
    -------
    torch.Tensor
        (V, 6) array of edge lengths.
    """
    face_verts = verts[faces]
    v0, v1, v2, v3 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2], face_verts[:, 3]

    # Edge lengths of each tet, of shape (F, 6)
    # where length corresponds to [3 0],[3 1],[3 2],[1 2],[2 0],[0 1]
    A = (v3 - v0).norm(dim=1)
    B = (v3 - v1).norm(dim=1)
    C = (v3 - v2).norm(dim=1)
    D = (v1 - v2).norm(dim=1)
    E = (v2 - v0).norm(dim=1)
    F = (v0 - v1).norm(dim=1)
    return torch.stack([A, B, C, D, E, F], dim=1)

def compute_doublearea(lengths, nan_replacement=0.0):

    lengths_sorted, _ = torch.sort(lengths, dim=1, descending=True)
    """ (l(i,0)+(l(i,1)+l(i,2)))*
        (l(i,2)-(l(i,0)-l(i,1)))*
        (l(i,2)+(l(i,0)-l(i,1)))*
        (l(i,0)+(l(i,1)-l(i,2))); """
    scalar = (lengths_sorted[:, 0] + (lengths_sorted[:, 1] + lengths_sorted[:, 2])) * \
        (lengths_sorted[:, 2] - (lengths_sorted[:, 0] - lengths_sorted[:, 1])) * \
        (lengths_sorted[:, 2] + (lengths_sorted[:, 0] - lengths_sorted[:, 1])) * \
        (lengths_sorted[:, 0] + (lengths_sorted[:, 1] - lengths_sorted[:, 2]))
    dblArea = 2.0*0.25*torch.sqrt(scalar)
    
    return dblArea


def tet_face_areas(edge_lengths):

    """    // (unsigned) face Areas (opposite vertices: 1 2 3 4)
    Matrix<typename DerivedA::Scalar,Dynamic,1> 
    A0(m,1), A1(m,1), A2(m,1), A3(m,1);
    Matrix<typename DerivedA::Scalar,Dynamic,3> 
    L0(m,3), L1(m,3), L2(m,3), L3(m,3);
    L0<<L.col(1),L.col(2),L.col(3);
    L1<<L.col(0),L.col(2),L.col(4);
    L2<<L.col(0),L.col(1),L.col(5);
    L3<<L.col(3),L.col(4),L.col(5);
    doublearea(L0,doublearea_nan_replacement,A0);
    doublearea(L1,doublearea_nan_replacement,A1);
    doublearea(L2,doublearea_nan_replacement,A2);
    doublearea(L3,doublearea_nan_replacement,A3);
    A.resize(m,4);
    A.col(0) = 0.5*A0;
    A.col(1) = 0.5*A1;
    A.col(2) = 0.5*A2;
    A.col(3) = 0.5*A3;
    """

    area1 = 0.5*compute_doublearea(edge_lengths[:, [1, 2, 3]], nan_replacement=0.0)
    area2 = 0.5*compute_doublearea(edge_lengths[:, [0, 2, 4]], nan_replacement=0.0)
    area3 = 0.5*compute_doublearea(edge_lengths[:, [0, 1, 5]], nan_replacement=0.0)
    area4 = 0.5*compute_doublearea(edge_lengths[:, [3, 4, 5]], nan_replacement=0.0)

    return torch.stack([area1, area2, area3, area4], dim=1)


def tet_dihedral_angles_from_lengths(edge_lengths, face_areas):
    """
    Compute the dihedral angles of a tet mesh from its edge lengths.

    Parameters
    ----------
    edge_lengths : torch.Tensor
        (F, 6) array of edge lengths.

    Returns
    -------
    torch.Tensor
        (F, 6) array of dihedral angles.
    """
    
    H_sqr = torch.zeros((edge_lengths.shape[0], 6))
    H_sqr[:, 0] += (1./16.) * (4. * edge_lengths[:, 3]**2 * edge_lengths[:, 0]**2 - 
                            ((edge_lengths[:, 1]**2 + edge_lengths[:, 4]**2) - (edge_lengths[:, 2]**2 + edge_lengths[:, 5]**2))**2)
    H_sqr[:, 1] += (1./16.) * (4. * edge_lengths[:, 4]**2 * edge_lengths[:, 1]**2 - 
                            ((edge_lengths[:, 2]**2 + edge_lengths[:, 5]**2) - (edge_lengths[:, 3]**2 + edge_lengths[:, 0]**2))**2)
    H_sqr[:, 2] += (1./16.) * (4. * edge_lengths[:, 5]**2 * edge_lengths[:, 2]**2 - 
                            ((edge_lengths[:, 3]**2 + edge_lengths[:, 0]**2) - (edge_lengths[:, 4]**2 + edge_lengths[:, 1]**2))**2)
    H_sqr[:, 3] += (1./16.) * (4. * edge_lengths[:, 0]**2 * edge_lengths[:, 3]**2 - 
                            ((edge_lengths[:, 4]**2 + edge_lengths[:, 1]**2) - (edge_lengths[:, 5]**2 + edge_lengths[:, 2]**2))**2)
    H_sqr[:, 4] += (1./16.) * (4. * edge_lengths[:, 1]**2 * edge_lengths[:, 4]**2 - 
                            ((edge_lengths[:, 5]**2 + edge_lengths[:, 2]**2) - (edge_lengths[:, 0]**2 + edge_lengths[:, 3]**2))**2)
    H_sqr[:, 5] += (1./16.) * (4. * edge_lengths[:, 2]**2 * edge_lengths[:, 5]**2 - 
                            ((edge_lengths[:, 0]**2 + edge_lengths[:, 3]**2) - (edge_lengths[:, 1]**2 + edge_lengths[:, 4]**2))**2)

    cos_theta = torch.zeros((edge_lengths.shape[0], 6))
    cos_theta[:, 0] += (H_sqr[:, 0] - face_areas[:, 1]**2 - face_areas[:, 2]**2) / (-2.*face_areas[:, 1] * face_areas[:, 2])
    cos_theta[:, 1] += (H_sqr[:, 1] - face_areas[:, 2]**2 - face_areas[:, 0]**2) / (-2.*face_areas[:, 2] * face_areas[:, 0])
    cos_theta[:, 2] += (H_sqr[:, 2] - face_areas[:, 0]**2 - face_areas[:, 1]**2) / (-2.*face_areas[:, 0] * face_areas[:, 1])
    cos_theta[:, 3] += (H_sqr[:, 3] - face_areas[:, 3]**2 - face_areas[:, 0]**2) / (-2.*face_areas[:, 3] * face_areas[:, 0])
    cos_theta[:, 4] += (H_sqr[:, 4] - face_areas[:, 3]**2 - face_areas[:, 1]**2) / (-2.*face_areas[:, 3] * face_areas[:, 1])
    cos_theta[:, 5] += (H_sqr[:, 5] - face_areas[:, 3]**2 - face_areas[:, 2]**2) / (-2.*face_areas[:, 3] * face_areas[:, 2])

    theta = torch.arccos(cos_theta)

    return theta, cos_theta

def tet_volume(edge_lengths):
    """
    Compute volume of the tets from edge lengths.

    Parameters
    ----------
    edge_lengths : torch.Tensor
        (F, 6) array of edge lengths.

    Returns
    -------
    torch.Tensor
        (F, 1) array of tet volumes.
    """

    u = edge_lengths[:, 0]
    v = edge_lengths[:, 1]
    w = edge_lengths[:, 2]
    U = edge_lengths[:, 3]
    V = edge_lengths[:, 4]
    W = edge_lengths[:, 5]
    X = (w - U + v) * (U + v + w)
    x = (U - v + w) * (v - w + U)
    Y = (u - V + w) * (V + w + u)
    y = (V - w + u) * (w - u + V)
    Z = (v - W + u) * (W + u + v)
    z = (W - u + v) * (u - v + W)
    a = torch.sqrt(x * Y * Z)
    b = torch.sqrt(y * Z * X)
    c = torch.sqrt(z * X * Y)
    d = torch.sqrt(x * y * z)
    vol = torch.sqrt((-a + b + c + d) * (a - b + c + d) * (a + b - c + d) * (a + b + c - d)) / (192. * u * v * w)
    
    return vol

def tet_cotmatrix_entries(verts, faces):
    """
    Compute the cotangent matrix entries for a mesh.
    
    Parameters
    ----------
    verts : torch.Tensor
        (V, 3) array of vertex positions.
    faces : torch.Tensor
        (F, 4) array of faces.
    
    Returns
    -------
    torch.Tensor
        (F, 6) array of cotangent matrix entries.
    """

    # Compute edge lengths
    edge_lengths = tet_edge_lengths(verts, faces)
    # print(edge_lengths, edge_lengths.min(), edge_lengths.max())

    # Compute face areas
    face_areas = tet_face_areas(edge_lengths)

    # Compute dihedral angles
    dihedral_angles, cos_theta = tet_dihedral_angles_from_lengths(edge_lengths, face_areas)

    # Compute volume
    vol = tet_volume(edge_lengths)

    # print(torch.any(torch.isnan(dihedral_angles)), torch.any(torch.isnan(cos_theta)), torch.any(torch.isnan(vol)), torch.any(torch.isnan(face_areas)), torch.any(torch.isnan(edge_lengths)))

    # print(edge_lengths.shape, face_areas.shape, dihedral_angles.shape, vol.shape)
    sin_theta = torch.zeros((edge_lengths.shape[0], 6))
    sin_theta[:, 0] += vol / ((2. / (3. * edge_lengths[:, 0])) * face_areas[:, 1] * face_areas[:, 2])
    sin_theta[:, 1] += vol / ((2. / (3. * edge_lengths[:, 1])) * face_areas[:, 2] * face_areas[:, 0])
    sin_theta[:, 2] += vol / ((2. / (3. * edge_lengths[:, 2])) * face_areas[:, 0] * face_areas[:, 1])
    sin_theta[:, 3] += vol / ((2. / (3. * edge_lengths[:, 3])) * face_areas[:, 3] * face_areas[:, 0])
    sin_theta[:, 4] += vol / ((2. / (3. * edge_lengths[:, 4])) * face_areas[:, 3] * face_areas[:, 1])
    sin_theta[:, 5] += vol / ((2. / (3. * edge_lengths[:, 5])) * face_areas[:, 3] * face_areas[:, 2])
    
    cot_entries = (1. / 6.) * edge_lengths * cos_theta / sin_theta
    
    return cot_entries

def tet_laplacian(verts, faces):

    V = verts.shape[0]
    F = faces.shape[0]
    cot_entries = tet_cotmatrix_entries(verts, faces)

    ii = faces[:, [1,2,0,3,3,3]]
    jj = faces[:, [2,0,1,0,1,2]]
    idx = torch.stack([ii, jj], dim=0).view(2, F*6)

    L = torch.sparse.FloatTensor(idx, cot_entries.view(-1), (V, V))

    # Make it symmetric;
    L += L.t()

    # Add the diagonal indices
    vals = torch.sparse.sum(L, dim=1).to_dense()
    indices = torch.arange(V, device='cuda')
    idx = torch.stack([indices, indices], dim=0)
    L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L

    return L

def tet_massmatrix(verts, faces):

    """ Matrix<typename DerivedF::Scalar,Dynamic,1> MI;
    Matrix<typename DerivedF::Scalar,Dynamic,1> MJ;
    Matrix<Scalar,Dynamic,1> MV;
    assert(V.cols() == 3);
    assert(eff_type == MASSMATRIX_TYPE_BARYCENTRIC);
    MI.resize(m*4,1); MJ.resize(m*4,1); MV.resize(m*4,1);
    MI.block(0*m,0,m,1) = F.col(0);
    MI.block(1*m,0,m,1) = F.col(1);
    MI.block(2*m,0,m,1) = F.col(2);
    MI.block(3*m,0,m,1) = F.col(3);
    MJ = MI;
    // loop over tets
    for(int i = 0;i<m;i++)
    {
      // http://en.wikipedia.org/wiki/Tetrahedron#Volume
      Matrix<Scalar,3,1> v0m3,v1m3,v2m3;
      v0m3.head(V.cols()) = V.row(F(i,0)) - V.row(F(i,3));
      v1m3.head(V.cols()) = V.row(F(i,1)) - V.row(F(i,3));
      v2m3.head(V.cols()) = V.row(F(i,2)) - V.row(F(i,3));
      Scalar v = fabs(v0m3.dot(v1m3.cross(v2m3)))/6.0;
      MV(i+0*m) = v/4.0;
      MV(i+1*m) = v/4.0;
      MV(i+2*m) = v/4.0;
      MV(i+3*m) = v/4.0;
    }
    sparse(MI,MJ,MV,n,n,M); """
    
    v0m3 = verts[faces[:, 3]] - verts[faces[:, 0]]
    v1m3 = verts[faces[:, 1]] - verts[faces[:, 3]]
    v2m3 = verts[faces[:, 2]] - verts[faces[:, 3]]
    v = torch.abs(torch.einsum('ij,ij->i', v0m3, torch.cross(v1m3, v2m3))) / (6.)

    ii = faces[:, [0, 1, 2, 3]]

    idx = torch.stack([ii, ii], dim=0).view(2, -1)
    vals = torch.stack([v/4., v/4., v/4., v/4.], dim=0).view(-1)
    # print(idx.shape, vals.shape)
    # print(v.shape[0])
    # print(vals[torch.where(ii.reshape(-1)==0)[0]], vals[torch.where(ii.reshape(-1)==0)[0]].sum())

    M = torch.sparse_coo_tensor(idx, vals, (verts.shape[0], verts.shape[0]))
    return M.coalesce()

def tet_compute_matrix(verts, faces, lambda_, alpha=None, L=None, cotan=True):
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
        L = tet_laplacian(verts, faces)
        
    idx = torch.arange(verts.shape[0], dtype=torch.long, device='cuda')
    eye = torch.sparse_coo_tensor(torch.stack((idx, idx), dim=0), torch.ones(verts.shape[0], dtype=torch.float, device='cuda'), (verts.shape[0], verts.shape[0]))
    if alpha is None:
        M = torch.add(eye, lambda_*L) # M = I + lambda_ * L
    else:
        if alpha < 0.0 or alpha >= 1.0:
            raise ValueError(f"Invalid value for alpha: {alpha} : it should take values between 0 (included) and 1 (excluded)")
        M = torch.add((1-alpha)*eye, alpha*L) # M = (1-alpha) * I + alpha * L
    return M.coalesce()
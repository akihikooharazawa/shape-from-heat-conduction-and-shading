import numpy as np
from lib.utils_sim import *

# -------------- Objective function for optimization -------------- 
def obj_function_t(x, idx_i, idx_a, idx_b, idx_d, idx_e, idx_f,
                   i_v, lap_gt, width, height, eta_gt, s_hat, w_lap=None):
    
    t_i, t_a, t_b, t_f = x

    # Eta
    eta_i = eta_gt[idx_i]
    eta_a = eta_gt[idx_a]
    eta_b = eta_gt[idx_b]
    eta_f = eta_gt[idx_f]

    # t -> p,q
    p_i, q_i = normal_components(t_i, eta_i, s_hat)
    p_a, q_a = normal_components(t_a, eta_a, s_hat)
    p_b, q_b = normal_components(t_b, eta_b, s_hat)
    p_f, q_f = normal_components(t_f, eta_f, s_hat)
    
    p_d = p_i + q_b - q_i
    p_e = p_f + q_a - q_f

    # Laplacian error
    pq_modified = np.zeros((height*width, 2), dtype=np.float64)
    pq_modified[idx_i, 0] = p_i
    pq_modified[idx_i, 1] = q_i

    pq_modified[idx_a, 0] = p_a
    pq_modified[idx_a, 1] = q_a

    pq_modified[idx_b, 0] = p_b
    pq_modified[idx_b, 1] = q_b

    pq_modified[idx_d, 0] = p_d

    pq_modified[idx_e, 0] = p_e

    pq_modified[idx_f, 0] = p_f
    pq_modified[idx_f, 1] = q_f
    
    lap_hat = cotan_laplacians_from_pq_np(pq_modified, width, height)
    diff_lap = lap_gt[i_v] - lap_hat[i_v]
    E_lap = np.mean(diff_lap**2)

    return w_lap * E_lap

# -------------- Create propagation order -------------- 
def prepare_neighbor_dict(ind_v, ind_a, ind_b, ind_c, ind_d, ind_e, ind_f, width, height):
    
    N = (width - 2)*(height - 2)
    
    global2local = {}
    for local_id in range(N):
        g_id = ind_v[local_id]
        global2local[g_id] = local_id

    neighbor_dict = {}
    
    for local_id in range(N):
        a_g = ind_a[local_id]
        b_g = ind_b[local_id]
        c_g = ind_c[local_id]
        d_g = ind_d[local_id]
        e_g = ind_e[local_id]
        f_g = ind_f[local_id]

        a_in = global2local.get(a_g, -1)
        b_in = global2local.get(b_g, -1)
        c_in = global2local.get(c_g, -1)
        d_in = global2local.get(d_g, -1)
        e_in = global2local.get(e_g, -1)
        f_in = global2local.get(f_g, -1)
        
        neighbors = []
        if a_in >= 0: neighbors.append((a_in, 'up'))
        if b_in >= 0: neighbors.append((b_in, 'right'))
        if c_in >= 0: neighbors.append((c_in, 'diag-right'))
        if d_in >= 0: neighbors.append((d_in, 'down'))
        if e_in >= 0: neighbors.append((e_in, 'left'))
        if f_in >= 0: neighbors.append((f_in, 'diag-left'))
        neighbor_dict[local_id] = neighbors

    return neighbor_dict

# -------------- Convert global index to row-column -------------- 
def to_rc(i, width):
    r = i // (width-2)
    c = i %  (width-2)
    return r, c

# -------------- Convert row-column to global index --------------
def to_id(r, c, width):
    return r * (width-2) + c

# -------------- Propagation --------------
def bfs_propagation(i_v_init, width, height):

    # Prepare
    N = (width - 2) * (height - 2)
    if not (0 <= i_v_init < N):
        return []
    visited = set()
    bfs_order_with_flag = []

    # Functions
    # (1) Scan Right
    def scan_first_right(r0, c0):
        for c_ in range(c0, (width-2)):
            lid = to_id(r0, c_, width)
            if lid not in visited:
                visited.add(lid)
                bfs_order_with_flag.append((lid, "right"))

    # (2) Scan Right (down)
    def scan_diagonal_down_right(r0, c0):
        current_r, current_c = r0, c0
        while True:
            if not (0 <= current_r < (height - 2)):
                break
            if not (0 <= current_c < (width - 2)):
                break
            visited_any = False
            col_start = current_c
            first_in_this_row = True
            for c in range(col_start, (width - 2)):
                lid = to_id(current_r, c, width)
                if lid not in visited:
                    if first_in_this_row:
                        visited.add(lid)
                        bfs_order_with_flag.append((lid, "right_diag"))
                        first_in_this_row = False
                    else:
                        visited.add(lid)
                        bfs_order_with_flag.append((lid, "right_diag"))
                    visited_any = True
            if not visited_any:
                break
            current_r += 1
            current_c += 1

    # (3) Scan Right (up)
    def scan_diagonal_down_right_left_side(r0, c0):
        current_r, current_c = r0, c0

        while True:
            if not (0 <= current_r < (height - 2)):
                break
            if not (0 <= current_c < (width - 2)):
                break

            visited_any = False
            first_in_this_row = True

            for col in range(current_c, (width - 2)):
                lid = to_id(current_r, col, width)
                if lid not in visited:
                    visited.add(lid)
                    if first_in_this_row:
                        bfs_order_with_flag.append((lid, "diagL_first"))
                        first_in_this_row = False
                    else:
                        bfs_order_with_flag.append((lid, "diagL_right"))
                    visited_any = True
            if not visited_any:
                break

            current_r -= 1
            current_c -= 1

    # (4) Scan Left up
    def scan_up_left_sequence(r0, c0):
        for row in range(r0 - 1, -1, -1):
            if row < 0:
                break
            visited_any = False
            first_in_line = True

            for col_ in reversed(range(0, c0 + 1)):
                lid = to_id(row, col_, width)
                if lid not in visited:
                    visited.add(lid)
                    if first_in_line:
                        bfs_order_with_flag.append((lid, "diagR_first"))
                        first_in_line = False
                    else:
                        bfs_order_with_flag.append((lid, "diagR_left"))
                    visited_any = True

            if not visited_any:
                break

    # (5) Scan Left down
    def scan_left(r0, c0):
        if c0 < 0:
            return
        start_col = c0 - 1 if (c0 - 1 >= 0) else c0

        current_r = r0
        current_c = start_col

        while True:
            if not (0 <= current_r < (height - 2)):
                break
            if not (0 <= current_c < (width - 2)):
                break

            visited_any = False
            first_in_line = True

            for col_ in range(current_c, -1, -1):
                lid = to_id(current_r, col_, width)
                if lid not in visited:
                    visited.add(lid)
                    if first_in_line:
                        bfs_order_with_flag.append((lid, "left"))
                        first_in_line = False
                    else:
                        bfs_order_with_flag.append((lid, "left"))
                    visited_any = True

            if not visited_any:
                break

            current_r += 1
            current_c += 1

            if not (0 <= current_r < (height - 2)):
                break
            if not (0 <= current_c < (width - 2)):
                break

    # (6) Scan Down
    def scan_right_to_left(r0, c0):
        current_r= r0
        current_c= c0
        while True:
            if not (0 <= current_r < (height-2)):
                break
            # clamp col
            if current_c>= (width-2):
                col_start= (width-2)-1
            else:
                col_start= current_c

            # scanning from col_start down to 0
            for col_ in range(col_start, -1, -1):
                if col_<0:
                    break
                lid= to_id(current_r, col_, width)
                if lid not in visited:
                    visited.add(lid)
                    bfs_order_with_flag.append((lid,"left"))

            current_r += 1

    # Main
    visited.add(i_v_init)
    bfs_order_with_flag.append((i_v_init, "init"))
    init_r, init_c = to_rc(i_v_init, width)
    
    # (1)
    scan_first_right(init_r, init_c + 1)
    # (2-1, 2-2)
    scan_diagonal_down_right(init_r + 1, init_c + 1)
    # (3-1, 3-2)
    if init_c - 1 >= 0:
       scan_diagonal_down_right_left_side(init_r - 1, init_c)
    # (4)
    if init_r > 0 and init_c > 0:
        scan_up_left_sequence(init_r, init_c)
    # (5)
    scan_left(init_r, init_c)
    # (6)
    scan_right_to_left(0, (width-2)-1)
    
    return bfs_order_with_flag

def build_candidate_values(current_val, mode, n_div=None):
    if mode == "fix":
        return [current_val]
    else:  # "free"
        return np.linspace(0.0, 2.0*math.pi, n_div, endpoint=False).tolist()
    
def decide_optim_mode(direction):
    if direction == "init":
        return {"t_i": "fix", "t_a": "fix", "t_b": "fix", "t_f": "fix"}
    # (1)
    elif direction == "right":
        return {"t_i": "free", "t_a": "free", "t_b": "free", "t_f": "fix"}
    # (2)
    elif direction == "right_diag":
        return {"t_i": "free", "t_a": "fix", "t_b": "free", "t_f": "fix"}
    # (3-1)
    elif direction == "diagL_first":
        return {"t_i": "fix", "t_a": "free", "t_b": "fix", "t_f": "free"}
    # (3-2)
    elif direction == "diagL_right":
        return {"t_i": "fix", "t_a": "free", "t_b": "fix", "t_f": "free"}
    # (4-1)
    elif direction == "diagR_first":
        return {"t_i": "free", "t_a": "fix", "t_b": "fix", "t_f": "free"}
    # (4-2)
    elif direction == "diagR_left":
        return {"t_i": "free", "t_a": "fix", "t_b": "fix", "t_f": "free"}
    # (5)
    elif direction == "left":
        return {"t_i": "free", "t_a": "fix", "t_b": "fix", "t_f": "fix"}
    else:
        return {"t_i": "free", "t_a": "free", "t_b": "free", "t_f": "free"}

# -------------- Available initial settings --------------
# Helper function
def get_t_value(lid, key, refined_dict, default=0.0):
    if lid in refined_dict:
        return refined_dict[lid][key]
    return default

# (1) Scan Right
def scanR(current_id, current_vals, refined_dict, height, width):
    r, c = to_rc(current_id, width)
    # Left neighbor
    if c - 1 >= 0:
        left_id = to_id(r, c - 1, width)
        current_vals["t_f"] = get_t_value(left_id, "t_a", refined_dict)
        current_vals["t_i"] = get_t_value(left_id, "t_b", refined_dict)
    else:
        current_vals["t_f"] = 0.0
        current_vals["t_i"] = 0.0
    return current_vals

# (2) Diag-Right-Down (Refer to left-up)
def scanRD(current_id, current_vals, refined_dict, height, width):
    r, c = to_rc(current_id, width)
    # Diag-Left neighbor
    if r > 0 and c > 0:
        diagL_id = to_id(r - 1, c - 1, width)
        current_vals["t_f"] = get_t_value(diagL_id, "t_i", refined_dict)
    else:
        current_vals["t_f"] = 0.0

    # Up neighbor
    if r > 0:
        up_id = to_id(r - 1, c, width)
        current_vals["t_a"] = get_t_value(up_id, "t_i", refined_dict)
    else:
        current_vals["t_a"] = 0.0
    return current_vals

# (3-1) Scan Right (Refer to right-down)
def scanRU(current_id, current_vals, refined_dict, height, width):
    r, c = to_rc(current_id, width)
    # Diag-Right neighbor
    if (r + 1) < (height - 2) and (c + 1) < (width - 2):
        diagR_id = to_id(r + 1, c + 1, width)
        current_vals["t_i"] = get_t_value(diagR_id, "t_f", refined_dict)
        current_vals["t_b"] = get_t_value(diagR_id, "t_a", refined_dict)
    else:
        current_vals["t_i"] = 0.0
        current_vals["t_b"] = 0.0
    return current_vals

# (3-2) Scan Right (Refer to right-down)
def scanRUS(current_id, current_vals, refined_dict, height, width):
    r, c = to_rc(current_id, width)
    # Diag-Right neighbor
    if (r + 1) < (height - 2) and (c + 1) < (width - 2):
        diagR_id = to_id(r + 1, c + 1, width)
        current_vals["t_i"] = get_t_value(diagR_id, "t_f", refined_dict)
        current_vals["t_b"] = get_t_value(diagR_id, "t_a", refined_dict)
    else:
        current_vals["t_i"] = 0.0
        current_vals["t_b"] = 0.0

    return current_vals

# (4) Scan Left Up (Refer to Right)
def scanLU(current_id, current_vals, refined_dict, height, width):
    r, c = to_rc(current_id, width)
    # Right neighbor
    if (c + 1) < (width - 2):
        right_id = to_id(r, c + 1, width)
        current_vals["t_b"] = get_t_value(right_id, "t_i", refined_dict)
        current_vals["t_a"] = get_t_value(right_id, "t_f", refined_dict)
    else:
        current_vals["t_b"] = 0.0
        current_vals["t_a"] = 0.0

    return current_vals

# (5) Scan Left Down (Refer to Right)
def scanLD(current_id, current_vals, refined_dict, height, width):
    r, c = to_rc(current_id, width)
    # Right neighbor
    if c + 1 < (width - 2):
        right_id = to_id(r, c + 1, width)
        current_vals["t_b"] = get_t_value(right_id, "t_i", refined_dict)
    else:
        current_vals["t_b"] = 0.0
    # Diag-Left neighbor
    if r > 0 and c > 0:
        diagL_id = to_id(r - 1, c - 1, width)
        current_vals["t_a"] = get_t_value(diagL_id, "t_b", refined_dict)
        current_vals["t_f"] = get_t_value(diagL_id, "t_i", refined_dict)
    else:
        current_vals["t_a"] = 0.0
        current_vals["t_f"] = 0.0
    return current_vals
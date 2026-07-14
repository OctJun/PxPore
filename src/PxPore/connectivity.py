import numpy as np
from numba import njit, prange

@njit(cache=True)
def _push_if_unvisited_void(label_flat, parent, q, tail, nb, cur):
    if label_flat[nb] == -1:
        label_flat[nb] = -2      # in current component (temporary mark)
        parent[nb] = cur
        q[tail] = nb
        return tail + 1
    return tail

@njit(cache=True)
def percolation_masks(void_mask, conn=6):
    """
    void_mask: uint8 (gx,gy,gz), 1=void, 0=solid
    conn: 6 or 26 (non-periodic)

    Returns:
      label_mask: int8 (gx,gy,gz)
          0 = solid
          1 = trap
          2 = accessible (spanning/percolating)
      parent_id: int32 (gx,gy,gz)
          linear parent index in BFS tree, -1 for none/solid
    """
    gx, gy, gz = void_mask.shape
    total = gx * gy * gz
    void_flat = void_mask.reshape(total)
    # label_flat doubles as visited:
    #  0  : solid
    # -1  : void & unvisited
    # -2  : void & in-current-component (temporary)
    #  1  : trap (final)
    #  2  : accessible (final)
    label_flat = np.empty(total, dtype=np.int8)
    parent = np.empty(total, dtype=np.int32)

    for i in range(total):
        if void_flat[i] == 1:
            label_flat[i] = -1
            parent[i] = -1
        else:
            label_flat[i] = 0
            parent[i] = -1

    q = np.empty(total, dtype=np.int32)

    stride_x = gy * gz
    stride_y = gz

    use26 = 1 if conn == 26 else 0

    for s in range(total):
        if label_flat[s] != -1:
            continue  # not an unvisited void

        # BFS start
        head = 0
        tail = 0
        label_flat[s] = -2
        parent[s] = -1
        q[tail] = s
        tail += 1

        touch_x0 = 0
        touch_x1 = 0
        touch_y0 = 0
        touch_y1 = 0
        touch_z0 = 0
        touch_z1 = 0

        while head < tail:
            cur = q[head]
            head += 1

            x = cur // stride_x
            rem = cur - x * stride_x
            y = rem // stride_y
            z = rem - y * stride_y

            if x == 0:      touch_x0 = 1
            if x == gx - 1: touch_x1 = 1
            if y == 0:      touch_y0 = 1
            if y == gy - 1: touch_y1 = 1
            if z == 0:      touch_z0 = 1
            if z == gz - 1: touch_z1 = 1

            # 6-neighborhood
            if x > 0:
                tail = _push_if_unvisited_void(label_flat, parent, q, tail, cur - stride_x, cur)
            if x + 1 < gx:
                tail = _push_if_unvisited_void(label_flat, parent, q, tail, cur + stride_x, cur)
            if y > 0:
                tail = _push_if_unvisited_void(label_flat, parent, q, tail, cur - stride_y, cur)
            if y + 1 < gy:
                tail = _push_if_unvisited_void(label_flat, parent, q, tail, cur + stride_y, cur)
            if z > 0:
                tail = _push_if_unvisited_void(label_flat, parent, q, tail, cur - 1, cur)
            if z + 1 < gz:
                tail = _push_if_unvisited_void(label_flat, parent, q, tail, cur + 1, cur)

            # 26-neighborhood: add diagonals
            if use26 == 1:
                for dx in (-1, 0, 1):
                    nx = x + dx
                    if nx < 0 or nx >= gx:
                        continue
                    for dy in (-1, 0, 1):
                        ny = y + dy
                        if ny < 0 or ny >= gy:
                            continue
                        for dz in (-1, 0, 1):
                            nz = z + dz
                            if nz < 0 or nz >= gz:
                                continue
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            nb = nx * stride_x + ny * stride_y + nz
                            tail = _push_if_unvisited_void(label_flat, parent, q, tail, nb, cur)

        # spanning? (your original definition)
        is_accessible = 0
        if (touch_x0 == 1 and touch_x1 == 1) or \
           (touch_y0 == 1 and touch_y1 == 1) or \
           (touch_z0 == 1 and touch_z1 == 1):
            is_accessible = 1

        final_lab = np.int8(2 if is_accessible == 1 else 1)

        # finalize labels for this component (q[0:tail))
        for i in range(tail):
            label_flat[q[i]] = final_lab

    label_mask = label_flat.reshape((gx, gy, gz))
    parent_id  = parent.reshape((gx, gy, gz))
    return label_mask, parent_id


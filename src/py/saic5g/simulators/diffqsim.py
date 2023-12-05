"""
A differentiable version of QSim written in PyTorch.
"""
import numpy as np
import torch
from torch.autograd import Function
from saic5g.scenarios.basic.geom import PrecomputedGeom
EPS = 1e-8

# TODO: test this stuff out on GPU

def iptput(ue_demand_alloc, tput_factors, cell_capacities):
    """
    Function to compute average IP throughput for each cell.

    IP throughput is computed as cell capacity / total demand  on the cell
    (scaled according to tput factor). This would technically allow for
    infinite iptputs for each cell with a low enough load. We may want to
    add some "max speed" clipping in the future.


    args:
        ue_demand_alloc ((x_pix, y_pix, n_cell, n_ue) Tensor): Desribes how much
            load from each UE is allocated to each cell in each geographic area.
            For the purposes of this function, a single UE may be distributed in
            space and allocated to multiple cells. Any kind of UE locality can be
            enforced at a higher level.
        tput_factors ((x_pix, y_pix, n_cell) Tensor): Quantifies connection quality
            between cells and points on the map. These should range from 0 (no connection)
            to 1 (perfect connection).
        cell_capacities ((n_cell,) Tensor): Capacity of each cell (in load units).

    returns:
        (n_cell,) Tensor: iptput for each cell
    """
    # The lower the tput factor the more capacity required to serve the demand
    cap_req = ue_demand_alloc / tput_factors[:,:,:,None]
    per_cell_demand = torch.sum(cap_req, dim=(0, 1, 3))
    per_cell_tput = cell_capacities / (per_cell_demand + EPS)
    return per_cell_tput


def packet_loss(ue_demand_alloc, tput_factors, cell_capacities):
    """
    Packet loss a la normal QSim

    Args:
        ue_demand_alloc ((x_pix, y_pix, n_cell, n_ue) Tensor): Desribes how much
            load from each UE is allocated to each cell in each geographic area.
            For the purposes of this function, a single UE may be distributed in
            space and allocated to multiple cells. Any kind of UE locality can be
            enforced at a higher level.
        tput_factors ((x_pix, y_pix, n_cell) Tensor): Quantifies connection quality
            between cells and points on the map. These should range from 0 (no connection)
            to 1 (perfect connection).
        cell_capacities ((n_cell,) Tensor): Capacity of each cell (in load units).
    
    Returns:
        float: system level packet loss
    """
    # EPS avoids divide by zero
    per_cell_demand = torch.sum(ue_demand_alloc, dim=(0, 1, 3)) + EPS
    per_cell_alloc_frac = cell_capacities / per_cell_demand
    # last dim is cell to facilitate multiplication
    tmp = ue_demand_alloc.permute(0, 1, 3, 2)
    # allocate capacity in proportion to demand
    ue_cap_alloc = (tmp * per_cell_alloc_frac).permute(0, 1, 3, 2)
    # amount that each UE can get through the network
    ue_data_cap = ue_cap_alloc * tput_factors[:,:,:,None]
    lost = torch.clamp(ue_demand_alloc - ue_data_cap, min=0)
    #  lost = ue_demand_alloc - ue_data_cap
    # Could easily be modified to give per UE or per cell PL, but
    # we usually care about optimizing network level.
    pl = lost.sum() / ue_demand_alloc.sum()
    return pl

def get_mpp(bbox, grid_shape):
    """
    Get meters per pixel in the bbox.

    Args:
        bbox (tuple): (min x, min y, max x, max y)
        shape (tuple)
    
    Returns:
        float: meters per pixel in x dimension
        float: meters per pixel in y dimension
    """
    xpix, ypix = grid_shape
    xmin, ymin, xmax, ymax = bbox
    xpixm = (xmax - xmin) / xpix
    ypixm = (ymax - ymin) / ypix
    return xpixm, ypixm

def get_grid_pos(bbox, grid_shape, ue_height=1.5):
    """
    Get positions of a uniform grid over the bbox.

    Args:
        bbox (tuple): (min x, min y, max x, max y)
        grid_shape (tuple)

    Returns:
        np.Array: grid_shape + (3,)
    """
    xpix, ypix = grid_shape
    xpixm, ypixm = get_mpp(bbox, grid_shape)
    xmin, ymin, xmax, ymax = bbox
    xpos = np.linspace(xmin + xpixm/2, xmax - xpixm/2, xpix)
    ypos = np.linspace(ymin + ypixm/2, ymax - ypixm/2, ypix)
    ue_pos = np.zeros((xpix*ypix, 3))
    ue_pos[:,0] = np.repeat(xpos, ypix)
    ue_pos[:,1] = np.tile(ypos, xpix)
    ue_pos[:,2] = ue_height
    return ue_pos

def rsrp_map(radio, cell_pos, bbox, out_shape, ue_height=1.5):
    """
    Create a 2D map of which signal is strongest in each geographic area.

    Args:
        radio (QSimRadioBase): Radio model
        cell_pos (np.array): n_cell x 3 numpy array specifying x,y,z
            cell postions in meters
        bbox (tuple): xmin, ymin, xmax, ymax in meters
        out_shape (tuple): xpixels, ypixels
        ue_height (float): UE height in meters
        
    returns:
        np.array: (xpixels, ypixels, n_cell) shaped array with RSRP values across
            the whole map for each cell.
    """
    n_cell = len(cell_pos)
    ue_pos = get_grid_pos(bbox, out_shape, ue_height)
    dist, azim = PrecomputedGeom.vector_cell_ue_distance_azimuth(cell_pos, ue_pos)
    rsrp = radio.rsrp(0, dist, azim).reshape((n_cell,) + out_shape)
    rsrp = np.moveaxis(rsrp, 0, -1)
    return rsrp

def tput_factor_map(radio, cell_pos, bbox, out_shape, flatten=True, ue_height=1.5):
    """
    Creates maps of throuput factor.

    A lower throughput factor means that a UE must consume more cell capacity to transmit
    its data.

    Args:
        cell_pos (np.array): n_cell x 3 numpy array specifying x,y,z
            cell postions in meters
        bbox (tuple): xmin, ymin, xmax, ymax in meters
        out_shape (tuple): xpixels, ypixels
        flatten (bool): If true, tput factor is set to max(tput_factor, 1)
        ue_height (float): UE height in meters
        
    Returns:
        np.array: (xpixels, ypixels, n_cell) shaped array with tput factor values across
            the whole map for each cell.
    """
    rsrp = rsrp_map(radio, cell_pos, bbox, out_shape, ue_height)
    n_cell = len(cell_pos)
    in_mean = radio.i_mean + radio.n_mean
    # assume this doesn't change with time
    n_prb = radio.n_prb(0, n_cell)[:, None]
    rsrp_multiplier = 12 * radio.kwargs.get('activity_factor', 3.) * radio.kwargs.get('n_prb', 6)
    sinr = rsrp * rsrp_multiplier / in_mean
    tput_factor = np.log2(1 + sinr)
    if flatten:
        tput_factor = np.minimum(1, tput_factor)
    return tput_factor


class UeDemandAllocator(Function):
    """
    Performs mapping between the standard formulation, where each
    UE has an (x,y) position, a demand, and an assignment to a single
    cell into the (x_pix, y_pix, n_cell, n_ue) dimensional
    tensor which is easy to automatically differentiate. Of course we 
    handle the backwards case as well. 
    
    See https://pytorch.org/docs/stable/notes/extending.html#extending-autograd
    for more context. 
    """

    @staticmethod
    def forward(ctx, ue_pos, demand, assign, grid_shape):
        """
        Args:
            ctx: torch Context
            ue_pos (torch.Tensor): (n_ues, 2) dimensional tensor with the last dimension
                encoding x, y. x and y positions should be in the range [0, 1].
            demand (torch.Tensor): (n_ues,) dimensional tensor encoding the demand of each UE.
            assign (torch.Tensor): (n_cell, n_ues) dimensional tensor, a one-hot encoding of ue-cell 
                association. We use this input shape so as to be able to provide gradients for
                assigning the UE to each cell, and have the gradients be the same shape as the input.
            grid_shape (tuple): shape of the grid
        """
        n_cell, n_ue  = assign.shape
        out = torch.zeros(grid_shape + (n_cell, n_ue))
        # reasonably accurate as long as grid is sufficiently large
        x_idx = torch.clamp((ue_pos[:, 0] * grid_shape[0]).long(), 0, grid_shape[0] - 1)
        y_idx = torch.clamp((ue_pos[:, 1] * grid_shape[1]).long(), 0, grid_shape[1] - 1)
        assign_idx = torch.argmax(assign, 0)
        # allocate demand, negative treated as 0 using relu
        out[x_idx, y_idx, assign_idx, torch.arange(n_ue)] = torch.nn.functional.relu(demand)
        ctx.save_for_backward(x_idx, y_idx, assign_idx, torch.tensor(grid_shape))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_idx, y_idx, assign_idx, grid_shape = ctx.saved_tensors
        n_x, n_y, n_cell, n_ue = grad_output.shape
        ue_pos_grad = demand_grad = assign_grad = None
        ue_idx = torch.arange(n_ue)
        if ctx.needs_input_grad[0]:
            x_idx_right = torch.clamp(x_idx + 1, 0, grid_shape[0] - 1)
            x_idx_left = torch.clamp(x_idx - 1, 0, grid_shape[0] - 1)
            y_idx_top = torch.clamp(y_idx + 1, 0, grid_shape[1] - 1)
            y_idx_bottom = torch.clamp(y_idx - 1, 0, grid_shape[1] - 1)
            ue_pos_grad = torch.zeros((n_ue, 2))
            dx = grad_output[x_idx_right, y_idx, assign_idx, ue_idx] - grad_output[x_idx_left, y_idx, assign_idx, ue_idx]
            dy = grad_output[x_idx, y_idx_top, assign_idx, ue_idx] - grad_output[x_idx, y_idx_bottom, assign_idx, ue_idx]
            ue_pos_grad[:, 0] = dx
            ue_pos_grad[:, 1] = dy
        if ctx.needs_input_grad[1]:
            demand_grad = grad_output[x_idx, y_idx, assign_idx, ue_idx]
        if ctx.needs_input_grad[2]:
            assign_grad = torch.transpose(grad_output[x_idx, y_idx, :, ue_idx], 0, 1)

        # Last input is non-differentiable
        return ue_pos_grad, demand_grad, assign_grad,  None

allocate_demand = UeDemandAllocator.apply

import numpy as np
from PIL import Image
from numba import jit, prange
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod
from os.path import basename
from typing import List
import functools


def NI_decor(fn):
    def wrap_fn(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except NotImplementedError as e:
            print(e)

    return wrap_fn


@jit(nopython=True)
def _rgb_to_grayscale(np_img):
    gs_img = np_img[:,:,0]*np.float32(0.2989) + np_img[:,:,1]*np.float32(0.5870) + np_img[:,:,2]*np.float32(0.1140)
    gs_img[0, :] = .5
    gs_img[-1, :] = .5
    gs_img[:, 0] = .5
    gs_img[:, -1] = .5
    return gs_img


@jit(nopython=True, parallel=True)
def _calc_gradient_magnitude(gs_src):
    sobel_x = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=np.float32)
    sobel_y = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=np.float32)

    h, w = gs_src.shape
    gs_padded = np.zeros((h + 2, w + 2), dtype=np.float32)
    gs_padded[1:h+1, 1:w+1] = gs_src
    gs_padded[0, :] = 0.5
    gs_padded[h+1, :] = 0.5
    gs_padded[:, 0] = 0.5
    gs_padded[:, w+1] = 0.5

    gradient_mag = np.zeros((h, w), dtype=np.float32)
    for y in prange(h):
        for x in range(w):
            gx = np.float32(0.0)
            gy = np.float32(0.0)
            for dy in range(3):
                for dx in range(3):
                    v = gs_padded[y + dy, x + dx]
                    gx += v * sobel_x[dy, dx]
                    gy += v * sobel_y[dy, dx]
            gradient_mag[y, x] = np.sqrt(gx * gx + gy * gy)

    max_val = np.max(gradient_mag)
    if max_val > 0:
        gradient_mag = gradient_mag / max_val
    return gradient_mag


class SeamImage:
    def __init__(self, img_path: str, vis_seams: bool = True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path

        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T

        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()

        self.H, self.W = self.rgb.shape[:2] # -> original image dimaensions
        self.h, self.w = self.H, self.W

        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.zeros_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices 
        xx, yy = np.meshgrid(range(self.w), range(self.h))
        self.idx_map = np.stack((yy, xx), axis=-1)

    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        return _rgb_to_grayscale(np_img.astype(np.float32))
    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            - In order to calculate a gradient of a pixel, only its neighborhood is required.
            - keep in mind that values must be in range [0,1]
            - np.gradient or other off-the-shelf tools are NOT allowed, however feel free to compare yourself to them
        """
        return _calc_gradient_magnitude(self.resized_gs.astype(np.float32))


    def update_ref_mat(self):
        """ Updates matrices for seam visualization

        Guidelines & hints:
            - Given the latest computed seam, you need to track its original indices and mark them (self.cumm_mask) using self.ixd_map
            - Resize self.idx_map each seam update
        """
        seam = np.array(self.seam_history[-1])
        rows = np.arange(len(seam))
        orig_yx = self.idx_map[rows, seam]          # shape (h, 2)
        self.cumm_mask[orig_yx[:, 0].astype(int),
                       orig_yx[:, 1].astype(int)] = True

    def reinit(self):
        """
        Re-initiates instance and resets all variables.
        """
        self.__init__(img_path=self.path)

    @staticmethod
    def load_image(img_path, format='RGB'):
        return np.asarray(Image.open(img_path).convert(format)).astype('float32') / 255.0

    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seams to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, mask) where:
                - E is the gradient magnitude matrix
                - mask is a boolean matrix for removed seams
            iii) find the best seam to remove and store it
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the chosen seam (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you wish, but it needs to support:
            - removing seams a couple of times (call the function more than once)
            - visualize the original image with removed seams marked in red (for comparison)

        NOTE: you may not use np.gradient or other off-the-shelf tools for gradient calculation, but you can use them to compare your results.
        """
        for _ in tqdm(range(num_remove)):
            self.E = self.calc_gradient_magnitude()
            self.mask = np.ones_like(self.E, dtype=bool)

            seam = self.find_minimal_seam()
            self.seam_history.append(seam)
            if self.vis_seams:
                self.update_ref_mat()
            self.remove_seam(seam)

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the seam with the minimal energy.
        Returns:
            The found seam, represented as a list of indexes
        """
        raise NotImplementedError("TODO: Implement SeamImage.find_minimal_seam in one of the subclasses")

    def remove_seam(self, seam: List[int]):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using:
        3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.

        :arg seam: The seam to remove
        """
        # Create a boolean mask for pixels to keep (True = keep, False = remove)
        mask = np.ones((self.h, self.w), dtype=bool)
        
        # Mark seam pixels as False (to be removed) using vectorized indexing. 
        # This is more efficient than a loop.
        mask[np.arange(self.h), seam] = False
        
        # Expand mask to 3 channels (RGB)
        mask_3d = np.stack([mask] * 3, axis=2)
        
        # Remove seam from resized_rgb, by udating it with the mask and reshaping it to the new dimensions
        self.resized_rgb = self.resized_rgb[mask_3d].reshape(self.h, self.w - 1, 3)
        
        # Update grayscale version
        self.resized_gs = self.resized_gs[mask].reshape(self.h, self.w - 1)
        
        # Look up original coordinates BEFORE shrinking idx_map
        if self.vis_seams:
            orig_yx = self.idx_map[np.arange(self.h), seam]

        # Update idx_map by removing the seam column
        self.idx_map = self.idx_map[mask].reshape(self.h, self.w - 1, 2)

        # Mark seam on visualization using original coordinates
        if self.vis_seams:
            self.seams_rgb[orig_yx[:, 0].astype(int), orig_yx[:, 1].astype(int)] = [1.0, 0.0, 0.0]
        
        # Update dimensions
        self.w -= 1

    def rotate_mats(self, clockwise: bool):
        """
        Rotates the matrices either clockwise or counter-clockwise.
        """
        # Determine rotation direction: -1 for clockwise, 1 for counter-clockwise
        k = -1 if clockwise else 1
        
        # Rotate all matrices
        self.resized_rgb = np.rot90(self.resized_rgb, k=k)
        self.resized_gs = np.rot90(self.resized_gs, k=k)
        self.E = np.rot90(self.E, k=k)
        # cumm_mask and seams_rgb track original-image coordinates — never rotate them

        # Rotate index map
        self.idx_map = np.rot90(self.idx_map, k=k)
        
        # Swap height and width after rotation
        self.h, self.w = self.w, self.h

    def seams_removal_vertical(self, num_remove: int):
        """ A wrapper for removing num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seam to be removed
        """
        self.seams_removal(num_remove)

    def seams_removal_horizontal(self, num_remove: int):
        """ Removes num_remove horizontal seams by rotating the image, removing vertical seams, and restoring the original rotation.

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
        - No need to reimplement SC for horizontal seam removal!
        - Once you figure out how, this method should look like:
                SOME_OPERATION(...)
                seam_removal(...)
                SOME_OPERATION(...)
            and thats it!
        """
        self.rotate_mats(clockwise=False)  # Rotate counter-clockwise 90°
        self.seams_removal(num_remove)     # Remove vertical seams (which are now horizontal)
        self.rotate_mats(clockwise=True)   # Rotate back clockwise 90°

    """
    BONUS SECTION
    """

    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seams to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        # ── Step 1: save current state before the discovery removals ──────────
        saved_rgb     = self.resized_rgb.copy()
        saved_gs      = self.resized_gs.copy()
        saved_idx_map = self.idx_map.copy()
        saved_w       = self.w

        # ── Step 2: find the num_add cheapest seams via temporary removal ─────
        # Disable vis so the discovery pass doesn't paint red / touch cumm_mask.
        saved_vis      = self.vis_seams
        self.vis_seams = False
        hist_start     = len(self.seam_history)
        self.seams_removal(num_add)
        self.vis_seams = saved_vis
        # seams recorded in shrinking-frame coordinates, in order of removal
        new_seams = self.seam_history[hist_start:]

        # ── Step 3: restore state ─────────────────────────────────────────────
        self.resized_rgb = saved_rgb
        self.resized_gs  = saved_gs
        self.idx_map     = saved_idx_map
        self.w           = saved_w

        # ── Step 4: map shrinking-frame indices back to original-frame cols ───
        # seams_orig[k, i] = column in the *original* (pre-removal) frame for
        # seam k at row i.  Each removal shifts indices of later seams rightward
        # by 1 for every earlier seam that sat at or to its left.
        seams_orig = np.array(new_seams, dtype=int)   # shape (num_add, h)
        for k in range(1, num_add):
            for s in range(k):
                seams_orig[k] += (seams_orig[s] <= seams_orig[k]).astype(int)

        # ── Step 5: insert all seams into the live arrays ────────────────────
        new_rgb     = np.zeros((self.h, self.w + num_add, 3), dtype=np.float32)
        new_gs      = np.zeros((self.h, self.w + num_add),    dtype=np.float32)
        new_idx_map = np.zeros((self.h, self.w + num_add, 2), dtype=self.idx_map.dtype)

        for i in range(self.h):
            # Original-frame insertion columns for this row, sorted ascending
            cols = sorted(seams_orig[:, i])

            rgb_row = list(self.resized_rgb[i])   # lists allow O(1) insert
            gs_row  = list(self.resized_gs[i])
            idx_row = list(self.idx_map[i])

            for rank, orig_col in enumerate(cols):
                # Each earlier insertion shifts the target position right by 1
                ins = orig_col + rank + 1
                ins = min(ins, len(rgb_row))   # clamp at row end

                # New pixel = average of left and right neighbours
                left_rgb  = np.array(rgb_row[ins - 1], dtype=np.float32)
                right_rgb = np.array(rgb_row[min(ins, len(rgb_row) - 1)], dtype=np.float32)
                left_gs   = gs_row[ins - 1]
                right_gs  = gs_row[min(ins, len(gs_row) - 1)]

                rgb_row.insert(ins, ((left_rgb + right_rgb) / 2).astype(np.float32))
                gs_row.insert(ins,  np.float32((left_gs + right_gs) / 2))
                idx_row.insert(ins, idx_row[ins - 1])  # inherit neighbour's orig coords

                # Paint the corresponding original pixel green in seams_rgb
                if self.vis_seams:
                    orig_y, orig_x = self.idx_map[i, orig_col]
                    self.seams_rgb[int(orig_y), int(orig_x)] = [0.0, 1.0, 0.0]

            new_rgb[i]     = np.stack(rgb_row)
            new_gs[i]      = np.array(gs_row, dtype=np.float32)
            new_idx_map[i] = np.stack(idx_row)

        self.resized_rgb = new_rgb
        self.resized_gs  = new_gs
        self.idx_map     = new_idx_map
        self.w          += num_add

    def seams_addition_horizontal(self, num_add: int):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_add (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        self.rotate_mats(clockwise=False)
        self.seams_addition(num_add)
        self.rotate_mats(clockwise=True)

    def seams_addition_vertical(self, num_add: int):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """
        self.seams_addition(num_add)


class GreedySeamImage(SeamImage):
    """Implementation of the Seam Carving algorithm using a greedy approach"""

    @staticmethod
    @jit(nopython=True)
    def find_seam_static(E, h, w):
        seam = np.zeros(h, dtype=np.int64)
        seam[0] = np.argmin(E[0, :])
        for i in range(1, h):
            prev = seam[i - 1]
            lo = max(prev - 1, np.int64(0))
            hi = min(prev + 2, np.int64(w))
            best_j = lo
            best_e = E[i, lo]
            for jj in range(lo + 1, hi):
                if E[i, jj] < best_e:
                    best_e = E[i, jj]
                    best_j = jj
            seam[i] = best_j
        return seam

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using a greedy algorithm.

        Guidelines & hints:
        The first pixel of the seam should be the pixel with the lowest cost.
        Every row chooses the next pixel based on which neighbor has the lowest cost.
        """
        return list(GreedySeamImage.find_seam_static(
            self.E.astype(np.float32), np.int64(self.h), np.int64(self.w)
        ))


class DPSeamImage(SeamImage):
    """
    Implementation of the Seam Carving algorithm using dynamic programming (DP).
    """

    def __init__(self, *args, **kwargs):
        """ DPSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    @staticmethod
    @jit(nopython=True)
    def calc_M_static(E, gs):
        h, w = E.shape
        M = np.zeros((h, w), dtype=np.float32)
        M[0, :] = E[0, :]
        for i in range(1, h):
            for j in range(w):
                left  = gs[i, j - 1] if j > 0     else gs[i, j]
                right = gs[i, j + 1] if j < w - 1 else gs[i, j]
                c_v = abs(right - left)
                c_l = c_v + abs(gs[i-1, j] - left)  if j > 0     else np.inf
                c_r = c_v + abs(gs[i-1, j] - right) if j < w - 1 else np.inf
                m_left  = M[i-1, j-1] + c_l if j > 0     else np.inf
                m_up    = M[i-1, j]   + c_v
                m_right = M[i-1, j+1] + c_r if j < w - 1 else np.inf
                M[i, j] = E[i, j] + min(m_left, m_up, m_right)
        return M

    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        return DPSeamImage.calc_M_static(
            self.E.astype(np.float32),
            self.resized_gs.astype(np.float32)
        )

    def init_mats(self):
        self.M = self.calc_M()
        self.backtrack_mat = np.zeros_like(self.M, dtype=int)

    @staticmethod
    @jit(nopython=True)
    def calc_bt_mat(M, E, GS, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.

        Recommended parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            E: np.ndarray (float32) of shape (h,w)
            GS: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a reference type. Changing it here may affect it on the outside.
        """
        raise NotImplementedError("TODO: Implement DPSeamImage.calc_bt_mat")
        h, w = M.shape


    @staticmethod
    @jit(nopython=True)
    def find_seam_static(M, gs):
        h, w = M.shape
        backtrack = np.zeros((h, w), dtype=np.int32)
        for i in range(1, h):
            for j in range(w):
                left  = gs[i, j-1] if j > 0     else gs[i, j]
                right = gs[i, j+1] if j < w - 1 else gs[i, j]
                c_v = abs(right - left)
                c_l = c_v + abs(gs[i-1, j] - left)  if j > 0     else np.inf
                c_r = c_v + abs(gs[i-1, j] - right) if j < w - 1 else np.inf
                m_left  = M[i-1, j-1] + c_l if j > 0     else np.inf
                m_up    = M[i-1, j]   + c_v
                m_right = M[i-1, j+1] + c_r if j < w - 1 else np.inf
                best = m_up
                bt = np.int32(1)
                if m_left < best:
                    best = m_left
                    bt = np.int32(0)
                if m_right < best:
                    bt = np.int32(2)
                backtrack[i, j] = bt
        seam = np.zeros(h, dtype=np.int64)
        seam[h-1] = np.argmin(M[h-1, :])
        for i in range(h-2, -1, -1):
            j = seam[i+1]
            bt = backtrack[i+1, j]
            if bt == np.int32(0):   seam[i] = j - 1
            elif bt == np.int32(2): seam[i] = j + 1
            else:                   seam[i] = j
        return seam

    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using dynamic programming.

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (M, backtracking matrix) where:
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
        """
        self.M = self.calc_M()
        seam = DPSeamImage.find_seam_static(
            self.M,
            self.resized_gs.astype(np.float32)
        )
        return list(seam)

def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    new_height = int(orig_shape[0] * scale_factors[0])
    new_width = int(orig_shape[1] * scale_factors[1])
    return np.array([new_height, new_width], dtype=int)


def resize_seam_carving(seam_img: SeamImage, shapes: tuple):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (tuple): (orig_shape, new_shape) where each is [height, width]

    Returns
        the resized rgb image
    """
    orig_shape, new_shape = shapes
    orig_h, orig_w = orig_shape[0], orig_shape[1]
    new_h, new_w = new_shape[0], new_shape[1]
    
    # Work on a fresh instance to avoid mutating the input object
    sc_cls = seam_img.__class__
    sc_obj = sc_cls(img_path=seam_img.path, vis_seams=seam_img.vis_seams)
    
    # Calculate seams to remove
    num_vertical_remove = orig_w - new_w
    num_horizontal_remove = orig_h - new_h
    
    # Remove vertical seams first
    if num_vertical_remove > 0:
        sc_obj.seams_removal_vertical(num_vertical_remove)
    
    # Remove horizontal seams
    if num_horizontal_remove > 0:
        sc_obj.seams_removal_horizontal(num_horizontal_remove)
    
    return sc_obj.resized_rgb


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)

    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org

    scaled_x_grid = [get_scaled_param(x, in_width, out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y, in_height, out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid, dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid, dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:, x1s] * dx + (1 - dx) * image[y1s][:, x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:, x1s] * dx + (1 - dx) * image[y2s][:, x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image



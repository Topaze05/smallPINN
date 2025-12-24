import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import h5py
import torch
from torch.utils.data import DataLoader, Dataset

dataset_path = "Put the path of the file containing the simulations here"


class SWEDataset(Dataset):
    def __init__(self, h5_file_path, transform=None, nb_points: int = 300_000):
        """
        Args:
            h5_file_path (str): Path to the .h5 file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.h5_file_path = h5_file_path
        self.transform = transform

        # We open the file once in __init__ just to get the list of keys (simulation IDs)
        # We close it immediately to avoid pickling issues with multi-process dataloaders
        with h5py.File(h5_file_path, 'r') as f:
            # We sort the keys to ensure deterministic ordering
            self.keys = sorted(list(f.keys()))

        # This will hold the actual file handle in the worker process
        self.file = None

        # Number of points to load per simulation
        self.nb_points = nb_points

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        This function loads ONE simulation from disk.
        """
        if self.file is None:
            # Open the file in read-only mode.
            # This happens once per worker process.
            self.file = h5py.File(self.h5_file_path, 'r')

        group_key = self.keys[idx]

        # 1. Load the data lazily
        # Shape: (101, 128, 128, 3)
        # We use [:] to force h5py to read the data from disk into a numpy array
        data_numpy = self.file[group_key]['data'][:]

        total_points = data_numpy.shape[0] * data_numpy.shape[1] * data_numpy.shape[2]
        idx = np.random.choice(total_points, self.nb_points, replace=False)

        # Height and velocity
        HH = data_numpy[:, :, :, 0].flatten()[:, None][idx]
        UU = data_numpy[:, :, :, 1].flatten()[:, None][idx]
        VV = data_numpy[:, :, :, 2].flatten()[:, None][idx]

        XX = np.tile(self.file[group_key]['grid']['x'][:], (1, data_numpy.shape[0] * data_numpy.shape[1])).flatten()[
            :, None][idx]
        YY = np.tile(self.file[group_key]['grid']['y'][:], (1, data_numpy.shape[0] * data_numpy.shape[1])).flatten()[
            :, None][idx]
        TT = np.tile(self.file[group_key]['grid']['t'][:], (1, data_numpy.shape[1] * data_numpy.shape[1])).flatten()[
            :, None][idx]

        # 2. Convert to PyTorch Tensor
        # It is usually best to work with float32 for PINNs
        X_train = torch.from_numpy(np.hstack((XX, YY, TT))).float()
        Y_train = torch.from_numpy(np.hstack((HH, UU, VV))).float()

        if self.transform:
            X_train = self.transform(X_train)
            Y_train = self.transform(Y_train)

        return X_train, Y_train

    def get_full_item(self, index: int = 0):
        """
        This function loads ONE simulation from disk and return the full data.

        :param index: Index of the simulation to load.
        :return:
        """
        if self.file is None:
            # Open the file in read-only mode.
            # This happens once per worker process.
            self.file = h5py.File(self.h5_file_path, 'r')

        group_key = self.keys[index]

        return self.file[group_key]['data'][:], self.file[group_key]['grid']


class SWEDataloader:
    def __init__(self, dataset_path, index: int = None, num_points: int = 30_000, transform = None, batch_size: int = 16):
        self.dataset = SWEDataset(dataset_path)
        self.index = index if index is not None else np.random.randint(len(self.dataset))
        self.num_points = num_points
        self.transform = transform
        self.batch_size = batch_size

        self.data, temp = self.dataset.get_full_item(self.index)

        self.grid = {
            "x": temp["x"][:],
            "y": temp["y"][:],
            "t": temp["t"][:]
        }

    def __iter__(self):
        return self

    def __next__(self):
        # 1. Prepare Flattened Arrays (Data and Coordinates)
        # We assume self.data shape is (T, X, Y, 3) or (T, X, Y, vars)
        # indexing='ij' ensures the meshgrid matches the matrix indexing order of self.data
        grid_t, grid_x, grid_y = np.meshgrid(
            self.grid['t'],
            self.grid['x'],
            self.grid['y'],
            indexing='ij'
        )

        # Flatten everything once
        # These will be 1D arrays of length (T*X*Y)
        flat_t = grid_t.flatten()
        flat_x = grid_x.flatten()
        flat_y = grid_y.flatten()

        flat_h = self.data[..., 0].flatten()
        flat_u = self.data[..., 1].flatten()
        flat_v = self.data[..., 2].flatten()

        total_points = flat_h.shape[0]

        # 2. Select Random Indices for the Batch
        # We create a matrix of random indices: (batch_size, num_points)
        # This gives us 'batch_size' different random samplings from the simulation
        batch_indices = np.empty((self.batch_size, self.num_points), dtype=int)

        for i in range(self.batch_size):
            # replace=False ensures unique points within one sample set
            batch_indices[i] = np.random.choice(total_points, self.num_points, replace=False)

        # 3. Extract Data using Advanced Indexing
        # The result shape will be (batch_size, num_points)
        # We add [..., None] to make it (batch_size, num_points, 1) for concatenation

        # Coordinates
        bt = flat_t[batch_indices][..., None]
        bx = flat_x[batch_indices][..., None]
        by = flat_y[batch_indices][..., None]

        # Data
        bh = flat_h[batch_indices][..., None]
        bu = flat_u[batch_indices][..., None]
        bv = flat_v[batch_indices][..., None]

        # 4. Stack and Convert to Tensor
        # IMPORTANT: Ordering changed to (T, X, Y) to match PDE_Models.py expectations
        X_train_np = np.concatenate((bt, bx, by), axis=2)  # Shape: (Batch, Num_Points, 3)
        Y_train_np = np.concatenate((bh, bu, bv), axis=2)  # Shape: (Batch, Num_Points, 3)

        X_train = torch.from_numpy(X_train_np).float()
        Y_train = torch.from_numpy(Y_train_np).float()

        if self.transform:
            X_train = self.transform(X_train)
            Y_train = self.transform(Y_train)

        return X_train, Y_train

    def get_grid(self):
        # indexing='ij' ensures the order matches the matrix dimensions (Nt, Nx, Ny)
        grid_t, grid_x, grid_y = np.meshgrid(
            self.grid['t'],
            self.grid['x'],
            self.grid['y'],
            indexing='ij'
        )

        # Flatten and stack them into a (Total_Points, 3) matrix
        # Order will be: t (slowest), x (middle), y (fastest)
        flat_t = grid_t.flatten()
        flat_x = grid_x.flatten()
        flat_y = grid_y.flatten()

        # Shape: (N_points, 3) -> columns are [t, x, y]
        grid_all = np.stack([flat_t, flat_x, flat_y], axis=1)

        return torch.from_numpy(grid_all).float()

    def get_grid_shape(self):
        return self.grid['t'].shape[0], self.grid['x'].shape[0], self.grid['y'].shape[0]


    def get_data(self):
        # Height and velocity
        HH = self.data[:, :, :, 0].flatten()[:, None]
        UU = self.data[:, :, :, 1].flatten()[:, None]
        VV = self.data[:, :, :, 2].flatten()[:, None]

        # 2. Convert to PyTorch Tensor
        # It is usually best to work with float32 for PINNs
        X_test = self.get_grid()
        Y_test = torch.from_numpy(np.hstack((HH, UU, VV))).float()

        return X_test, Y_test


def example_usage():
    # Instantiate the Dataset
    dataset = SWEDataset(dataset_path)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Iterate in your training loop
    print("Starting Data Loading Test...")

    for batch_idx, (batch_data_x, batch_data_y) in enumerate(dataloader):
        # batch_data shape will be: [Batch_Size, 101, 128, 128, 3]
        print(f"Batch {batch_idx} shape x: {batch_data_x.shape} shape y: {batch_data_y.shape}")

        # --- PINN Training Logic would go here ---
        # For testing, break after one batch
        break


def show_3D_graph(x, y, h, t=0):
    # Create a grid for plotting
    # indexing='ij' ensures the mesh matches the array matrix order (x, y)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Create the 3D Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(X, Y, h, cmap='viridis', edgecolor='none')

    # Add labels and colorbar
    ax.set_title(f"Water Height (h) at t={t}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Height")
    fig.colorbar(surf, label="Water Height", shrink=0.5, aspect=5)

    plt.show()


def generate_comparison_gif(dataloader, prediction_data, output_filename='comparison.gif', mode='2d'):
    """
    Generates a side-by-side GIF of Ground Truth vs PINN Prediction.

    :param dataloader: The instance of SWEDataset used for training/testing
    :param prediction_data: The numpy array output from swe_model.simulate()
                            Expected shape: (nt, nx, ny, 3)
    :param output_filename: Output filename
    :param mode: '2d' for Heatmap, '3d' for Surface plot
    """
    print(f"Preparing {mode.upper()} animation data...")

    # 1. Retrieve Dimensions and Data
    nt, nx, ny = dataloader.get_grid_shape()
    # Get Coordinates for 3D plotting
    x_vals = dataloader.grid['x']
    y_vals = dataloader.grid['y']

    # Get Ground Truth
    _, Y_test_flat = dataloader.get_data()
    if isinstance(Y_test_flat, torch.Tensor):
        Y_test_flat = Y_test_flat.numpy()
    true_data = Y_test_flat.reshape(nt, nx, ny, 3)

    # 2. Extract Variable (Height h)
    var_index = 0
    var_name = "Water Height (h)"
    h_true = true_data[:, :, :, var_index]
    h_pred = prediction_data[:, :, :, var_index]

    # 3. Global Limits (Crucial for fixed scales)
    global_min = min(np.min(h_true), np.min(h_pred))
    global_max = max(np.max(h_true), np.max(h_pred))

    # 4. Setup Figure
    if mode == '3d':
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})
        # Create Grid for 3D Plotting (indexing='ij' matches data shape)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].set_title("Ground Truth")
    axes[1].set_title("PINN Prediction")

    # Time Text
    time_text = fig.text(0.5, 0.05, '', ha='center', fontsize=12)

    # ---------------------------------------------------------
    # 2D IMPLEMENTATION
    # ---------------------------------------------------------
    if mode == '2d':
        # origin='lower' puts (0,0) at bottom-left
        # We transpose (.T) because imshow expects (Rows=Y, Cols=X)
        im1 = axes[0].imshow(h_true[0].T, cmap='viridis', origin='lower', vmin=global_min, vmax=global_max)
        im2 = axes[1].imshow(h_pred[0].T, cmap='viridis', origin='lower', vmin=global_min, vmax=global_max)

        fig.colorbar(im1, ax=axes.ravel().tolist(), label=var_name, shrink=0.8)

        def update_2d(frame_idx):
            im1.set_data(h_true[frame_idx].T)
            im2.set_data(h_pred[frame_idx].T)

            t_val = dataloader.grid['t'][frame_idx]
            time_text.set_text(f"Time: {t_val:.3f} s")
            return im1, im2, time_text

        anim_func = update_2d
        # Blit works well for 2D
        use_blit = True

    # ---------------------------------------------------------
    # 3D IMPLEMENTATION
    # ---------------------------------------------------------
    elif mode == '3d':
        # Container to store surface objects so we can remove them
        plot_surfs = [None, None]

        # Set constant Z-axis limits so the box doesn't resize
        # Adding a small margin
        z_margin = (global_max - global_min) * 0.1
        for ax in axes:
            ax.set_zlim(global_min - z_margin, global_max + z_margin)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('h')

        def update_3d(frame_idx):
            # Remove old surfaces
            if plot_surfs[0] is not None:
                plot_surfs[0].remove()
                plot_surfs[1].remove()

            # Plot new surfaces
            # rstride/cstride controls resolution (higher = faster rendering, coarser look)
            surf1 = axes[0].plot_surface(X, Y, h_true[frame_idx], cmap='viridis',
                                         vmin=global_min, vmax=global_max, rstride=2, cstride=2)
            surf2 = axes[1].plot_surface(X, Y, h_pred[frame_idx], cmap='viridis',
                                         vmin=global_min, vmax=global_max, rstride=2, cstride=2)

            plot_surfs[0] = surf1
            plot_surfs[1] = surf2

            t_val = dataloader.grid['t'][frame_idx]
            time_text.set_text(f"Time: {t_val:.3f} s")

            # 3D animation in Matplotlib is tricky with blit, usually return empty or disable blit
            return surf1, surf2, time_text

        anim_func = update_3d
        # Blit is often buggy with 3D plot_surface removal, safer to disable
        use_blit = False

    else:
        raise ValueError("Mode must be '2d' or '3d'")

    # 5. Generate and Save
    print(f"Generating {mode.upper()} GIF ({nt} frames)...")
    ani = FuncAnimation(fig, anim_func, frames=nt, interval=100, blit=use_blit)

    ani.save(output_filename, writer='pillow', fps=10)
    print(f"Saved to {output_filename}")
    plt.close()


def get_info_dataset():
    try:
        with h5py.File(dataset_path, 'r') as f:
            # print(f.keys())
            ls_key = list(f.keys())
            print(len(f.keys()))
            print(f[ls_key[0]])
            print(f[ls_key[0]].keys())
            print(f[ls_key[0]]['data'].shape)
            print(f[ls_key[0]]['grid'].keys())
            print(f[ls_key[0]]['grid']['t'].shape)
            print(f[ls_key[0]]['grid']['x'].shape)
            print(f[ls_key[0]]['grid']['y'].shape)

            data = np.asarray(f[ls_key[0]]['data'])
            print(data.shape)
            print(data.dtype)
            print(type(data))

            # show_3D_graph(f[ls_key[0]]['grid']['x'], f[ls_key[0]]['grid']['y'], f[ls_key[0]]['data'][0, :, :, 0])
            # show_3D_graph(f[ls_key[0]]['grid']['x'], f[ls_key[0]]['grid']['y'], f[ls_key[0]]['data'][10, :, :, 0])

    except Exception as e:
        print(f"Loading the dataset failed: {e}")


if __name__ == "__main__":
    # get_info_dataset()
    # create_pde_video(dataset_path)
    example_usage()

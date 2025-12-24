import json

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from tqdm import tqdm

from dataloader_ import SWEDataloader
from models import PINN_MLP, FF_PINN, TRM_PINN, MaskPINN, AdaptedTRMPINN

dataset_path = "Put the path of the file containing the simulations here"


class NavierStokesModel:
    def __init__(self, model, X_train, Y_train, lambda_1=1.0, lambda_2=0.01):
        self.model = model
        device = next(model.parameters()).device

        self.t = torch.tensor(X_train[:, 0:1], requires_grad=True).float().to(device)
        self.x = torch.tensor(X_train[:, 1:2], requires_grad=True).float().to(device)
        self.y = torch.tensor(X_train[:, 2:3], requires_grad=True).float().to(device)

        self.u_m = torch.tensor(Y_train[:, 0:1]).float().to(device)
        self.v_m = torch.tensor(Y_train[:, 1:2]).float().to(device)

        self.l1 = lambda_1
        self.l2 = lambda_2

    def net(self, t, x, y):
        inp = torch.cat([t, x, y], dim=1)
        out = self.model(inp)

        psi = out[:, 0:1]
        p = out[:, 1:2]

        u = torch.autograd.grad(psi, y, torch.ones_like(psi), create_graph=True)[0]
        v = -torch.autograd.grad(psi, x, torch.ones_like(psi), create_graph=True)[0]

        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]

        v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]

        f_u = u_t + self.l1 * (u * u_x + v * u_y) + p_x - self.l2 * (u_xx + u_yy)
        f_v = v_t + self.l1 * (u * v_x + v * v_y) + p_y - self.l2 * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    def loss(self):
        u_pred, v_pred, _, f_u, f_v = self.net(self.t, self.x, self.y)
        return ((u_pred - self.u_m) ** 2).mean() + \
            ((v_pred - self.v_m) ** 2).mean() + \
            (f_u ** 2).mean() + (f_v ** 2).mean()

    def train(self, Adam_steps=800, LBFGS_steps=200):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for _ in range(Adam_steps):
            optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            optimizer.step()

        def closure():
            optimizer_lb.zero_grad()
            loss = self.loss()
            loss.backward()
            return loss

        optimizer_lb = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=LBFGS_steps,
            line_search_fn="strong_wolfe"
        )
        optimizer_lb.step(closure)

    def predict(self, X_full, batch=20000):
        device = next(self.model.parameters()).device
        u_list = []

        for i in range(0, len(X_full), batch):
            chunk = X_full[i:i + batch]
            xbatch = torch.tensor(chunk, dtype=torch.float32, device=device, requires_grad=True)

            t = xbatch[:, 0:1]
            x = xbatch[:, 1:2]
            y = xbatch[:, 2:3]

            x.requires_grad_()
            y.requires_grad_()

            out = self.model(torch.cat([t, x, y], dim=1))
            psi = out[:, 0:1]

            u = torch.autograd.grad(
                psi, y, torch.ones_like(psi),
                retain_graph=False, create_graph=False
            )[0]

            u_list.append(u.detach().cpu().numpy())

        return np.vstack(u_list)


class SWE_2D:
    def __init__(self, model, dataloader: DataLoader | SWEDataloader, lambda_phys: float = 1e-2, data_loss: str = "h",
                 regularization_coef: float = 1e-3, regularization_method: str | int = 1, logs: bool = False):
        self.model = model
        self.dataloader = dataloader
        self.lambda_phys = lambda_phys

        self.data_loss_selection = data_loss

        self.regularization_coef = regularization_coef
        self.regularization_method = regularization_method

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Logs related variables
        self.logs = logs
        self.log_history = self.init_logs()

    @staticmethod
    def init_logs():
        return {
            "loss": [],
            "data_loss": [],
            "physical_loss": [],
            "regularization": []
        }

    def log_step(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.log_history.keys():
                self.log_history[k].append(v)

    def get_logs(self):
        return self.log_history

    def display_logs(self, save: bool = False):
        for k, v in self.log_history.items():
            plt.plot(v)
            plt.xlabel(f"epochs")
            plt.ylabel(k)
            plt.title(f"{k} vs epochs")

            if save:
                plt.savefig(f"{k}_vs_epochs.png")

            else:
                plt.show()

    def shallow_water_residual(self, inp, pred, g: float = 9.81):
        """
        Computes the PDE residual for shallow water equations.

        :param inp: batched vector of inputs (t, x, y), must have requires_grad=True
        :param g: gravitational acceleration
        :return:res_mass, res_mom_x, res_mom_y: batched vector of residuals
        """

        h = pred[:, 0:1]
        u = pred[:, 1:2]
        v = pred[:, 2:3]

        # Note: We do not slice t, x, y for differentiation targets anymore.
        # We differentiate w.r.t 'inp' directly.

        # Helper to compute gradients w.r.t the input vector (t, x, y)
        def get_grads(outputs, inputs):
            """
            Computes gradients of outputs w.r.t inputs.
            inputs is shape (N, 3).
            Returns: (dt, dx, dy)
            """
            grads = torch.autograd.grad(
                outputs,
                inputs,
                grad_outputs=torch.ones_like(outputs),
                create_graph=True,
                retain_graph=True
            )[0]

            # Slice the gradients: Column 0 is dt, 1 is dx, 2 is dy
            d_dt = grads[:, 0:1]
            d_dx = grads[:, 1:2]
            d_dy = grads[:, 2:3]
            return d_dt, d_dx, d_dy

        # --- Conservative Variables & Fluxes ---

        # U = [h, hu, hv]^T
        U_mass = h
        U_mom_x = h * u
        U_mom_y = h * v

        # f(U) = [hu, hu^2 + 0.5gh^2, huv]^T
        F_mass = h * u
        F_mom_x = (h * u ** 2) + (0.5 * g * h ** 2)
        F_mom_y = h * u * v

        # g(U) = [hv, huv, hv^2 + 0.5gh^2]^T
        G_mass = h * v
        G_mom_x = h * u * v
        G_mom_y = (h * v ** 2) + (0.5 * g * h ** 2)

        # --- Compute Derivatives ---

        # For U terms, we need time derivative (d/dt is index 0 of get_grads)
        dt_U_mass, _, _ = get_grads(U_mass, inp)
        dt_U_mom_x, _, _ = get_grads(U_mom_x, inp)
        dt_U_mom_y, _, _ = get_grads(U_mom_y, inp)

        # For F terms, we need x derivative (d/dx is index 1 of get_grads)
        _, dx_F_mass, _ = get_grads(F_mass, inp)
        _, dx_F_mom_x, _ = get_grads(F_mom_x, inp)
        _, dx_F_mom_y, _ = get_grads(F_mom_y, inp)

        # For G terms, we need y derivative (d/dy is index 2 of get_grads)
        _, _, dy_G_mass = get_grads(G_mass, inp)
        _, _, dy_G_mom_x = get_grads(G_mom_x, inp)
        _, _, dy_G_mom_y = get_grads(G_mom_y, inp)

        # --- Assemble Residuals ---

        # Mass: dt(h) + dx(hu) + dy(hv) = 0
        res_mass = dt_U_mass + dx_F_mass + dy_G_mass

        # X-Momentum: dt(hu) + dx(hu^2 + ...) + dy(huv) = 0
        res_mom_x = dt_U_mom_x + dx_F_mom_x + dy_G_mom_x

        # Y-Momentum: dt(hv) + dx(huv) + dy(hv^2 + ...) = 0
        res_mom_y = dt_U_mom_y + dx_F_mom_y + dy_G_mom_y

        return res_mass, res_mom_x, res_mom_y

    def regularization_loss(self):
        return sum(
            torch.linalg.norm(p.flatten(), ord=self.regularization_method)
            for p in self.model.parameters()
        )

    def compute_loss(self, inp, pred, true_values, g: float = 9.81):
        res_mass, res_mom_x, res_mom_y = self.shallow_water_residual(inp, pred, g)

        # Data loss
        if self.data_loss_selection == "all":
            data_loss = torch.nn.MSELoss()(pred, true_values)

        elif self.data_loss_selection == "h":
            data_loss = torch.nn.L1Loss()(pred[:, 0:1], true_values[:, 0:1])

        else:
            raise ValueError("Invalid data_loss_selection! The loss selection must be either 'all' or 'h'.")

        # Physical Loss using the conservative form of the shallow water equations
        phys_loss = (res_mass ** 2 + res_mom_x ** 2 + res_mom_y ** 2).float().mean()

        # Regularization term
        reg = self.regularization_coef * self.regularization_loss()

        return data_loss + self.lambda_phys * phys_loss + reg, data_loss, phys_loss, reg

    def train_adams(self, dataloader, num_epochs):
        chunk_size = dataloader.num_points
        self.model.to(self.device)
        optimizer_adam = optim.Adam(self.model.parameters(), lr=1e-4)

        print(f"--- Starting Phase 1: Adam ({num_epochs} epochs) ---")
        for epoch in range(num_epochs):
            self.model.train()
            total_loss_epoch = 0
            num_steps = 0

            # Load the massive simulation files
            batch_inputs, batch_data = dataloader.__iter__().__next__()

            # 1. Prepare Inputs and Targets
            # Flatten everything into one massive list of points
            flat_inputs = batch_inputs.reshape(-1, 3).to(self.device)
            flat_targets = batch_data.reshape(-1, 3).to(self.device)

            total_points = flat_inputs.shape[0]

            # 2. SHUFFLE THE DATA
            perm = torch.randperm(total_points)
            flat_inputs = flat_inputs[perm]
            flat_targets = flat_targets[perm]

            # 3. CHUNK LOOP (Mini-batching)
            # Iterate through the 4.8 million points in steps of 10,000
            pbar = tqdm(range(0, total_points, chunk_size), desc=f"Ep {epoch + 1}")

            for i in pbar:
                # Extract the chunk
                end_idx = min(i + chunk_size, total_points)

                # Enable gradients ONLY for the current chunk inputs
                x_chunk = flat_inputs[i:end_idx].clone().detach().requires_grad_(True)
                y_chunk = flat_targets[i:end_idx]

                # --- Adam Step ---
                optimizer_adam.zero_grad()

                pred = self.model(x_chunk)

                loss, data_loss, phys_loss, reg_loss = self.compute_loss(x_chunk, pred, y_chunk)

                self.log_step(loss=loss.item(), data_loss=data_loss.item(), physical_loss=phys_loss.item(),
                              regularization=reg_loss.item())

                loss.backward()
                optimizer_adam.step()

                # Logging
                total_loss_epoch += loss.item()
                num_steps += 1
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})


    def train_LBFGS(self, dataloader, num_epochs):
        chunk_size = dataloader.num_points
        self.model.to(self.device)


        print(f"\n--- Starting Phase 2: L-BFGS ({num_epochs} epochs) ---")
        # L-BFGS is memory intensive. We apply it per chunk.
        optimizer_lbfgs = optim.LBFGS(self.model.parameters(),
                                      lr=0.1,
                                      max_iter=20,
                                      history_size=50,
                                      line_search_fn="strong_wolfe")

        for epoch in range(num_epochs):
            self.model.train()

            batch_inputs, batch_data = dataloader.__iter__().__next__()

            # Prepare Data (Same as Adam)
            flat_inputs = batch_inputs.reshape(-1, 3).to(self.device)
            flat_targets = batch_data.reshape(-1, 3).to(self.device)

            # Shuffle
            perm = torch.randperm(flat_inputs.shape[0])
            flat_inputs = flat_inputs[perm]
            flat_targets = flat_targets[perm]

            # Chunk Loop
            pbar = tqdm(range(0, flat_inputs.shape[0], chunk_size), desc=f"L-BFGS Ep {epoch + 1}")

            for i in pbar:
                end_idx = min(i + chunk_size, flat_inputs.shape[0])

                # Prepare chunk
                x_chunk = flat_inputs[i:end_idx].clone().detach().requires_grad_(True)
                y_chunk = flat_targets[i:end_idx]

                # --- L-BFGS Closure ---
                def closure():
                    optimizer_lbfgs.zero_grad()
                    pred = self.model(x_chunk)
                    loss, data_loss, phys_loss, reg_loss = self.compute_loss(x_chunk, pred, y_chunk)
                    self.log_step(loss=loss.item(), data_loss=data_loss.item(), physical_loss=phys_loss.item(),
                                  regularization=reg_loss.item())

                    if torch.isnan(loss):
                        print("NaN detected in closure! Returning infinite loss.")
                        return torch.tensor(float('inf'), requires_grad=True)

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    return loss

                # Step
                loss_val = optimizer_lbfgs.step(closure)

                # Logging
                pbar.set_postfix({'Loss': f"{loss_val:.4f}"})


    def train(self, dataloader, adam_epochs=5, lbfgs_epochs=2, verbose: bool = True):
        self.train_adams(dataloader, adam_epochs)
        acc1 = self.get_score(dataloader)

        if verbose:
            print(f"Accuracy for Adams: {acc1}")

        self.train_LBFGS(dataloader, lbfgs_epochs)
        acc2 = self.get_score(dataloader)

        if verbose:
            print(f"Accuracy for L-BFGS: {acc2}")

        return acc1, acc2


    def get_score(self, dataloader):
        self.model.to(self.device)
        self.model.eval()
        chunk_size = dataloader.num_points

        batch_inputs, batch_data = dataloader.get_data()  # dataloader.__iter__().__next__()

        ls_acc = []

        flat_inputs = batch_inputs.reshape(-1, 3).to(self.device)
        flat_targets = batch_data.reshape(-1, 3).to(self.device)

        # Shuffle
        perm = torch.randperm(flat_inputs.shape[0])
        flat_inputs = flat_inputs[perm]
        flat_targets = flat_targets[perm]

        # Chunk Loop
        pbar = tqdm(range(0, flat_inputs.shape[0], chunk_size), desc=f"Prediction")

        for i in pbar:
            end_idx = min(i + chunk_size, flat_inputs.shape[0])

            # Prepare chunk
            x_chunk = flat_inputs[i:end_idx].clone().detach().requires_grad_(True)
            y_chunk = flat_targets[i:end_idx]

            # Logging
            with torch.no_grad():
                pred = self.model(x_chunk)

                # ls_acc.append(torch.sqrt(((pred - y_chunk) ** 2).mean(axis=0))[0].item())
                ls_acc.append(torch.nn.functional.mse_loss(pred, y_chunk).item())

        return np.mean(ls_acc)

    def simulate(self):
        self.model.to(self.device)
        self.model.eval()
        chunk_size = 10000

        grid_tensor = self.dataloader.get_grid()
        total_points = grid_tensor.shape[0]

        # 2. List to store the output of every chunk
        all_preds = []

        # 3. Iterate through the grid
        # We do NOT shuffle here because we want to reconstruct the image later
        pbar = tqdm(range(0, total_points, chunk_size), desc="Simulating")

        for i in pbar:
            end_idx = min(i + chunk_size, total_points)

            # Get chunk and move to GPU
            # We do NOT need requires_grad=True for inference (saves memory)
            x_chunk = grid_tensor[i:end_idx].to(self.device)

            with torch.no_grad():
                pred = self.model(x_chunk)
                # Move result back to CPU and numpy immediately to save GPU RAM
                all_preds.append(pred.cpu().numpy())

        # 4. Concatenate all chunks
        # Shape: (Total_Points, 3) -> [h, u, v]
        full_prediction = np.concatenate(all_preds, axis=0)

        # 5. Reshape to (Nt, Nx, Ny, 3)
        # We retrieve dimensions from the dataloader to reshape correctly
        nt, nx, ny = self.dataloader.get_grid_shape()

        # The reshape works because we used meshgrid with indexing='ij' and flatten() earlier
        reshaped_prediction = full_prediction.reshape(nt, nx, ny, 3)

        return reshaped_prediction


def comparison_all_models(dataset_path):
    dataloader = SWEDataloader(dataset_path, batch_size=50)

    with open("config_comparison.json", "r") as f:
        data = json.load(f)

    ls_model_names = data.keys()

    results = {}

    for name in ls_model_names:
        if name == "Classic MLP":
            model = PINN_MLP

        elif name == "MLP with FF":
            model = FF_PINN

        elif name == "Mask" or "Mask + FF":
            model = MaskPINN

        else:
            raise ValueError(f"Unknown model specified: {name}")

        print(f"Running {name}")

        model = model(**data[name])
        swe_model = SWE_2D(model, dataloader, regularization_coef=1e-5, logs=True)  # 1e-5

        acc1, acc2 = swe_model.train(dataloader, adam_epochs=200, lbfgs_epochs=50)

        results[name] = swe_model.get_logs()
        results[name]["acc1"] = acc1
        results[name]["acc2"] = acc2

        del swe_model
        del model
        torch.cuda.empty_cache()

    with open("results.json", "w") as f:
        json.dump(results, f)


def comparison_new_approach(dataset_path):

    dataloader = SWEDataloader(dataset_path, batch_size=50)

    results = {}

    for i in [1, 3, 5]:
        model = AdaptedTRMPINN(3, 3, 64, 3, sigma=0.5, activation="tanh", num_latent_refinements=3,
                               num_refinement_blocks=i)
        swe_model = SWE_2D(model, dataloader, regularization_coef=1e-5, logs=True)  # 1e-5

        acc1, acc2 = swe_model.train(dataloader, adam_epochs=200, lbfgs_epochs=0)

        results[f"approach_n3_T{i}"] = swe_model.get_logs()
        results[f"approach_n3_T{i}"]["acc1"] = acc1
        results[f"approach_n3_T{i}"]["acc2"] = acc2

        del model
        del swe_model
        torch.cuda.empty_cache()

    with open("result_proposed_approach.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    comparison_new_approach(dataset_path)
    comparison_all_models(dataset_path)

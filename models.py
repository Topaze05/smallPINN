import torch
from torch import nn


def activate(activation_name, a):
    if activation_name == "tanh":
        return torch.tanh(a)
    elif activation_name == "relu":
        return torch.relu(a)
    elif activation_name == "gelu":
        return nn.functional.gelu(a)
    elif activation_name == "sine":
        return torch.sin(a)
    elif activation_name == "sigmoid":
        return torch.sigmoid(a)
    else:
        raise ValueError(f"Activation function {activation_name} not supported.")


class PINN_MLP(nn.Module):
    def __init__(self, layers: list, activation: str = "tanh"):
        super().__init__()
        self.activation_name = activation
        self.layers = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        a = x
        for i in range(len(self.layers) - 1):
            a = activate(self.activation_name, self.layers[i](a))
        return self.layers[-1](a)


class FF_PINN(nn.Module):
    def __init__(self, input_dim: int, layers: list, activation: str = "tanh", sigma: float = 1.0):
        super().__init__()
        self.activation_name = activation
        self.layers = nn.ModuleList()

        # B is sampled from a Gaussian and fixed (not trainable)
        # We use hidden_features // 2 because we generate sin and cos for each
        self.B = torch.randn(layers[0] // 2, input_dim) * sigma
        self.B = nn.Parameter(self.B, requires_grad=False)

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        proj = 2 * torch.pi * x @ self.B.T

        a = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        for i in range(len(self.layers) - 1):
            a = activate(self.activation_name, self.layers[i](a))
        return self.layers[-1](a)


class TRM_PINN(nn.Module):
    def __init__(self, dim: int, inner_model: nn.Module, num_refinement_blocks: int = 3, num_latent_refinements: int = 6):
        super().__init__()
        assert num_refinement_blocks > 1 and num_latent_refinements > 1

        self.output_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)
        self.latent_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)

        self.network = inner_model

        self.num_refinement_blocks = num_refinement_blocks
        self.num_latent_refinements = num_latent_refinements

    def refine_latent_once(self, inputs, previous_outputs, latents):
        for _ in range(self.num_latent_refinements):
            latents = self.network(inputs + previous_outputs + latents)

        outputs = self.network(previous_outputs + latents)

        return outputs, latents

    def deep_refinement(self, inputs, previous_outputs, latents):
        for step in range(1, self.num_refinement_blocks):
            with torch.no_grad():
                previous_outputs, latents = self.refine_latent_once(inputs, previous_outputs, latents)

        outputs, latents = self.refine_latent_once(inputs, previous_outputs, latents)

        return outputs, latents

    def forward(self, inputs, previous_outputs = None, latents = None):
        if previous_outputs is None:
            previous_outputs = self.output_init_embed

        if latents is None:
            latents = self.latent_init_embed

        outputs, latents = self.deep_refinement(inputs, previous_outputs, latents)

        return outputs


class MaskLayer(nn.Module):
    """
    A single hidden layer for Mask-PINN that applies:
    1. Linear transformation: z = Wx + b
    2. Mask computation: m = 1 - exp(-alpha * z^2)
    3. Activation with mask: h = activation(z) * m
    """

    def __init__(self, in_features, out_features, activation_name: str = "tanh"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation_name = activation_name

        # alpha is a learnable vector of size equal to the number of output neurons (width).
        # "alpha is initialized with all entries set to 1.0" (Section 4.1)
        self.alpha = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        # Pre-activation z
        z = self.linear(x)

        # Compute the mask function: F(z) = 1 - exp(-alpha * z^2)
        # Note: z^2 is element-wise square.
        mask = 1.0 - torch.exp(-torch.pow(self.alpha * z, 2))

        # Apply activation and modulate with the mask
        return activate(self.activation_name, z) * mask


class MaskBlock(nn.Module):
    def __init__(self, hidden_dim, activation: str = "tanh"):
        super().__init__()
        self.layer1 = MaskLayer(hidden_dim, hidden_dim, activation)
        self.layer2 = MaskLayer(hidden_dim, hidden_dim, activation)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)

        # The summation symbol (+) in the gray box
        return residual + out


class MaskPINN(nn.Module):
    """
    Full architecture based on Figure 2.
    Inputs -> Embedding -> N x MaskBlocks -> Output
    """

    def __init__(self, input_dim, output_dim, hidden_dim, num_blocks, activation: str = "tanh", sigma: float = 1.0, FourierFeatures: bool = True):
        super().__init__()

        self.FourierFeatures = FourierFeatures

        # 1. Embedding Layer (Blue box)
        # Projects input coords to the hidden feature space
        # Instead of a classical projection using a dense layer, use Fourier Features
        if FourierFeatures:
            self.B = torch.randn(hidden_dim // 2, input_dim) * sigma
            self.B = nn.Parameter(self.B, requires_grad=False)

        else:
            self.embedding = nn.Linear(input_dim, hidden_dim)


        # 2. Stack of Residual Mask Blocks (Gray box repeated N times)
        self.blocks = nn.ModuleList([
            MaskBlock(hidden_dim, activation) for _ in range(num_blocks)
        ])

        # 3. Output Layer (Red box on the right)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Optional: Initialize activation (usually Tanh for PINNs)
        self.activation = activation

    def forward(self, x):
        # Embedding
        if self.FourierFeatures:
            proj = 2 * torch.pi * x @ self.B.T
            x = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

        else:
            x = self.embedding(x)

        # Pass through N residual blocks
        for block in self.blocks:
            x = block(x)

        # Final output projection
        return self.output_layer(x)


class AdaptedTRMPINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_blocks: int = 1, num_refinement_blocks: int = 3, num_latent_refinements: int = 6, sigma: float = 1.0, activation: str = "tanh"):
        super().__init__()

        assert num_refinement_blocks >= 1 and num_latent_refinements >= 1 and num_blocks >= 1

        self.num_refinement_blocks = num_refinement_blocks
        self.num_latent_refinements = num_latent_refinements

        self.output_init_embed = nn.Parameter(torch.randn(hidden_dim) * 1e-2)
        self.latent_init_embed = nn.Parameter(torch.randn(hidden_dim) * 1e-2)

        self.B = torch.randn(hidden_dim // 2, input_dim) * sigma
        self.B = nn.Parameter(self.B, requires_grad=False)

        self.blocks = nn.ModuleList([
            MaskLayer(hidden_dim, hidden_dim, activation) for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.activation_name = activation

    def inner_forward(self, inputs, latents):
        for block in self.blocks:
            latents = block(inputs + latents)

        return latents

    def refine_latent_once(self, inputs, latents, previous_outputs = None):
        for _ in range(self.num_latent_refinements):
            latents = self.inner_forward(inputs, latents)

        outputs = self.output_layer(latents)

        return outputs, latents

    def deep_refinement(self, inputs, latents, previous_outputs = None):
        for step in range(1, self.num_refinement_blocks):
            with torch.no_grad():
                previous_outputs, latents = self.refine_latent_once(inputs, latents, previous_outputs)

        outputs, latents = self.refine_latent_once(inputs, latents, previous_outputs)

        return outputs, latents

    def forward(self, inputs, previous_outputs = None, latents = None):
        if previous_outputs is None:
            previous_outputs = self.output_init_embed

        if latents is None:
            latents = self.latent_init_embed

        proj = 2 * torch.pi * inputs @ self.B.T
        inputs = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

        outputs, latents = self.deep_refinement(inputs, previous_outputs, latents)

        return outputs




if __name__ == "__main__":
    l = MaskLayer(3, 64, "tanh")
    print(l)
    print(l._parameters)

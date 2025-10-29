# import torch


# class RoPENd(torch.nn.Module):
#     """N-dimensional Rotary Positional Embedding."""
#     def __init__(self, shape, base=10000, padding = False):
#         super(RoPENd, self).__init__()

#         channel_dims, feature_dim = shape[:-1], shape[-1]
#         k_max = feature_dim // (2 * len(channel_dims))
#         if not padding:
#             assert feature_dim % k_max == 0, f'shape[-1] ({feature_dim}) is not divisible by 2 * len(shape[:-1]) ({2 * len(channel_dims)})'

#         # tensor of angles to use
#         theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))

#         # create a stack of angles multiplied by position
#         angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
#                             torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)
#         # we apply padding here to match the feature dimension
#         angles = torch.cat([angles, torch.zeros(*angles.shape[:-1], feature_dim // 2 - angles.shape[-1])], dim=-1)
#         # convert to complex number to allow easy rotation
#         rotations = torch.polar(torch.ones_like(angles), angles)

#         # store in a buffer so it can be saved in model parameters
#         self.register_buffer('rotations', rotations)

#     def forward(self, x):
#         # convert input into complex numbers to perform rotation
#         x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
#         pe_x = self.rotations * x
#         return torch.view_as_real(pe_x).flatten(-2)
    

import torch

class RoPENd(torch.nn.Module):
    """N-dimensional Rotary Positional Embedding (Real-valued computation)."""
    def __init__(self, shape, base=10000, padding=False):
        super(RoPENd, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))
        
        # RoPE applies to pairs of features (x_0, x_1), (x_2, x_3), etc.
        # feature_dim must be even to form pairs, and k_max * 2 must be <= feature_dim
        if not padding:
            assert feature_dim % 2 == 0, f'shape[-1] ({feature_dim}) must be even.'
            assert k_max * 2 * len(channel_dims) <= feature_dim, \
                f'feature_dim ({feature_dim}) is too small for the given number of spatial dimensions ({len(channel_dims)}).'

        # tensor of angles (theta_k) to use
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))

        # 1. Create a stack of angles multiplied by position (pos_i * theta_k)
        # Angles for each spatial dimension (pos_0, pos_1, ...) are concatenated
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)
        
        # 2. Pad to match feature_dim // 2
        # The rotation is applied to feature_dim // 2 pairs of elements.
        num_pairs = feature_dim // 2
        angles = torch.cat([angles, torch.zeros(*angles.shape[:-1], num_pairs - angles.shape[-1])], dim=-1)

        # 3. Calculate cos and sin components for the rotation matrix
        # rotation_cos.shape = (*channel_dims, feature_dim // 2)
        # rotation_sin.shape = (*channel_dims, feature_dim // 2)
        rotation_cos = torch.cos(angles)
        rotation_sin = torch.sin(angles)

        # Store in buffers
        self.register_buffer('rotation_cos', rotation_cos.bfloat16())
        self.register_buffer('rotation_sin', rotation_sin.bfloat16())

    def forward(self, x):
        # x.shape is (*channel_dims, feature_dim)

        # 1. Reshape x into pairs: (*channel_dims, feature_dim // 2, 2)
        # The pairs are (x_0, x_1), (x_2, x_3), etc.
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_0 = x_reshaped[..., 0]  # First element of each pair: (*channel_dims, feature_dim // 2)
        x_1 = x_reshaped[..., 1]  # Second element of each pair: (*channel_dims, feature_dim // 2)

        # 2. Apply the real-valued 2D rotation:
        # R_theta * (x_0, x_1)^T = (x_0*cos - x_1*sin, x_0*sin + x_1*cos)^T
        
        # The stored buffers are broadcasted over the batch dimensions (channel_dims).
        # pe_x_0 = x_0 * cos(theta) - x_1 * sin(theta)
        pe_x_0 = x_0 * self.rotation_cos - x_1 * self.rotation_sin
        # pe_x_1 = x_0 * sin(theta) + x_1 * cos(theta)
        pe_x_1 = x_0 * self.rotation_sin + x_1 * self.rotation_cos

        # 3. Combine the rotated elements and flatten back to the original feature dimension
        # Stack the results along a new dimension: (*channel_dims, feature_dim // 2, 2)
        pe_x_stacked = torch.stack([pe_x_0, pe_x_1], dim=-1)
        
        # Flatten the last two dimensions: (*channel_dims, feature_dim)
        return pe_x_stacked.flatten(-2)
import torch


def generate_field(f, xi, H, W, el, roll, device):
    u0 = W / 2.0
    v0 = H / 2.0

    grid_x, grid_y = torch.meshgrid(
        torch.arange(0, W).float().to(device),
        torch.arange(0, H).float().to(device),
    )

    X_Cam = (grid_x - u0) / f
    Y_Cam = -(grid_y - v0) / f

    # 2. Projection on the sphere

    AuxVal = X_Cam * X_Cam + Y_Cam * Y_Cam

    # alpha_cam = np.real(xi + torch.sqrt(1 + (1 - xi*xi)*AuxVal))
    alpha_cam = xi + torch.sqrt(1 + (1 - xi * xi) * AuxVal)

    alpha_div = AuxVal + 1

    alpha_cam_div = alpha_cam / alpha_div

    X_Sph = X_Cam * alpha_cam_div
    Y_Sph = Y_Cam * alpha_cam_div
    Z_Sph = alpha_cam_div - xi

    # rot_el
    cosel, sinel = torch.cos(el), torch.sin(el)
    Y_Sph = Y_Sph * cosel - Z_Sph * sinel
    Z_Sph = Y_Sph * sinel + Z_Sph * cosel

    # rot_roll
    cosroll, sinroll = torch.cos(roll), torch.sin(roll)
    X_Sph = X_Sph * cosroll - Y_Sph * sinroll
    Y_Sph = X_Sph * sinroll + Y_Sph * cosroll

    # sph = rot_az.dot(sph)

    # sph = sph.reshape((3, H, W))#.transpose((1,2,0))
    coords = torch.stack((X_Sph.reshape(-1), Y_Sph.reshape(-1), Z_Sph.reshape(-1)))
    coords = coords / torch.sqrt(torch.sum(coords**2, dim=0))
    return coords

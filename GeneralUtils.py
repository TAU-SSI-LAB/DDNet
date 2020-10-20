from Imports import *


def fix_state_dict(state_dict):

    assert isinstance(state_dict, OrderedDict)

    if list(state_dict.keys())[0].split('.')[0] == 'module':
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    return new_state_dict


def tensor_to_image(input_image):

    if isinstance(input_image, torch.Tensor):
        if input_image.device != 'cpu':
            input_image = input_image.cpu()
        input_image = input_image.numpy().squeeze()

    assert isinstance(input_image, np.ndarray)

    input_image = input_image.transpose((1, 2, 0))
    if input_image.shape[-1] == 1:
        input_image = np.squeeze(input_image, axis=-1)

    return input_image


def hs_to_rgb(cube, wave_lengths_range=None, gamma=False):

    # These values correspond to 400-720 nm (33 spectral bands with 10nm bandwidth)
    x = np.array([0.0143, 0.0435, 0.1344, 0.2839, 0.3483, 0.3362, 0.2908, 0.1954, 0.0956, 0.032, 0.0049, 0.0093, 0.0633, 0.1655,
                  0.2904, 0.4335, 0.5945, 0.7621, 0.9163, 1.0263, 1.0622, 1.0026, 0.8545, 0.6424, 0.4479, 0.2835,
                  0.1649, 0.0874, 0.0468, 0.0227, 0.0114, 0.0058, 0.0029])

    y = np.array([0.000396, 0.0012, 0.004, 0.0116, 0.023, 0.038, 0.06, 0.091, 0.139, 0.208, 0.323, 0.503, 0.71, 0.862, 0.954, 0.995,
                  0.995, 0.952, 0.87, 0.757, 0.631, 0.503, 0.381, 0.265, 0.175, 0.107, 0.061, 0.032, 0.017, 0.0082,
                  0.0041, 0.0021, 0.0010])
    z = np.array([0.0679, 0.2074, 0.6456, 1.3856, 1.7471, 1.7721, 1.6692, 1.2876, 0.8130, 0.4652, 0.272, 0.1582, 0.0783, 0.0422,
                  0.0203, 0.0088, 0.0039, 0.0021, 0.0017, 0.0011, 0.0008, 0.00034, 0.00019, 0.00005, 0.00002, 0.,
                  0., 0., 0., 0., 0., 0., 0.])

    min_index = (max(400, min(wave_lengths_range)) - 400) // 10
    max_index = (min(max(wave_lengths_range), 720) - 400) // 10
    spectral_range = range(min_index, max_index + 1)

    x = x[spectral_range]
    y = y[spectral_range]
    z = z[spectral_range]

    if 400 in wave_lengths_range:
        min_cube_index = np.where(wave_lengths_range == 400)[0][0]
    else:
        min_cube_index = 0

    if 720 in wave_lengths_range:
        max_cube_index = np.where(wave_lengths_range == 720)[0][0]
        cube = cube[:, min_cube_index: max_cube_index + 1, :, :]
    else:
        cube = cube[:, min_cube_index:, :, :]

    if isinstance(cube, torch.Tensor):

        bs = cube.shape[0]
        x, y, z = map(lambda p: torch.from_numpy(p).type(cube.dtype).to(cube.device), [x, y, z])
        cube = cube.transpose(0, 1)
        X, Y, Z = map(lambda p: torch.mul(cube, p.unsqueeze(1).unsqueeze(2).unsqueeze(3)).sum(0), [x, y, z])
        max_val = torch.cat([X.unsqueeze(1), Y.unsqueeze(1), Z.unsqueeze(1)], dim=1).view(bs, -1).max(-1)[0]
        X, Y, Z = map(lambda p: p / max_val[:, None, None], [X, Y, Z])
        X, Y, Z = map(lambda p: torch.clamp(p, min=0), [X, Y, Z])

        R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
        G = - 0.9689 * X + 1.87582 * Y + 0.0414 * Z
        B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

        R, G, B = map(lambda p: torch.clamp(p, min=0, max=1).unsqueeze(1), [R, G, B])

        return torch.cat([R, G, B], dim=1) ** ((1 / 0.45) if gamma else 1)

    elif isinstance(cube, np.ndarray):
        X, Y, Z = map(lambda p: np.dot(cube, np.expand_dims(p, -1)), [x, y, z])
        max_val = np.max(np.concatenate([X, Y, Z], axis=-1))
        X, Y, Z = map(lambda p: p / max_val, [X, Y, Z])
        X, Y, Z = map(lambda p: np.clip(p, 0, None), [X, Y, Z])

        R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
        G = - 0.9689 * X + 1.87582 * Y + 0.0414 * Z
        B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

        R, G, B = map(lambda p: np.clip(p, 0, 1), [R, G, B])

        return np.concatenate([R, G, B], axis=-1) ** ((1 / 0.45) if gamma else 1)













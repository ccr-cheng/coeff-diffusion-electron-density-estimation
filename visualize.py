import io

import PIL
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torchvision.transforms import ToTensor

plt.switch_backend('agg')
cmap = ListedColormap(['grey', 'white', 'red', 'blue', 'green', 'white'])


def draw_stack(density, atom_type=None, atom_coord=None, dim=-1):
    plt.figure(figsize=(3, 3))
    plt.imshow(density.sum(dim).detach().cpu().numpy(), cmap='viridis')
    plt.colorbar()
    if atom_type is not None:
        idx = [i for i in range(3) if i != dim % 3]
        coord = atom_coord.detach().cpu().numpy()
        color = cmap(atom_type.detach().cpu().numpy())
        plt.scatter(coord[:, idx[1]], coord[:, idx[0]], c=color, alpha=0.8)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close()
    return image
import numpy as np
from scipy.stats import norm
import torch
from Vae_Pokemon_new import ConvVAE
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

state = torch.load('best_model_pokemon')

latent_dim = 32
inter_dim = 128
mid_dim = (128, 6, 6)
mid_num = 4608

model = ConvVAE(latent_dim, mid_num, inter_dim, mid_dim)
model.load_state_dict(state)

transform = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    transforms.CenterCrop(20),
    transforms.ToTensor(),
])

n = 10
image_size = 20

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

model.eval()

image_path = './pokemon/025MS.png'
with torch.no_grad():
    base = transform(image_path).unsqueeze(0)
    x = model.encoder(base)
    x = x.view(1, -1)
    x = model.fc1(x)
    h = model.fc2(x)
    mu, logvar = h.chunk(2, dim=-1)
    z = model.reparameterise(mu, logvar)
    z = z.squeeze(0)

selected = 0
latent_dim = 5
coll = [(selected, i) for i in range(latent_dim) if i != selected]

for idx, (p, q) in enumerate(coll):
    figure = np.zeros((3, image_size * n, image_size * n))
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z[p], z[q] = xi, yi
            z_sampled = torch.FloatTensor(z).unsqueeze(0)
            with torch.no_grad():
                decode = model.fcr1(model.fcr2(z_sampled))
                decode = decode.view(1, *mid_dim)
                decode = model.decoder(decode)
                decode = decode.squeeze(0)

                figure[:,
                i * image_size: (i + 1) * image_size,
                j * image_size: (j + 1) * image_size
                ] = decode

    plt.title("X: {}, Y: {}".format(p, q))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.imshow(figure.transpose(1, 2, 0))
    plt.show()

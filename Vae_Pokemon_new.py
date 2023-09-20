import os

import random

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class Pokemon(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(Pokemon, self).__init__()
        self.root = root
        self.image_path = [os.path.join(root, x) for x in os.listdir(root)]
        random.shuffle(self.image_path)

        if transform is not None:
            self.transform = transform

        if train:
            self.images = self.image_path[: int(.8 * len(self.image_path))]
        else:
            self.images = self.image_path[int(.8 * len(self.image_path)):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.transform(self.images[item])


class ConvVAE(nn.Module):
    def __init__(self,
                 latent,
                 mid_num,
                 inter_dim,
                 mid_dim):
        super(ConvVAE, self).__init__()
        self.latent = latent
        self.mid_num = mid_num
        self.inter_dim = inter_dim
        self.mid_dim = mid_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.2),

            nn.Conv2d(128, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.2),
        )

        self.fc1 = nn.Linear(self.mid_num, self.inter_dim)
        self.fc2 = nn.Linear(self.inter_dim, self.latent * 2)

        self.fcr2 = nn.Linear(self.latent, self.inter_dim)
        self.fcr1 = nn.Linear(self.inter_dim, self.mid_num)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(128, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(64, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(32, 16, 4),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(.2),

            nn.Conv2d(16, 3, 3),
            nn.Sigmoid()
        )

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        batch = x.size(0)
        x = self.encoder(x)
        # print(x.shape)
        x = self.fc1(x.view(batch, -1))
        h = self.fc2(x)

        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterise(mu, logvar)

        decode = self.fcr2(z)
        decode = self.fcr1(decode)
        recon_x = self.decoder(decode.view(batch, *self.mid_dim))

        return recon_x, mu, logvar

def main():
    latent_dim = 32
    inter_dim = 128
    mid_dim = (128, 6, 6)
    mid_num = 1
    for i in mid_dim:
        mid_num *= i

    kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    recon_loss = lambda recon_x, x: F.mse_loss(recon_x, x, reduction='sum')

    epochs = 1000
    batch_size = 512

    best_loss = 1e9
    best_epoch = 0

    valid_losses = []
    train_losses = []

    transform = transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.CenterCrop(20),
        transforms.ToTensor(),
    ])

    pokemon_train = Pokemon('./Pokemon/', train=True, transform=transform)
    pokemon_valid = Pokemon('./Pokemon/', train=False, transform=transform)

    train_loader = DataLoader(pokemon_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(pokemon_valid, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConvVAE(latent_dim, mid_num, inter_dim, mid_dim)

    if os.path.exists("./best_model_pokemon"):
        state = torch.load('best_model_pokemon')
        model.load_state_dict(state)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.
        train_num = len(train_loader.dataset)

        for idx, x in enumerate(train_loader):
            batch = x.size(0)
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            recon = recon_loss(recon_x, x)
            kl = kl_loss(mu, logvar)

            loss = recon + kl
            train_loss += loss.item()
            loss = loss / batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f"Training loss {loss: .3f} \t Recon {recon / batch: .3f} \t KL {kl / batch: .3f} in Step {idx}")

        train_losses.append(train_loss / train_num)

        valid_loss = 0.
        valid_recon = 0.
        valid_kl = 0.
        valid_num = len(test_loader.dataset)
        model.eval()
        with torch.no_grad():
            for idx, x in enumerate(test_loader):
                x = x.to(device)
                recon_x, mu, logvar = model(x)
                recon = recon_loss(recon_x, x)
                kl = kl_loss(mu, logvar)
                loss = recon + kl
                valid_loss += loss.item()
                valid_kl += kl.item()
                valid_recon += recon.item()

            valid_losses.append(valid_loss / valid_num)

            print(
                f"Valid loss {valid_loss / valid_num: .3f} \t Recon {valid_recon / valid_num: .3f} \t KL {valid_kl / valid_num: .3f} in epoch {epoch}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch

                torch.save(model.state_dict(), 'best_model_pokemon')
                print("Model saved")


if __name__ == '__main__':
    main()

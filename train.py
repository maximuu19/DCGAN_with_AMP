import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms_module
from torch.utils.data import DataLoader
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
from init_weights import init_weights


def train_fn(gen, disc, dataloader, opt_gen, opt_disc, NUM_EPOCHS, device, BATCH_SIZE, Z_DIM, criterion,  g_scaler, d_scaler):
    gen.train()
    disc.train()
    loop = tqdm(dataloader, leave=True)
    for epoch in range(NUM_EPOCHS):
        for idx, (real, _) in enumerate(loop):
            real = real.to(device)
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake_img = gen(noise)

            with torch.cuda.amp.autocast():
                ## Train Discriminator
                disc_ri = disc(real).reshape(-1)
                loss_disc_ri = criterion(disc_ri, torch.ones_like(disc_ri))
                disc_fi = disc(fake_img.detach()).reshape(-1)
                loss_disc_fi = criterion(disc_fi, torch.zeros_like(disc_fi))
                loss_disc = (loss_disc_fi + loss_disc_ri) / 2

            disc.zero_grad()
            d_scaler.scale(loss_disc).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            with torch.cuda.amp.autocast():
                output = disc(fake_img).reshape(-1)
                loss_gen = criterion(output, torch.ones_like(output))

            gen.zero_grad()
            g_scaler.scale(loss_gen).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            print(f"Epoch {epoch}/{NUM_EPOCHS} loss gen {loss_gen.item()}, loss disc {loss_disc.item()} \n\n")

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 128
    IMAGE_SIZE = 64
    CHANNELS_IMG = 1
    Z_DIM = 100
    NUM_EPOCHS = 1
    FEATURES_DISC = 64
    FEATURES_GEN = 64
    transforms = transforms_module.Compose([
        transforms_module.Resize(IMAGE_SIZE),
        transforms_module.ToTensor(),
        transforms_module.Normalize(
            [0.5] * int(CHANNELS_IMG), [0.5] * int(CHANNELS_IMG)
        ),
    ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    gen = Generator(Z_DIM, img_channels=CHANNELS_IMG, features_g=FEATURES_GEN).to(device)
    disc = Discriminator(features_d=FEATURES_DISC, img_channels=CHANNELS_IMG).to(device)
    init_weights(gen)
    init_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    train_fn(gen, disc, dataloader, opt_gen, opt_disc, NUM_EPOCHS, device, BATCH_SIZE, Z_DIM, criterion, g_scaler, d_scaler)

if __name__ == "__main__":
    main()

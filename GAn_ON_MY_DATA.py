import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Hyper Parameters

latent_dim = 100
lr = 0.0001
image_size = 28
channels = 1
batch_size = 60
epochs = 100

# First we define the our normalization 
# This transformation actually transform our Mnist data to array
#This also hepls us in resizing
# whenever we have images and we have to convert them to arrays also done the normalization we use this way
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(), # This will a dense array
    transforms.Normalize([0.5],[0.5]) # This will convert data to -1,1
])

# Load MNIST data using DataLoader

dataset = datasets.ImageFolder(root='E:/Generative AI and LLMs/data', transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Takes Random Noise and generate Images

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,image_size*image_size*channels),
            nn.Tanh()  # Output between -1 and 1
        )
    def forward(self,z):
        img = self.model(z)
        img = img.view(img.size(0),channels,image_size,image_size)
        return img
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size*image_size*channels,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid() # Output a probablity real/fake

        )
    def forward(self,img):
        img_flat = img.view(img.size(0),-1) # This will flatten the Image means make one column
        validity = self.model(img_flat)
        return validity

# Initialize Model and Optimizer
generator = Generator()
discriminator = Discriminator()

# Loss function
adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(),lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(),lr=lr)

for epoch in range(epochs):
    for i, (imgs,_) in enumerate(dataloader):
        # Ground truths
        valid = torch.ones(imgs.size(0),1)
        fake = torch.zeros(imgs.size(0),1)

    # Train Generator
        optimizer_G.zero_grad()
        # Sample random Noise as an input to Generator
        z = torch.randn(imgs.size(0),latent_dim)
        # Generate the Image using noise
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs),valid)
    # Train Discriminator

        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(imgs),valid)
        # fake loss
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()),fake)
        # Total dicriminator loss
        d_loss = (real_loss+fake_loss)/2

        # Back propagate and Optimize
        d_loss.backward()
        optimizer_D.step()
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        # if epoch % 10 == 0:
        #     with torch.no_grad():
        #         sample_imgs = generator(torch.randn(25, latent_dim)).detach().numpy()
        #         plt.imshow(sample_imgs[0].reshape(28, 28), cmap='gray')
        #         plt.show()

         


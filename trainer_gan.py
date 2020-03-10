from comet_ml import Experiment
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5,), (.5,))
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    return n

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

def train_discriminator(optimizer, real_data, fake_data, discriminator, loss):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data, discriminator, loss):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

def plot_test_images(test_images, experiment, send_to_log):
    fig = plt.figure()
    # number_of_images = len(test_images)
    # y_range = 2
    # x_range = number_of_images // y_range + 1

    for idx in range(6):
        plt.subplot(2, 3, idx + 1)
        plt.tight_layout()
        plt.imshow(test_images[idx].squeeze(0), cmap='gray', interpolation='none')
        plt.title("GAN")
        plt.xticks([])
        plt.yticks([])

    if send_to_log:
        experiment.log_figure(figure=plt)

    fig.show()

def main():
    experiment = Experiment(api_key="1x1ZQpvbtvDyO2s5DrlUyYpzv", project_name="GAN1", workspace="verlyn-fischer")

    discriminator_path = 'models/discriminator.pth'
    generator_path = 'models/generator.pth'

    # Load data
    data = mnist_data()
    # Create loader with data, so that we can iterate over it
    data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
    # Num batches
    num_batches = len(data_loader)
    discriminator = DiscriminatorNet()
    generator = GeneratorNet()

    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    loss = nn.BCELoss()

    num_test_samples = 16
    test_noise = noise(num_test_samples)

    # Create logger instance
    # logger = Logger(model_name='VGAN', data_name='MNIST')
    # Total number of epochs to train
    num_epochs = 70
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')
        for n_batch, (real_batch, _) in enumerate(data_loader):
            N = real_batch.size(0)
            # 1. Train Discriminator
            real_data = Variable(images_to_vectors(real_batch))
            # Generate fake data and detach
            # (so gradients are not calculated for generator)
            fake_data = generator(noise(N)).detach()
            # Train D
            d_error, d_pred_real, d_pred_fake = \
                train_discriminator(d_optimizer, real_data, fake_data, discriminator, loss)

            # 2. Train Generator
            # Generate fake data
            fake_data = generator(noise(N))
            # Train G
            g_error = train_generator(g_optimizer, fake_data, discriminator, loss)
            # Log batch error
            # logger.log(d_error, g_error, epoch, n_batch, num_batches)
            # Display Progress every few batches
            experiment.log_metric("d_error", d_error, step=n_batch)
            experiment.log_metric("g_error", g_error, step=n_batch)
            # if (n_batch) % 100 == 0:
            #     # logger.log_images(
            #     #     test_images, num_test_samples,
            #     #     epoch, n_batch, num_batches
            #     # );
            #     # Display status Logs
            #     # logger.display_status(
            #     #     epoch, num_epochs, n_batch, num_batches,
            #     #     d_error, g_error, d_pred_real, d_pred_fake
            #     # )

        # Plot test images after each epoch
        test_images = vectors_to_images(generator(test_noise))
        test_images = test_images.data
        plot_test_images(test_images, experiment, False)

    # Save models and log images

    torch.save(discriminator.state_dict(), discriminator_path)
    torch.save(discriminator.state_dict(), generator_path)

    test_images = vectors_to_images(generator(test_noise))
    test_images = test_images.data
    plot_test_images(test_images, experiment, True)

main()
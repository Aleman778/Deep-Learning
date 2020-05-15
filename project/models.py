"""Generative Adversarial Network models are based on the implementation from
[DCGAN PyTorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) 
which references the [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf)."""


import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal


"""Different model implementations that can be used.
"""


def _init_weights(model):
    """From the DCGAN paper the authors specify that all model weights shall 
    be randomly initialized from a normal distribution with mean=0 and stdev=0.02.
    """
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

        
def densegan4_discriminator(e):
    """Discriminator model for GANs using 4-layer fully connected network."""
    input_size = e.params["im_size"]**2 # square image
    ndf = e.params["ndf"] # number of discriminator features
    model = nn.Sequential(
        nn.Linear(input_size, ndf),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Linear(ndf,   ndf//2),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Linear(ndf//2, ndf//4),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Linear(ndf//4, 1),
        nn.Sigmoid()
    ).to(e.device)
    return model
    

def densegan4_generator(e):
    """Generator model for GANs using 4-layer fully connected network."""
    input_size = e.params["nz"] # size of latent vector z
    output_size = e.params["im_size"]**2 # output size is the squared of the image size
    ngf = e.params["ndf"] # number of discriminator features
    model = nn.Sequential(
        nn.Linear(input_size, ngf),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),

        nn.Linear(ngf,   ngf*2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),

        nn.Linear(ngf*2, ngf*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),

        nn.Linear(ngf*4, output_size),
        nn.Tanh()
    ).to(e.device)
    return model
    

def dcgan3_discriminator(e):
    """Discriminator model for GANs using CNN. This is based on the dcgan paper
    [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf) except it uses 3 conv layers instead of 4.
    This model should work perfectly images of size 32 or smaller.

    e - the experiment instance
    nc - the number of components in the image (3 for RGB, 1 for grayscale)
    im_size - the size of the input image in pixels"""
    nc = e.params["nc"] 
    im_size = e.params["im_size"]

    model = nn.Sequential(
        # INPUT - image of dimensions (nc) * im_size * im_size
        nn.Conv2d(nc, im_size, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        # CONV1 - current dimension is: (im_size*2) * (im_size/2) * (im_size/2)
        nn.Conv2d(im_size, im_size*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(im_size*2),
        nn.LeakyReLU(0.2, inplace=True),

        # CONV2 - current dimension is: (im_size*4) * (im_size/4) * (im_size/4)
        nn.Conv2d(im_size*2, im_size*4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(im_size*4),
        nn.LeakyReLU(0.2, inplace=True),

        # CONV3 - current dimension is: (im_size*8) * (im_size/8) * (im_size/8)
        nn.Conv2d(im_size*4, 1, kernel_size=4, stride=1, padding=0, bias=False),
        nn.Sigmoid()

        # OUTPUT - single number between 0.0 (fake) and 1.0 (real)
    ).to(e.device)
    model.apply(_init_weights)
    return model


def dcgan3_generator(e):
    """Generator model for GANs using CNN. This is based on the dcgan paper
    [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf) except it uses 3 conv layers instead of 4.
    This model should generate images of size 32 or smaller.

    e - the experiment instance with params set:
    - nz - the size of the latent vector (input noise to generator)
    - nc - the number of components in the image (3 for RGB, 1 for grayscale)
    - im_size - the size of the output image in pixels"""

    nz = e.params["nz"] 
    nc = e.params["nc"] 
    im_size = e.params["im_size"]

    model = nn.Sequential(
        # INPUT - latent vector Z is the input to convolution
        nn.ConvTranspose2d(nz, im_size*4, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(im_size*4),
        nn.ReLU(inplace=True),

        # CONV1 - current dimension is: (im_size) * (im_size/8) * (im_size/8)
        nn.ConvTranspose2d(im_size*4, im_size*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(im_size*2),
        nn.ReLU(inplace=True),

        # CONV2 - current dimension is: (im_size*4) * (im_size/4) * (im_size/4)
        nn.ConvTranspose2d(im_size*2, im_size, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(im_size),
        nn.ReLU(inplace=True),

        # CONV3 - current dimension is: (im_size) * (im_size/2) * (im_size/2)
        nn.ConvTranspose2d(im_size, nc, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()

        # OUTPUT - dimension is: (nc) * im_size * im_size
    ).to(e.device)
    model.apply(_init_weights)
    return model

    

def dcgan4_discriminator(e):
    """Discriminator model for GANs using CNN. This is based on the dcgan paper
    [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf) except it uses 3 conv layers instead of 4.
    This model should work perfectly images of size 32 or smaller.

    e - the experiment instance
    nc - the number of components in the image (3 for RGB, 1 for grayscale)
    im_size - the size of the input image in pixels"""
    nc = e.params["nc"]
    im_size = e.params["im_size"]

    model = nn.Sequential(
        # INPUT - image of dimensions (nc) * im_size * im_size
        nn.Conv2d(nc, im_size, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        # CONV1 - current dimension is: (im_size*2) * (im_size/2) * (im_size/2)
        nn.Conv2d(im_size, im_size*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(im_size*2),
        nn.LeakyReLU(0.2, inplace=True),

        # CONV2 - current dimension is: (im_size*4) * (im_size/4) * (im_size/4)
        nn.Conv2d(im_size*2, im_size*4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(im_size*4),
        nn.LeakyReLU(0.2, inplace=True),

        # CONV3 - current dimension is: (im_size*4) * (im_size/8) * (im_size/8)
        nn.Conv2d(im_size*4, im_size*8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(im_size*8),
        nn.LeakyReLU(0.2, inplace=True),

        # CONV4 - current dimension is: (im_size*8) * (im_size/16) * (im_size/16)
        nn.Conv2d(im_size*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        nn.Sigmoid()

        # OUTPUT - single number between 0.0 (fake) and 1.0 (real)
    ).to(e.device)
    model.apply(_init_weights)
    return model


def dcgan4_generator(e):
    """Generator model for GANs using CNN. This is based on the dcgan paper
    [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf).
    This model should generate images of size 32 or smaller.

    e - the experiment instance with params set:
    - nz - the size of the latent vector (input noise to generator)
    - nc - the number of components in the image (3 for RGB, 1 for grayscale)
    - im_size - the size of the output image in pixels"""

    nz = e.params["nz"] 
    nc = e.params["nc"] 
    im_size = e.params["im_size"]

    model = nn.Sequential(
        # INPUT - latent vector Z is the input to convolution
        nn.ConvTranspose2d(nz, im_size*8, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(im_size*8),
        nn.ReLU(inplace=True),

        # CONV1 - current dimension is: (im_size) * (im_size/16) * (im_size/16)
        nn.ConvTranspose2d(im_size*8, im_size*4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(im_size*4),
        nn.ReLU(inplace=True),

        # CONV2 - current dimension is: (im_size*4) * (im_size/8) * (im_size/8)
        nn.ConvTranspose2d(im_size*4, im_size*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(im_size*2),
        nn.ReLU(inplace=True),

        # CONV3 - current dimension is: (im_size*4) * (im_size/4) * (im_size/4)
        nn.ConvTranspose2d(im_size*2, im_size, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(im_size),
        nn.ReLU(inplace=True),

        # CONV4 - current dimension is: (im_size) * (im_size/2) * (im_size/2)
        nn.ConvTranspose2d(im_size, nc, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()

        # OUTPUT - dimension is: (nc) * im_size * im_size
    ).to(e.device)
    model.apply(_init_weights)
    return model


class VAE_Encoder(nn.Module):
    """Encoder model for VAEs using CNN."""
    def __init__(e):
        """Creates a new VAE encoder.
        e - the exeriment instance with parameters set:
        im_size - the shape of image e.g. mnist has (28, 28, 1)
        """
        super(VAE_Encoder, self).__init__()

        nc = self.params["nc"]           # number of image componens
        nef = self.params["nef"]         # number of encoder features
        nz = self.params["nz"]           # size of latent vector
        im_size = self.params["im_size"] # size of input image

        self.conv1 = nn.Conv2D(nc,  nef//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2D(nef, nef,    kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2D(nef, nef,    kernel_size=3, padding=1)
        self.conv4 = nn.Conv2D(nef, nef,    kernel_size=3, padding=1)

        self.shape_before_flattening = im_size//2

        self.fc = nn.Linear(self.shape_before_flattening, 32)

    def forward(x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.fc(x))
        return self.sampling(x)


class VAE_Sampling_Layer(nn.Module):
    def __init__(e):
        nz = e.params["nz"]
        mean = e.params["mean"]
        stdev = e.params["stdev"]
        self.normal_dist = Normal(torch.tensor([self.mean]), torch.tensor([self.stdev]))
        self.shape = torch.tensor([32, nz])

        self.z_mean = nn.Linear(32, nz)
        self.z_log_var = nn.Linear(32, nz)
        

    def forward(inputs):
        z_mean = self.z_mean(inputs)
        z_log_var = self.z_mean(inputs)
        epsilon = self.normal_dist(sample_shape=self.shape)
        return z_mean + K.exp(z_log_var) * epsilon
        
# def vae_decoder(e):
#     
#     im_size = e.params["im_size"]

#     input_img = keras.Input(shape=img_shape)

#     model = nn.Sequential(
#         nn.Conv2D(32, 3, padding='same', )(input_img)
#         model = nn.Conv2D(64, 3, padding='same', , strides=(2, 2))(model)
#         model = nn.Conv2D(64, 3, padding='same', )(model)
#         model = nn.Conv2D(64, 3, padding='same', )(model)
#         }
#     shape_before_flattening = K.int_shape(model)

#     model = layers.Flatten()(model)
#     model = layers.Dense(32, )(model)

#     z_mean =    layers.Dense(latent_dim)(model)
#     z_log_var = layers.Dense(latent_dim)(model)
#     return model


# def vae_latent_sampling(args):
#     z_mean, z_log_var = args
#     epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
#     return z_mean + K.exp(z_log_var) * epsilon


# def vae_encoder(e):
#     z_mean =    layers.Dense(latent_dim)(x)
#     z_log_var = layers.Dense(latent_dim)(x)

#     # Latent vector
#     z = layers.Lambda(sampling)([z_mean, z_log_var])

#     # This is the input where we will feed `z`.
#     decoder_input = layers.Input(K.int_shape(z)[1:])
    
#     # Upsample to the correct number of units
#     x = layers.Dense(np.prod(shape_before_flattening[1:]), )(decoder_input)

#     # Reshape into an image of the same shape as before our last `Flatten` layer
#     x = layers.Reshape(shape_before_flattening[1:])(x)

#     # We then apply then reverse operation to the initial
#     # stack of convolution layers: a `Conv2DTranspose` layers
#     # with corresponding parameters.
#     x = layers.Conv2DTranspose(32, 3, padding='same', , strides=(2, 2))(x)
#     x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
#     # We end up with a feature map of the same size as the original input.

#     # This is our decoder model.
#     decoder = Model(decoder_input, x)

#     # We then apply it to `z` to recover the decoded `z`.
#     return decoder(z)


# class CustomVariationalLayer(keras.layers.Layer):
#     def vae_loss(self, x, z_decoded):
#         x = K.flatten(x)
#         z_decoded = K.flatten(z_decoded)
#         xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
#         kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#         return K.mean(xent_loss + kl_loss)

#     def call(self, inputs):
#         x = inputs[0]
#         z_decoded = inputs[1]
#         loss = self.vae_loss(x, z_decoded)
#         self.add_loss(loss, inputs=inputs)
#         # We don't use this output.
#         return x

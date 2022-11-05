import torch
import torch.nn.functional as F
import torch.autograd as autograd

from .layer_utils import CONV3_3_IN_VGG_19


def perceptual_loss(deblurred, sharp):
    model = CONV3_3_IN_VGG_19

    # feature map of the output and target
    deblurred_feature_map = model.forward(deblurred)
    sharp_feature_map = model.forward(sharp).detach()  # we do not need the gradient of it          # creat a new tensor detach from the current graph
    loss = F.mse_loss(deblurred_feature_map, sharp_feature_map)                                     # MSE loss: mean square error / squared L2 norm
                                                                                                    # can be arveraged or sumed by using the "reduction"
    return loss


def wgan_gp_loss(type, **kwargs):
    if type == 'G':  # generator loss                                                               # G loss: the negtive mean of the output of the D
        deblurred_discriminator_out = kwargs['deblurred_discriminator_out']                         # min the loss: make the predict to ones (larger)
        return -deblurred_discriminator_out.mean()

    elif type == 'D':  # discriminator loss                                                         # D loss: WGAN loss
        gp_lambda = kwargs['gp_lambda']  # lambda coefficient of gradient penalty term              # make the output of real larger than fake ones
        interpolates = kwargs['interpolates']  # interpolates = alpha * sharp + (1 - alpha) * deblurred
        interpolates_discriminator_out = kwargs['interpolates_discriminator_out']
        sharp_discriminator_out = kwargs['sharp_discriminator_out']
        deblurred_discriminator_out = kwargs['deblurred_discriminator_out']

        # WGAN loss
        wgan_loss = deblurred_discriminator_out.mean() - sharp_discriminator_out.mean()             # the difference of the means of all pixels
                                                                                                    # minimize this loss to make the real larger
        # gradient penalty
        gradients = autograd.grad(outputs=interpolates_discriminator_out, inputs=interpolates,
                                  grad_outputs=torch.ones(interpolates_discriminator_out.size()).cuda(),
                                  retain_graph=True,
                                  create_graph=True)[0]
        gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()     # calculate the gradient penalty

        return wgan_loss, gp_lambda * gradient_penalty


def gan_loss(type, **kwargs):
    if type == 'G':
        deblurred_discriminator_out = kwargs['deblurred_discriminator_out']                         # get the cross_entropy between the D out and ones (true)                      
        return F.binary_cross_entropy(deblurred_discriminator_out, torch.ones_like(deblurred_discriminator_out))        # maximize this loss, gradient asent

    elif type == 'D':
        sharp_discriminator_out = kwargs['sharp_discriminator_out']
        deblurred_discriminator_out = kwargs['deblurred_discriminator_out']

        # GAN loss                                                                                  # get the loss by predict all true to 1 and all fake to 0
        real_loss = F.binary_cross_entropy(sharp_discriminator_out, torch.ones_like(sharp_discriminator_out))
        fake_loss = F.binary_cross_entropy(deblurred_discriminator_out, torch.zeros_like(deblurred_discriminator_out))
        return (real_loss + fake_loss) / 2.0                                                        # D try to maximize this loss

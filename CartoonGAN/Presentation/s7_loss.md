### Generator Loss (`g_loss`)

The generator loss in the code is defined as:
\[ g\_loss = -\text{torch.mean}(\text{torch.log}(\text{torch.sigmoid}(\text{self.discriminator}(fake\_cartoon)))) + 5000 \times vgg\_loss \]

This loss combines two components:
1. **Adversarial Loss**: This encourages the generator to produce images that the discriminator classifies as real.
2. **VGG Loss (Perceptual Loss)**: This encourages the generated images to be perceptually similar to the real images.

#### Adversarial Loss

The adversarial loss for the generator can be expressed mathematically as:
\[ \mathcal{L}_{adv} = - \mathbb{E}_{\mathbf{z} \sim p_z(\mathbf{z})} \left[ \log D(G(\mathbf{z})) \right] \]
where:
- \( G \) is the generator.
- \( D \) is the discriminator.
- \( \mathbf{z} \) is the input noise vector.
- \( p_z(\mathbf{z}) \) is the distribution of the noise.

The code computes this as:
\[ \mathcal{L}_{adv} = -\frac{1}{N} \sum_{i=1}^{N} \log \left( \sigma \left( D(G(\mathbf{z}_i)) \right) \right) \]
where \( \sigma \) is the sigmoid function, \( N \) is the batch size, and \( G(\mathbf{z}_i) \) is the generated image for the \( i \)-th noise vector.

#### VGG Loss

The VGG loss can be expressed as:
\[ \mathcal{L}_{VGG} = \frac{1}{HWC} \sum_{h=1}^{H} \sum_{w=1}^{W} \sum_{c=1}^{C} \left| \phi(x)_{h,w,c} - \phi(\hat{x})_{h,w,c} \right| \]
where:
- \( \phi(x) \) and \( \phi(\hat{x}) \) are the VGG feature representations of the real image \( x \) and the generated image \( \hat{x} \), respectively.
- \( H \), \( W \), and \( C \) are the height, width, and number of channels of the feature map.

Combining these, the total generator loss is:
\[ \mathcal{L}_{G} = \mathcal{L}_{adv} + 5000 \times \mathcal{L}_{VGG} \]

### Discriminator Loss (`d_loss`)

The discriminator loss in the code is defined as:
\[ d\_loss = -\text{torch.mean}(\text{torch.log}(\text{torch.sigmoid}(real\_logit\_cartoon)) + \text{torch.log}(1 - \text{torch.sigmoid}(fake\_logit\_cartoon)) + \text{torch.log}(1 - \text{torch.sigmoid}(logit\_blur))) \]

This loss is designed to encourage the discriminator to correctly classify real images as real and fake (including blurred) images as fake.

#### Real vs. Fake Loss

The adversarial loss for the discriminator can be expressed as:
\[ \mathcal{L}_{real} = - \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})} \left[ \log D(\mathbf{x}) \right] \]
\[ \mathcal{L}_{fake} = - \mathbb{E}_{\mathbf{z} \sim p_z(\mathbf{z})} \left[ \log (1 - D(G(\mathbf{z}))) \right] \]

#### Blur Discriminator Loss

Additionally, a blurred version of the images is used to further train the discriminator:
\[ \mathcal{L}_{blur} = - \mathbb{E}_{\mathbf{x}_{blur}} \left[ \log (1 - D(\mathbf{x}_{blur})) \right] \]

Combining these, the total discriminator loss is:
\[ \mathcal{L}_{D} = \mathcal{L}_{real} + \mathcal{L}_{fake} + \mathcal{L}_{blur} \]

In terms of the code, this can be expressed as:
\[ \mathcal{L}_{D} = -\frac{1}{N} \sum_{i=1}^{N} \left( \log \left( \sigma \left( D(\mathbf{x}_i) \right) \right) + \log \left( 1 - \sigma \left( D(G(\mathbf{z}_i)) \right) \right) + \log \left( 1 - \sigma \left( D(\mathbf{x}_{blur_i}) \right) \right) \right) \]
where \( \sigma \) is the sigmoid function, \( N \) is the batch size, \( \mathbf{x}_i \) are real images, \( G(\mathbf{z}_i) \) are fake images, and \( \mathbf{x}_{blur_i} \) are blurred images.

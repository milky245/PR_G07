import torch
import torch.nn as nn
import torch.nn.functional as F

def adaptive_instance_norm(content, style, epsilon=1e-5):
    c_mean, c_var = torch.mean(content, dim=[2, 3], keepdim=True), torch.var(content, dim=[2, 3], keepdim=True)
    s_mean, s_var = torch.mean(style, dim=[2, 3], keepdim=True), torch.var(style, dim=[2, 3], keepdim=True)
    c_std, s_std = torch.sqrt(c_var + epsilon), torch.sqrt(s_var + epsilon)
    return s_std * (content - c_mean) / c_std + s_mean

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        weight = getattr(self.module, self.name + "_bar")

        height = weight.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = F.normalize(torch.matmul(weight.data.view(height, -1).t(), u.data), dim=0)
            u.data = F.normalize(torch.matmul(weight.data.view(height, -1), v.data), dim=0)

        sigma = torch.dot(u.data, torch.matmul(weight.data.view(height, -1), v.data))
        weight_sn = weight / sigma.expand_as(weight)
        setattr(self.module, self.name, weight_sn)

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            weight = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        weight = getattr(self.module, self.name)

        height = weight.data.shape[0]
        width = weight.data.view(height, -1).shape[1]

        u = F.normalize(weight.data.new(height).normal_(0, 1), dim=0, eps=1e-12)
        v = F.normalize(weight.data.new(width).normal_(0, 1), dim=0, eps=1e-12)
        weight_bar = weight.data

        self.module.register_parameter(self.name + "_u", nn.Parameter(u, requires_grad=False))
        self.module.register_parameter(self.name + "_v", nn.Parameter(v, requires_grad=False))
        self.module.register_parameter(self.name + "_bar", nn.Parameter(weight_bar, requires_grad=True))

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def conv_spectral_norm(in_channels, out_channels, kernel_size, stride=1, padding=0):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    return SpectralNorm(conv)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

if __name__ == '__main__':
    pass

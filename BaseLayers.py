from Imports import *


class EncoderStep(nn.Module):
    def __init__(self, in_channels, kernel_size,  out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        if not isinstance(kernel_size, tuple):
            kernel_size = tuple([kernel_size])
        self.padding = tuple([np.floor(k / 2).astype(int) for k in kernel_size])

        self.conv2d_1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding=self.padding)
        self.conv2d_2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, padding=self.padding)

    def forward(self, x):
        c1 = F.leaky_relu(self.conv2d_1(x), 0.3)
        c2 = F.leaky_relu(self.conv2d_2(c1), 0.3)
        maxpool = F.max_pool2d(c2, 2)
        return [c2, maxpool]


class DecoderStep(nn.Module):
    def __init__(self, in_channels, n_replicas, kernel_size, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.n_replicas = n_replicas

        if not isinstance(kernel_size, tuple):
            kernel_size = tuple([kernel_size])
        self.padding = tuple([np.floor(k / 2).astype(int) for k in kernel_size])

        self.conv2d_1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding=self.padding)
        self.conv2d_2 = nn.Conv2d(int((self.in_channels / 2) * (self.n_replicas + 1)), self.out_channels, self.kernel_size, padding=self.padding)
        self.conv2d_3 = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, padding=self.padding)

    def forward(self, x, skip_input):
        upsample = F.interpolate(x, scale_factor=2)
        c1 = F.leaky_relu(self.conv2d_1(upsample), 0.3)

        concat = torch.cat([skip_input, c1], dim=1)

        c2 = F.leaky_relu(self.conv2d_2(concat), 0.3)
        c3 = F.leaky_relu(self.conv2d_3(c2), 0.3)

        return c3


class Bottleneck(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels):
        super().__init__()

        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Bottleneck layers
        self.convs_list = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 3, padding=1)])
        for i in range(self.num_layers - 1):
            self.convs_list.append(nn.Conv2d(self.out_channels, self.out_channels,
                                             3, padding=1))

    def forward(self, x):
        for layer in self.convs_list:
            x = F.leaky_relu(layer(x), 0.3)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, kernel_size, num_levels, base_num_filters, attention, attention_heads, axes):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_levels = num_levels
        self.base_num_filters = base_num_filters
        self.attention = attention

        # Encoder conv_down blocks
        self.convs_down_list = nn.ModuleList([EncoderStep(self.in_channels, self.kernel_size, self.base_num_filters)])
        for i in range(self.num_levels - 1):
            self.convs_down_list.append(EncoderStep(2**i * self.base_num_filters, self.kernel_size, 2**(i+1) * self.base_num_filters))

        if self.attention:
            self.attentions = nn.ModuleList([Attention(2**i * self.base_num_filters, axes=axes, heads=attention_heads) for i in range(self.num_levels)])

    def forward(self, x):
        skip_tensors = []
        for i, block in enumerate(self.convs_down_list):
            skip, x = block(x)

            if self.attention and i:
                x = self.attentions[i](x)

            skip_tensors.append(skip)

        return [skip_tensors, x]


class Decoder(nn.Module):
    def __init__(self, in_channels, n_replicas, num_levels, kernel_size):
        super().__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.n_replicas = n_replicas

        # Decoder conv_up blocks
        self.convs_up_list = nn.ModuleList([DecoderStep(int(2**(-i) * self.in_channels), self.n_replicas, self.kernel_size, int(2**(-i-1) * self.in_channels))
                                            for i in range(self.num_levels)])

    def forward(self, x, skip_inputs):
        for block_id, block in enumerate(self.convs_up_list):
            x = block(x, skip_inputs[self.num_levels - block_id - 1])

        return x


class FinalLayer(nn.Module):

    def __init__(self, in_channels, out_channels, activation=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        # Output layer
        self.last_conv2d = nn.Conv2d(in_channels, out_channels, 1, padding=0)

    def forward(self, x):

        x = self.last_conv2d(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Attention(nn.Module):

    def __init__(self, in_channels, axes, heads):
        super().__init__()
        self.axes = axes
        self.heads = heads
        self.in_channels = in_channels

        assert not self.in_channels % self.heads

        self.q_convs = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, groups=self.heads)
        self.k_convs = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, groups=self.heads)
        self.v_convs = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, groups=self.heads)

        self.norm_factor = np.sqrt(self.in_channels / self.heads).astype(float)
        if self.axes == 'xy':
            self.dropout = nn.Dropout2d(p=0.2)
        else:
            self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x, *y):

        q = self.q_convs(x)
        k = self.k_convs(x if not y else y[0])
        v = self.v_convs(x if not y else y[0])

        b_q, c_q, h_q, w_q = q.shape
        b_k, c_k, h_k, w_k = k.shape
        b_v, c_v, h_v, w_v = v.shape

        if self.axes == 'xy':
            q = q.view(b_q, self.heads, c_q // self.heads, -1)
            k = k.view(b_k, self.heads, c_k // self.heads, -1)
            v = v.view(b_v, self.heads, c_v // self.heads, -1)
            attention = torch.einsum('bncd,bnce -> bnde', q, k)
            attention = nn.Softmax(-1)(attention / self.norm_factor)
            #attention = self.dropout(attention)
            o = torch.einsum('bned,bncd -> bnce', attention, v).contiguous().view(b_q, c_q, h_q, w_q)

        elif self.axes == 'x':
            q = q.view(b_q, self.heads, c_q // self.heads, h_k, w_k)
            k = k.view(b_k, self.heads, c_k // self.heads, h_k, w_k)
            v = v.view(b_v, self.heads, c_v // self.heads, h_v, w_v)
            attention = torch.einsum('bnchw,bnche -> bnhwe', q, k)
            attention = nn.Softmax(-1)(attention / self.norm_factor)
            #attention = self.dropout(attention)
            o = torch.einsum('bnhwe,bnche -> bnchw', attention, v).contiguous().view(b_q, c_q, h_q, w_q)

        elif self.axes == 'y':
            q = q.view(b_q, self.heads, c_q // self.heads, h_k, w_k)
            k = k.view(b_k, self.heads, c_k // self.heads, h_k, w_k)
            v = v.view(b_v, self.heads, c_v // self.heads, h_v, w_v)
            attention = torch.einsum('bnchw,bncew -> bnhwe', q, k)
            attention = nn.Softmax(-1)(attention / self.norm_factor)
            #attention = self.dropout(attention)
            o = torch.einsum('bnhwe,bncew -> bnchw', attention, v).contiguous().view(b_q, c_q, h_q, w_q)

        return x + o
















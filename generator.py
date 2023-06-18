import torch.nn as nn
import torch

# The blocks used in Generator are defined below
# Gated Linear Unit
class GLU(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.linear1 = nn.Linear(features, features)
        self.linear2 = nn.Linear(features, features)

    def forward(self, x):
        return (self.linear1(x) * self.linear2(x).sigmoid())

# Convolutional Block       
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, drop=False, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose1d(in_channels, out_channels, **kwargs),
            nn.Dropout()
            if drop==True
            else nn.Identity(),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.Mish(out_channels) if use_act else nn.Identity()
            )

    def forward(self, x):
        return (self.conv(x))

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=9, stride=1, padding=4),
            ConvBlock(channels, channels, use_act=False, kernel_size=9, stride=1, padding=4),
        )
    def forward(self, x):
        return x + self.block(x)

# Now forming the generator  
class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__() 
        self.Mapping_1 = GLU(1024)                                                                              #Nx1x1024
        self.Mapping_2 = GLU(1024)                                                                              #Nx1x1024
        self.Mapping_3 = GLU(1024)                                                                              #Nx1x1024

        
        self.initial = ConvBlock(channels*1, channels*64, kernel_size=9, stride=1, padding=4)                   #Nx64x1024
        self.down_1 = ConvBlock(channels*64, channels*128, kernel_size=6, stride=2, padding=2)                  #Nx128x512
        self.down_2 = ConvBlock(channels*128, channels*256, kernel_size=6, stride=2, padding=2)                 #Nx256x256
        self.down_3 = ConvBlock(channels*256, channels*512, kernel_size=6, stride=2, padding=2)                 #Nx512x128

        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(channels*512) for _ in range(5)
             ]       
        )                                                                                                          #Nx512x128
        self.up_1 = ConvBlock(channels*512, channels*256, down=False, kernel_size=6, stride=2, padding=2)          #Nx256x256
        self.up_2 = ConvBlock(channels*256, channels*128, down=False, kernel_size=6, stride=2, padding=2)          #Nx128x512
        self.up_3 = ConvBlock(channels*128, channels*64, down=False, kernel_size=6, stride=2, padding=2)           #Nx64x1024
        self.last = nn.Conv1d(channels*64, channels*1, kernel_size=9, stride=1, padding=4, padding_mode="reflect") #Nx1x1024

    def forward(self, x):
        x_0 = self.Mapping_1(x)
        x_00 = self.Mapping_2(x_0+x)
        x_000 = self.Mapping_3(x_00+x_0+x)
        
        x_1 = self.initial(x_000)
        x_2 = self.down_1(x_1)
        x_3 = self.down_2(x_2)
        x_4 = self.down_3(x_3)

        x_res = self.res_blocks(x_4)
        
        x_5 = self.up_1(x_res+x_4)
        x_6 = self.up_2(x_5+x_3)
        x_7 = self.up_3(x_6+x_2)
        x_8 = self.last(x_7+x_1)
        
        return torch.tanh(x_8)
    
    

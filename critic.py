import torch.nn as nn
from generator import ConvBlock, GLU

# Now forming the Critic
class Critic(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.Mapping_1 = GLU(1024)                                                                  #Nx1x1024
        self.Mapping_2 = GLU(1024)                                                                  #Nx1x1024
        self.Mapping_3 = GLU(1024)                                                                  #Nx1x1024
        
        self.down_block_1 = ConvBlock(channels*1, channels*64, kernel_size=9, stride=1, padding=4) #Nx64x1024
        self.down_block_2 = ConvBlock(channels*64, channels*128, kernel_size=6, stride=2, padding=2) #Nx128x512
        self.down_block_3 = ConvBlock(channels*128, channels*256, kernel_size=6, stride=2, padding=2) #Nx256x256
        self.down_block_4 = ConvBlock(channels*256, channels*512,  kernel_size=6, stride=2, padding=2) #Nx512x128
        self.down_block_5 = ConvBlock(channels*512, channels*1024,  kernel_size=6, stride=2, padding=2) #Nx1024x64

        self.last = nn.Conv1d(channels*1024, channels*1, kernel_size=64, stride=1, padding=0, padding_mode="reflect") #Nx1x1
        
    def forward(self, x):
        x_0 = self.Mapping_1(x)
        x_00 = self.Mapping_2(x_0+x)
        x_000 = self.Mapping_3(x_00+x_0+x)
        
        x = self.down_block_1(x_000)
        x = self.down_block_2(x)
        x = self.down_block_3(x)
        x = self.down_block_4(x)
        x = self.down_block_5(x)

        return self.last(x)
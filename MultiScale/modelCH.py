import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

print('Loading CH-Model (Inter-FFT)')


#Hard-coded: 
# Number of channels for the frequency feature maps 
# In this example, we use 16 channels, equal to the implementation of our paper
#   with the encoders: ResNet-50 and Dino_v2 (ViT-Small)
num_FFT = 16
16

class FFT_features(nn.Module):
    def __init__(self,out_planes) -> None:
        super().__init__()
        self.out_planes = out_planes
        self.in_conv = nn.Sequential(
            nn.Conv2d(3,out_planes//4,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_planes//4),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_planes//2,out_planes//2,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_planes//2),
            nn.ReLU()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_planes//4,out_planes,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self,img):
        f_data = int(self.out_planes//4)
        x = self.in_conv(img)
        x_fft = torch.fft.fft(x,dim=1)
        x_freq = torch.cat((torch.real(x_fft),torch.imag(x_fft)),dim=1)
        out_freq = self.conv(x_freq)
        out_complex = torch.complex(out_freq[:,:f_data],out_freq[:,f_data:])
        out = abs(torch.fft.ifft(out_complex,dim=1))
        out = self.out_conv(out)
        return out

#Frequency details part

class ExtraFreq(nn.Module):
    def __init__(self,scales) -> None:
        super().__init__()
        self.scales = scales

        self.layer0 = FFT_features(num_FFT)
        self.layer1 = FFT_features(num_FFT)
        self.layer2 = FFT_features(num_FFT)
        self.layer3 = FFT_features(num_FFT)
        self.layer4 = FFT_features(num_FFT)

    def forward(self,x):
        out = []
        size_0 = np.array(np.ceil(np.asarray(x.shape[-2:])*self.scales[0]),dtype=int)
        l0 = F.interpolate(x,size=tuple(size_0),mode='bilinear',align_corners=True)
        l0 = self.layer0(l0); out.append(l0)
        size_1 = np.array(np.ceil(np.asarray(x.shape[-2:])*self.scales[1]),dtype=int)
        l1 = F.interpolate(x,size=tuple(size_1),mode='bilinear',align_corners=True)
        l1 = self.layer1(l1); out.append(l1)
        size_2 = np.array(np.ceil(np.asarray(x.shape[-2:])*self.scales[2]),dtype=int)
        l2 = F.interpolate(x,size=tuple(size_2),mode='bilinear',align_corners=True)
        l2 = self.layer2(l2); out.append(l2)
        size_3 = np.array(np.ceil(np.asarray(x.shape[-2:])*self.scales[3]),dtype=int)
        l3 = F.interpolate(x,size=tuple(size_3),mode='bilinear',align_corners=True)
        l3 = self.layer3(l3); out.append(l3)
        return out
        
#Decoder part

class FeatureFusion(nn.Module):
    def __init__(self,in_planes) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_FFT+in_planes,in_planes,kernel_size=1,bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )
    
    def forward(self,x1,x2):
        x = torch.cat((x1,x2),dim=1)
        out = self.conv(x)
        return out



# How to include the frequency features in your architecture

class Decoder(nn.Module):
    def __init__(self, out_channels=[256, 512, 1024, 1024]):
        super().__init__()
        
        '''
        Your decoder layers
        '''
        
        self.ff1 = FeatureFusion(out_channels[0])
        self.ff2 = FeatureFusion(out_channels[1])
        self.ff3 = FeatureFusion(out_channels[2])
        self.ff4 = FeatureFusion(out_channels[3])


    def forward(self, SpatialFeatures, FrequencyFeatures):

        layer_1, layer_2, layer_3, layer_4 = SpatialFeatures
        
        # Frequency meets space: Fusion of multi-resolution spatial features 
        #   and multi-resolution frequency features
        layer_1 = self.ff1(layer_1,FrequencyFeatures[0])
        layer_2 = self.ff2(layer_2,FrequencyFeatures[1])
        layer_3 = self.ff3(layer_3,FrequencyFeatures[2])
        layer_4 = self.ff4(layer_4,FrequencyFeatures[3])

        '''
            Your decoder forward
        '''

        return True
        
        
class Your_Network(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485,0.456,0.406])[None, :, None, None])
    x_std  = torch.FloatTensor(np.array([0.229,0.224,0.225])[None, :, None, None])
    def __init__(self, out_channels=[48, 96, 192, 384]):
        super().__init__()
                
        scales = [2/7,1/7,1/14,1/28]

        '''
        Your Encoder
        '''    

        self.freqDetail = ExtraFreq(scales)
        # Your Decoder
        self.decoder = Decoder(out_channels)

    def forward(self,x):
        bs,ch,h,w = x.shape
        h_patches,w_patches = h//14,w//14
        x = self._prepare_x(x)
        #Your encoder goes here to obtain the spatial features
        spa_features = self.encoder(x)
        # This example if the a ViT-like network
        # Reshape the outpur of the encoder as [Batch x Channels x Height x Width]
        spa_features = spa_features.reshape(bs,h_patches,w_patches,-1).permute([0,3,1,2])
        #Compute the Frequency features
        freq_det = self.freqDetail(x)
        #Feed to the decoder the spatial features from your encoder and the Frequency features
        out = self.decoder(spa_features,freq_det)
        return out

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

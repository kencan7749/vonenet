
from collections import OrderedDict
from torch import nn
from .modules import VOneBlock, MNBlock
from .back_ends import ResNetBackEnd, Bottleneck, AlexNetBackEnd, CORnetSBackEnd, VGG19BackEnd
from .params import generate_gabor_param, generate_MN_param
import numpy as np


def VOneNet(sf_corr=0.75, sf_max=9, sf_min=0, rand_param=False, gabor_seed=0,
            simple_channels=256, complex_channels=256,
            noise_mode='neuronal', noise_scale=0.35, noise_level=0.07, k_exc=25,
            model_arch='resnet50', image_size=224, visual_degrees=8, ksize=25, stride=4):


    out_channels = simple_channels + complex_channels

    sf, theta, phase, nx, ny = generate_gabor_param(out_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min)

    gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
                    'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                    'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
    arch_params = {'k_exc': k_exc, 'arch': model_arch, 'ksize': ksize, 'stride': stride}


    # Conversions
    ppd = image_size / visual_degrees

    sf = sf / ppd
    sigx = nx / sf
    sigy = ny / sf
    theta = theta/180 * np.pi
    phase = phase / 180 * np.pi

    vone_block = VOneBlock(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase,
                           k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                           simple_channels=simple_channels, complex_channels=complex_channels,
                           ksize=ksize, stride=stride, input_size=image_size)

    if model_arch:
        bottleneck = nn.Conv2d(out_channels, 64, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(bottleneck.weight, mode='fan_out', nonlinearity='relu')

        if model_arch.lower() == 'resnet50':
            print('Model: ', 'VOneResnet50')
            model_back_end = ResNetBackEnd(block=Bottleneck, layers=[3, 4, 6, 3])
        elif model_arch.lower() == 'alexnet':
            print('Model: ', 'VOneAlexNet')
            model_back_end = AlexNetBackEnd()
        elif model_arch.lower() == 'cornets':
            print('Model: ', 'VOneCORnet-S')
            model_back_end = CORnetSBackEnd()

        model = nn.Sequential(OrderedDict([
            ('vone_block', vone_block),
            ('bottleneck', bottleneck),
            ('model', model_back_end),
        ]))
    else:
        print('Model: ', 'VOneNet')
        model = vone_block

    model.image_size = image_size
    model.visual_degrees = visual_degrees
    model.gabor_params = gabor_params
    model.arch_params = arch_params

    return model


def MNOneNet(sf_corr=0.75, sf_max=9, sf_min=0, tf_max=4, tf_min=0, rand_param=False, gabor_seed=0,
                        simple_channels=0, complex_channels=512, 
                        k_exc=25, model_arch='alexnet', image_size=224, visual_degrees=8, ksize=25, kernel_size_t = 16, strides=(17, 4,4), fps=30):
    
    out_channels = simple_channels + complex_channels

    sf, theta, phase, nx, ny, tf = generate_MN_param(out_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min, tf_max, tf_min)
    
    MN_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
                    'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                     'tf_max': tf_max, 'tf_min': tf_min, 'tf': tf.copy(),
                    'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
    arch_params = {'k_exc': k_exc, 'arch': model_arch, 'ksize': ksize, 'ksize_t': kernel_size_t, 'stride': strides}
    
    ppd = image_size / visual_degrees
    sf = sf / ppd
    sigx = nx / sf
    sigy = ny / sf
    theta = theta/180 * np.pi
    phase = phase / 180 * np.pi

    tf = tf
    
    
    mn_block = MNBlock(sf=sf, tf=tf,  theta=theta, sigx=sigx, sigy=sigy, phase=phase,
                           k_exc=k_exc, 
                           simple_channels=simple_channels, complex_channels=complex_channels,
                           ksize=ksize, ksize_t = kernel_size_t, stride=strides, input_size=image_size, fps=fps)
    
    if model_arch:
        bottleneck = nn.Conv2d(out_channels, 64, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(bottleneck.weight, mode='fan_out', nonlinearity='relu')

        if model_arch.lower() == 'resnet50':
            print('Model: ', 'VOneResnet50')
            model_back_end = ResNetBackEnd(block=Bottleneck, layers=[3, 4, 6, 3])
        elif model_arch.lower() == 'alexnet':
            print('Model: ', 'VOneAlexNet')
            model_back_end = AlexNetBackEnd()
        elif model_arch.lower() == 'cornets':
            print('Model: ', 'VOneCORnet-S')
            model_back_end = CORnetSBackEnd()
        elif model_arch.lower() == 'vgg19':
            print('Model: ', 'VOneCORnet-S')
            model_back_end = VGG19BackEnd()
            
            

        model = nn.Sequential(OrderedDict([
            ('mn_block', mn_block),
            ('bottleneck', bottleneck),
            ('model', model_back_end),
        ]))
    else:
        print('Model: ', 'VOneNet')
        model = vone_block

    model.image_size = image_size
    model.visual_degrees = visual_degrees
    model.MN_params = MN_params
    model.arch_params = arch_params

    return model

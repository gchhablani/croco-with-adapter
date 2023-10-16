# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
from models.croco import CroCoNet
from PIL import Image
import torchvision.transforms
from torchvision.transforms import ToTensor, Normalize, Compose

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count()>0 else 'cpu')
    
    # load 224x224 images and transform them to tensor 
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_mean_tensor = torch.tensor(imagenet_mean).view(1,3,1,1).to(device, non_blocking=True)
    imagenet_std = [0.229, 0.224, 0.225]
    imagenet_std_tensor = torch.tensor(imagenet_std).view(1,3,1,1).to(device, non_blocking=True)
    trfs = Compose([ToTensor(), Normalize(mean=imagenet_mean, std=imagenet_std)])
    image1 = trfs(Image.open('assets/Chateau1.png').convert('RGB')).to(device, non_blocking=True).unsqueeze(0)
    image2 = trfs(Image.open('assets/Chateau2.png').convert('RGB')).to(device, non_blocking=True).unsqueeze(0)
    
    # load model 
    ckpt = torch.load('pretrained_models/CroCo_V2_ViTBase_SmallDecoder.pth', 'cpu')
    # NOTE: Set `do_mask` in `forward` to False explicitly as that is random and can cause issues!
    model_with_adapter = CroCoNet(adapter=True, **ckpt.get('croco_kwargs',{})).to(device)
    model = CroCoNet(adapter=False, **ckpt.get('croco_kwargs',{})).to(device)
    model.eval()
    model_with_adapter.eval()
    msg = model.load_state_dict(ckpt['model'], strict=True)
    msg = model_with_adapter.load_state_dict(ckpt['model'], strict=False)

    print("Msg while loading adapter model: ", msg)
    
    # forward 
    with torch.inference_mode():
        out, mask, target = model(image1, image2)
    with torch.inference_mode():
        out2, mask2, target2 = model_with_adapter(image1, image2)
    
    # Because the initialization is with zeros for adapter module
    assert torch.allclose(out, out2, 1e-2)
    assert torch.allclose(mask, mask2, 1e-2)
    assert torch.allclose(target, target2, 1e-2)

if __name__=="__main__":
    main()

"""Frozen ResNet18 style descriptors, with cached features and explicit image IDs."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

class ImageFiles(Dataset):
    def __init__(self,paths):
        self.paths=paths
        self.transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as im:
            rgba=im.convert('RGBA')
            image=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB')
        return self.transform(image)

class StyleResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    def forward(self,x):
        m=self.backbone
        x=m.maxpool(m.relu(m.bn1(m.conv1(x))))
        features=[]
        for stage in [m.layer1,m.layer2,m.layer3,m.layer4]:
            x=stage(x)
            # Channel means and standard deviations summarize rendering texture.
            features.extend([x.mean((2,3)),x.var((2,3),unbiased=False).clamp_min(0).sqrt()])
        return torch.cat(features,dim=1)

def extract_learned(root,split,ids):
    root=Path(root); out=root/'artifacts'; out.mkdir(exist_ok=True)
    cache=out/f'learned_{split}.npz'
    if cache.exists():
        data=np.load(cache)
        assert data['ids'].tolist()==list(ids), 'Cached image ID mismatch'
        assert str(data['descriptor'])=='resnet18_imagenet_v1_mean_std_128_white'
        return data['features']
    torch.hub.set_dir(str(out/'torch_cache'))
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print('Feature extraction device:',device,flush=True)
    model=StyleResNet().to(device).eval()
    paths=[root/split/f'{id}.png' for id in ids]
    batches=DataLoader(ImageFiles(paths),batch_size=64,shuffle=False,num_workers=0)
    result=[]
    with torch.inference_mode():
        for i,x in enumerate(batches):
            result.append(model(x.to(device)).cpu().numpy())
            if (i+1)%20==0:print(split,'learned features:',min((i+1)*64,len(ids)),'/',len(ids),flush=True)
    X=np.concatenate(result)
    assert X.shape==(len(ids),1920) and np.isfinite(X).all()
    np.savez_compressed(cache,features=X,ids=np.asarray(ids,dtype=str),descriptor='resnet18_imagenet_v1_mean_std_128_white')
    return X

if __name__=='__main__':
    import pandas as pd,sys
    root=Path(__file__).resolve().parent
    split=sys.argv[1] if len(sys.argv)>1 else 'train'
    ids=pd.read_csv(root/'train_labels.csv',dtype={'Id':str}).Id.tolist() if split=='train' else sorted(p.stem for p in (root/'test').glob('*.png'))
    extract_learned(root,split,ids)

"""Image style features (color, opacity, local texture); no file metadata or IDs."""
import numpy as np
from PIL import Image
from scipy import ndimage

def extract_style(path):
    with Image.open(path) as im:
        rgba=np.asarray(im.convert('RGBA'),dtype=np.float32)/255
    rgb=rgba[:,:,:3]; a=rgba[:,:,3]; mask=a>.05
    if not mask.any(): mask=np.ones_like(a,dtype=bool)
    white=rgb*a[:,:,None]+1-a[:,:,None]
    gray=white@np.array([.299,.587,.114],dtype=np.float32)
    maxc=rgb.max(2); minc=rgb.min(2); sat=(maxc-minc)/(maxc+1e-6)
    feats=[]
    def hist(x,bins=32,ran=(0,1)):
        h=np.histogram(x,bins=bins,range=ran)[0].astype('float32'); return h/max(1,h.sum())
    for channel in [rgb[:,:,i] for i in range(3)]+[maxc,sat,rgb.mean(2),a]:
        feats.extend(hist(channel[mask]))
        feats.extend(np.quantile(channel[mask],[0,.05,.1,.25,.5,.75,.9,.95,1]))
        feats.extend([channel[mask].mean(),channel[mask].std()])
    feats.extend(hist(a.ravel(),64))
    # Joint color histogram captures provider palette and shading.
    for pair in [(0,1),(0,2),(1,2)]:
        h=np.histogram2d(rgb[:,:,pair[0]][mask],rgb[:,:,pair[1]][mask],bins=12,range=((0,1),(0,1)))[0]
        feats.extend((h/max(1,h.sum())).ravel())
    for image in [gray,a,sat]:
        for sigma in [0,1,2]:
            x=ndimage.gaussian_filter(image,sigma) if sigma else image
            dx=ndimage.sobel(x,axis=1)/8; dy=ndimage.sobel(x,axis=0)/8
            grad=np.hypot(dx,dy)
            lap=ndimage.laplace(x)
            feats.extend(hist(grad[mask],32,(0,.6)))
            feats.extend(hist(lap[mask],32,(-1,1)))
            feats.extend([np.mean(grad[mask]),np.std(grad[mask]),np.mean(abs(lap[mask])),np.std(lap[mask])])
        # Rotation-invariant local binary pattern: number of bright neighbours,
        # with a separate bin for non-uniform patterns.
        center=image[1:-1,1:-1]; bits=[]
        for dy,dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
            bits.append(image[1+dy:image.shape[0]-1+dy,1+dx:image.shape[1]-1+dx]>=center)
        bits=np.stack(bits); transitions=(bits!=np.roll(bits,1,axis=0)).sum(0)
        lbp=np.where(transitions<=2,bits.sum(0),9)
        feats.extend(hist(lbp[mask[1:-1,1:-1]],10,(0,10)))
    # Spatial color and opacity statistics preserve rough structure.
    for grid in [1,2,4]:
        for yy in range(grid):
            for xx in range(grid):
                region=np.concatenate([white,a[:,:,None]],axis=2)[yy*a.shape[0]//grid:(yy+1)*a.shape[0]//grid,xx*a.shape[1]//grid:(xx+1)*a.shape[1]//grid]
                feats.extend(region.mean((0,1))); feats.extend(region.std((0,1)))
    return np.asarray(feats,dtype=np.float32)

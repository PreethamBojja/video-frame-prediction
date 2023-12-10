import numpy as np

def MAE(pred, true):
    _pred = np.transpose(pred, (1, 0, 2, 3, 4))
    _true = np.transpose(true, (1, 0, 2, 3, 4))
    tmae = np.sum(np.mean(np.abs(_pred-_true),axis=1), axis=(1,2,3))

    return np.mean(np.abs(pred-true),axis=(0,1)).sum(), tmae

def MSE(pred, true):
    return np.mean((pred-true)**2,axis=(0,1)).sum()

def metric(pred, true, mean, std, return_ssim_psnr=False, clip_range=[0, 1]):
    pred = pred*std + mean
    true = true*std + mean
    mae, tmae = MAE(pred, true)
    mse = MSE(pred, true)
    return mse, mae

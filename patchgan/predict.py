import rasterio
import numpy as np
import torch
from torchvision.transforms.functional import five_crop


device = 'cuda' if torch.cuda.is_available() else 'cpu'
crop_size = 256


def load_tiff(img_tiff):
    img = rasterio.open(img_tiff)
    img_plot = np.dstack([img.read(1), img.read(2), img.read(3), img.read(4)])

    # clean up artifacts in the data
    img_plot[np.abs(img_plot) > 1.e20] = 0
    img_plot[np.isnan(img_plot)] = 0

    # remove negative signal
    img_plot = img_plot - np.percentile(img_plot.flatten(), 2)

    norm = np.nansum(img_plot, axis=-1, keepdims=True)
    img_plot = img_plot / (norm + 1.e-3)

    # if there is no signal, return an image with zeros
    # since the next step is normalization
    if np.nanmax(img_plot) < 0.02:
        return np.zeros_like(np.transpose(img_plot, axes=(2, 0, 1)))

    # normalize the image per-pixel
    img_plot = img_plot / np.percentile(img_plot[img_plot > 0.].flatten(), 99)
    img = np.clip(img_plot, 0, 1)
    img[~np.isfinite(img)] = 0.

    # reshape the array to be compatible with PyTorch inputs
    img = np.asarray(np.transpose(img, axes=(2, 0, 1)))

    return img


def get_five_crop(img):
    # do a five crop to the image
    xfull = load_tiff(img)

    XF = torch.Tensor(xfull)  # .to(device)

    X = torch.stack(
        list(five_crop(XF, size=(crop_size, crop_size)))).cpu().numpy()
    return X


def get_prediction(img, generator):
    # load the full image
    xfull = load_tiff(img)

    XF = torch.Tensor(xfull)  # .to(device)

    # do the 5-crop and predict on the image
    X = torch.stack(
        list(five_crop(XF, size=(crop_size, crop_size)))).to(device)
    Y = generator(X).cpu().numpy()

    Ytl, Ytr, Ybl, Ybr, Yc = Y[0], Y[1], Y[2], Y[3], Y[4]

    # recreate the full mask from the
    # 5-cropped output
    y = np.zeros(xfull.shape[1:])
    count = np.zeros(xfull.shape[1:])

    # we will add each 5 crop back to our
    # "full" mask
    y[:crop_size, :crop_size] += Ytl[0]
    y[:crop_size:, -crop_size:] += Ytr[0]
    y[-crop_size:, :crop_size] += Ybl[0]
    y[-crop_size:, -crop_size:] += Ybr[0]

    # and count how many predictions per pixel
    count[:crop_size, :crop_size] += 1
    count[:crop_size:, -crop_size:] += 1
    count[-crop_size:, :crop_size] += 1
    count[-crop_size:, -crop_size:] += 1

    # same for the center crop region
    crop_top = int(round((xfull.shape[1] - crop_size) / 2.0))
    crop_left = int(round((xfull.shape[2] - crop_size) / 2.0))
    y[crop_top:crop_top + crop_size, crop_left:crop_left + crop_size] += Yc[0]
    count[crop_top:crop_top + crop_size, crop_left:crop_left + crop_size] += 1

    # the final prediction is the total prediction
    # divided by the number of crops covering that pixel
    predmask = y / count

    return predmask


def get_disc_value(disc_input, disc):
    XF = torch.Tensor(disc_input)  # .to(device)

    X = torch.stack(
        list(five_crop(XF, size=(crop_size, crop_size)))).to(device)

    Y = disc(X).cpu().numpy()

    return np.mean(Y)

import numpy as np
import re
import pickle
import os
from tqdm import tqdm as tqdm
from PIL import Image


def read_pk(fn):
    with open(fn, 'rb') as fd:
        ret = pickle.load(fd)
    return ret


def write_pk(obj, fn):
    with open(fn, 'wb') as fd:
        pickle.dump(obj, fd)


def get_files(dr, ext='jpg|jpeg|bmp|png'):
    rex = re.compile(r'^.*\.({})$'.format(ext), re.I)
    return [os.path.join(dr, base) for base in os.listdir(dr) if rex.match(base)]


def rename_files(files, imagedir):
    # todo: don't rename files that are already sha256?
    import phashlib as ph
    newfiles = []
    hashes = []
    for fromfile in tqdm(files, total=len(files)):
        suffix = fromfile.split('.')[-1]
        sha256hash = ph.sha256_checksum(fromfile)
        tofile = sha256hash + '.' + suffix
        tofile = os.path.join(imagedir, tofile)

        newfiles.append(tofile)
        hashes.append(sha256hash)

        os.rename(fromfile, tofile)
    return newfiles, hashes


def preprocess_image_for_cropping(img):
    img = img - np.mean(img)
    img /= np.std(img)
    #img = img[:,:,0]/3 - img[:,:,1]/3 - img[:,:,2]/3
    img = np.std(img, axis=2)
    # convert image to some grayscale mush
    # for i in range(1,2):
    #    img = img[:, :, 0]/3 - img[:, :, i]/3
    #plt.imshow(img, cmap='gray')
    # plt.colorbar()
    # plt.show()
    return img


def crop_images(df, imagedir, size):
    cropped_folder = os.path.join(imagedir, 'cropped/')
    if not os.path.exists(cropped_folder):
        os.makedirs(os.path.dirname(cropped_folder), exist_ok=True)

    if 'cropped_filename' not in df:
        df['cropped_filename'] = None
    for file in tqdm(df.index, total=len(df.index)):
        try:
            original_filename = df.loc[file]['filename']
            pil_img = Image.open(original_filename)
            pil_img = pil_img.convert('RGB')

            # old: rename cropped files to hash
            #fhash = df.loc[file]['hash']

            # new: original filename
            fname = df.loc[file]['name']
            cropped_fname = os.path.join(imagedir, 'cropped/', fname + '.jpg')

            pil_img.thumbnail((size, size), Image.ANTIALIAS)
            img = np.array(pil_img)
            origimg = img.copy()
            croplines_x, croplines_y = get_crop_bbox(img)

            w, h = pil_img.size
            if len(croplines_x) is not 2:
                croplines_x = [0, w]
                print("couldn't crop {} in x-axis".format(file))
            if len(croplines_y) is not 2:
                croplines_y = [0, h]
                print("couldn't crop {} in y-axis".format(file))

            #plot_croplines(croplines_x, croplines_y, img)
            pil_img = pil_img.crop(
                (croplines_x[0], croplines_y[0], croplines_x[1], croplines_y[1]))
            pil_img.save(cropped_fname)
            df.loc[file, 'cropped_filename'] = cropped_fname
        except:
            print("Couldn't crop {}, dropping file from table".format(
                file))
            df = df.drop(file)
            continue

        # plt.imshow(pil_img)
        # plt.show()
    return df


def get_crop_bbox(img):

    img = preprocess_image_for_cropping(img)

    yrange, xrange = img.shape[:2]

    croplines_x = []
    croplines_y = []

    mean_x = [[], []]
    for x in range(xrange):
        # extract cross sections to analyze
        filterline = np.abs(img[:, x])
        # interpolate line
        boxwidth = 10
        box = np.ones(boxwidth)/boxwidth
        filterline = np.convolve(filterline, box, mode='same')

        filter_threshold = np.mean(img)/10

        # find pixels where threshold is crossed
        threshold_crossings = np.where(
            np.array(filterline) > filter_threshold)[0]

        # take mean of the found borders across image
        if len(threshold_crossings) > 0:
            croplines_x.append(
                [threshold_crossings[0], threshold_crossings[-1]])
    if len(croplines_x) > 1:
        mean_x = np.median(np.array(croplines_x), axis=0).astype(int)

    mean_y = [[], []]
    for y in range(yrange):
        # extract cross sections to analyze
        filterline = np.abs(img[y, :])
        # interpolate line
        boxwidth = 10
        box = np.ones(boxwidth)/boxwidth
        filterline = np.convolve(filterline, box, mode='same')

        filter_threshold = np.mean(img)/10

        threshold_crossings = np.where(
            np.array(filterline) > filter_threshold)[0]
        if len(threshold_crossings) > 0:
            croplines_y.append(
                [threshold_crossings[0], threshold_crossings[-1]])
    if len(croplines_y) > 1:
        mean_y = np.median(np.array(croplines_y), axis=0).astype(int)
    return mean_y, mean_x  # threshold crossings on y axis are x values to crop and vice versa

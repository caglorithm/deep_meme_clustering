import tqdm
from scipy.spatial import distance
from scipy.cluster import hierarchy
import numpy as np

import PIL.Image
import os
import shutil
from keras.preprocessing import image

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_input_InceptionV3

from keras.applications.resnet50 import preprocess_input as preprocess_input_ResNet50
from keras.applications.resnet50 import ResNet50


from keras.models import Model

import common as co

pj = os.path.join


def get_model(modelname='VGG16'):
    if modelname == 'VGG16':
        model = get_model_vgg16()
    elif modelname == 'VGG19':
        model = get_model_vgg19()
    elif modelname == 'InceptionV3':
        model = get_model_InceptionV3()
    elif modelname == 'ResNet50':
        model = get_model_ResNet50()
    else:
        raise Exception("modelname not found")
    return model


def get_model_vgg16():
    """Keras Model of the VGG16 network, with the output layer set to the
    second-to-last fully connected layer 'fc2' of shape (4096,)."""
    # base_model.summary():
    #     ....
    #     block5_conv4 (Conv2D)        (None, 15, 15, 512)       2359808
    #     _________________________________________________________________
    #     block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
    #     _________________________________________________________________
    #     flatten (Flatten)            (None, 25088)             0
    #     _________________________________________________________________
    #     fc1 (Dense)                  (None, 4096)              102764544
    #     _________________________________________________________________
    #     fc2 (Dense)                  (None, 4096)              16781312
    #     _________________________________________________________________
    #     predictions (Dense)          (None, 1000)              4097000
    #
    base_model = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('fc2').output)
    return model


def get_model_vgg19():
    base_model = VGG19(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('fc2').output)
    return model


def get_model_InceptionV3():
    base_model = InceptionV3(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('mixed10').output)
    return model


def get_model_ResNet50():
    model = ResNet50(weights='imagenet', include_top=True)
    # model = Model(inputs=base_model.input,
    #	outputs=base_model.get_layer('flatten_2').output)
    return model


def fingerprint(fn, model, size, modelname='VGG16'):
    """Load image from file `fn`, resize to `size` and run through `model`
    (keras.models.Model).

    Parameters
    ----------
    fn : str
        filename
    model : keras.models.Model instance
    size : tuple
        input image size (width, height), must match `model`, e.g. (224,224)
    prepcosseing_f: which function to use for preprocessing, depending
        on the keral model used!
    Returns
    -------
    fingerprint : 1d array
    """

    # keras.preprocessing.image.load_img() uses img.rezize(shape) with the
    # default interpolation of PIL.Image.resize() which is pretty bad (see
    # imagecluster/play/pil_resample_methods.py). Given that we are restricted
    # to small inputs of 224x224 by the VGG network, we should do our best to
    # keep as much information from the original image as possible. This is a
    # gut feeling, untested. But given that model.predict() is 10x slower than
    # PIL image loading and resizing .. who cares.
    #
    # (224, 224, 3)
    # img = image.load_img(fn, target_size=size)
    # try:
    img = PIL.Image.open(fn).resize(size, 2)

    # (224, 224, {3,1})
    arr3d = image.img_to_array(img)

    # (224, 224, 1) -> (224, 224, 3)
    #
    # Simple hack to convert a grayscale image to fake RGB by replication of
    # the image data to all 3 channels.
    #
    # Deep learning models may have learned color-specific filters, but the
    # assumption is that structural image features (edges etc) contibute more to
    # the image representation than color, such that this hack makes it possible
    # to process gray-scale images with nets trained on color images (like
    # VGG16).
    if arr3d.shape[2] == 1:
        arr3d = arr3d.repeat(3, axis=2)

    # (1, 224, 224, 3)
    arr4d = np.expand_dims(arr3d, axis=0)

    # (1, 224, 224, 3)
    if modelname == ' VGG16':
        arr4d_pp = preprocess_input_vgg16(arr4d)
    elif modelname == 'VGG19':
        arr4d_pp = preprocess_input_vgg19(arr4d)
    elif modelname == 'InceptionV3':
        arr4d_pp = preprocess_input_InceptionV3(arr4d)
    elif modelname == 'ResNet50':
        arr4d_pp = preprocess_input_ResNet50(arr4d)
    else:
        raise ValueError()

    # hack: crashes when image has alpha?
    if arr4d_pp.shape[3] == 4:
        arr4d_pp = arr4d_pp[:, :, :, :3]

    result = model.predict(arr4d_pp)[0, :].flatten()

    return result
    # except:
    #	print("FAILED: {}".format(fn))
    #	#raise Exception()
    #	return([])
# Cannot use multiprocessing (only tensorflow backend tested):
# TypeError: can't pickle _thread.lock objects
# The error doesn't come from functools.partial since those objects are
# pickable since python3. The reason is the keras.model.Model, which is not
# pickable. However keras with tensorflow backend runs multi-threaded
# (model.predict()), so we don't need that. I guess it will scale better if we
# parallelize over images than to run a muti-threaded tensorflow on each image,
# but OK. On low core counts (2-4), it won't matter.
#
# def _worker(fn, model, size):
# print(fn)
# return fn, fingerprint(fn, model, size)
##
# def fingerprints(files, model, size=(224,224)):
# worker = functools.partial(_worker,
# model=model,
# size=size)
# pool = multiprocessing.Pool(multiprocessing.cpu_count())
# return dict(pool.map(worker, files))


def fingerprints(files, model, size=(224, 224), modelname='VGG16'):
    """Calculate fingerprints for all `files`.

    Parameters
    ----------
    files : sequence
        image filenames
    model, size : see :func:`fingerprint`

    Returns
    -------
    fingerprints : dict
        {filename1: array([...]),
         filename2: array([...]),
         ...
         }
    """
    all_results = {}
    fingerprints = []
    for fn in tqdm.tqdm(files, total=len(files)):
        result = fingerprint(fn, model, size, modelname=modelname)
        if len(result) > 0:
            all_results[fn] = result

    # return dict((fn, fingerprint(fn, model, size)) for fn in files)
    return all_results


def cluster(fps, sim=0.5, method='average', metric='euclidean'):
    """Hierarchical clustering of images based on image fingerprints.

    Parameters
    ----------
    fps: dict
        output of :func:`fingerprints`
    sim : float 0..1
        similarity index
    method : see scipy.hierarchy.linkage(), all except 'centroid' produce
        pretty much the same result
    metric : see scipy.hierarchy.linkage(), make sure to use 'euclidean' in
        case of method='centroid', 'median' or 'ward'

    Returns
    -------
    clusters : nested list
        [[filename1, filename5],                    # cluster 1
         [filename23],                              # cluster 2
         [filename48, filename2, filename42, ...],  # cluster 3
         ...
         ]
    """
    assert 0 <= sim <= 1, "sim not 0..1"
    # array(list(...)): 2d array
    #   [[... fingerprint of image1 (4096,) ...],
    #    [... fingerprint of image2 (4096,) ...],
    #    ...
    #    ]
    dfps = distance.pdist(np.array(list(fps.values())), metric)
    files = list(fps.keys())
    # hierarchical/agglomerative clustering (Z = linkage matrix, construct
    # dendrogram)
    Z = hierarchy.linkage(dfps, method=method, metric=metric)
    # cut dendrogram, extract clusters
    cut = hierarchy.fcluster(Z, t=dfps.max()*(1.0-sim), criterion='distance')
    cluster_dct = dict((ii, []) for ii in np.unique(cut))
    for iimg, iclus in enumerate(cut):
        cluster_dct[iclus].append(files[iimg])
    return list(cluster_dct.values())


def make_links(clusters, cluster_dr):
    # group all clusters (cluster = list_of_files) of equal size together
    # {number_of_files1: [[list_of_files], [list_of_files],...],
    #  number_of_files2: [[list_of_files],...],
    # }
    cdct_multi = {}
    for x in clusters:
        nn = len(x)
        if nn > 1:
            if not (nn in cdct_multi.keys()):
                cdct_multi[nn] = [x]
            else:
                cdct_multi[nn].append(x)

    print("cluster dir: {}".format(cluster_dr))
    # print(f"cluster size : {ncluster}")
    if os.path.exists(cluster_dr):
        shutil.rmtree(cluster_dr)
    for nn in np.sort(list(cdct_multi.keys())):
        cluster_list = cdct_multi[nn]
        print("{} : {}".format(nn, len(cluster_list)))
        for iclus, lst in enumerate(cluster_list):
            dr = pj(cluster_dr,
                    'cluster_with_{}'.format(nn),
                    'cluster_{}'.format(iclus))
            for fn in lst:
                link = pj(dr, os.path.basename(fn))
                os.makedirs(os.path.dirname(link), exist_ok=True)
                os.symlink(os.path.abspath(fn), link)

import importlib
import pandas as pd
import numpy as np
import os, re

import imagecluster as ic
import phashlib as ph

import common as co
import imagecluster as ic

from tqdm import tqdm

from scipy.spatial import distance
from scipy.cluster import hierarchy

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ic_base_dir = 'imagecluster'
modelname = 'ResNet50'
input_size = 224
importlib.reload(ic)

imagedir = './data/'


def main():
	df, dbfn = init_df()
	cluster(df, dbfn)
	visualize_clusters(df)

def init_df():
	# initialize dataframe and run pipeline
	dbfn = os.path.join(imagedir, ic_base_dir, 'db.pk')
	if not os.path.exists(dbfn):
	    os.makedirs(os.path.dirname(dbfn), exist_ok=True)
	    print("no fingerprints database found in {}".format(dbfn))
	    #fps = ic.fingerprints(files, model, size=(input_size,input_size), modelname=modelname)
	    df_exists = 'df' in locals() or 'df' in globals()
	    if not df_exists:
	        print("Running processing pipeline ...")
	        df = process_dataset(imagedir)
	    else:
	        print("df exists already.")
	    print("writing {}".format(dbfn))
	    co.write_pk(df, dbfn)
	else:
	    print("loading fingerprints database {} ...".format(dbfn))
	    df = co.read_pk(dbfn)
	    print("done.")	
	return df, dbfn

# Feature extraction

def process_dataset(imagedir, modelname = 'ResNet50', input_size = 224):
    """
    processes a list of files (filenames) 
    
    1 - calculates sha256 hash and renames files to hash
    2 - crops out image from meme and copies into ./cropped/
    3 - calculates phash using the imagehash library
    4 - calculates dnn fingerprint using keras and tensorflow
    6 - does the same for cropped versions
    7 - applies a clustering algorithm on fingerprints of cropped images
    8 - plots all members of all clusters into a jpg file and saves results
    
    - returns a pandas dataframe with the information
    """
    files = co.get_files(imagedir)
    print("> Renaming {} files (to sha256 hash)".format(len(files)))
    files, hashes = co.rename_files(files, imagedir)
    print("done.")
    
    # create pandas dataframe containing all data
    df = pd.DataFrame(index=hashes)
    df['filename'] = files
    df['hash'] = hashes
    
    print("> Phashing {} files".format(len(files)))
    phashes = ph.return_phashes(files)
    df['phash'] = phashes
    print("done.")
    
    print("> Cropping and copying all images")
    df = co.crop_images(df, imagedir, input_size)
    print("done.")        
    
    print("> Loading Keras model {}".format(modelname))
    model, getFingerprint = ph.get_model(modelname=modelname)
    # construct fingerprint model (second to last layer)
    #getFingerprint = K.function([model.layers[0].input],
    #                              [model.layers[-2].output])
    
    print("done.")
    
    print("> Running images through DNN {}".format(modelname))
    # get fingerprints
    fps, preds, labels = ph.fingerprints(files, model, getFingerprint, size=(input_size,input_size), modelname=modelname)
    df['fingerprints'] = fps
    df['labels'] = labels
    
    print("> Running CROPPED images through DNN {}".format(modelname))
    # get fingerprints
    cfps, cpreds, clabels = ph.fingerprints(files, model, getFingerprint, size=(input_size,input_size), modelname=modelname)
    df['cropped_fingerprints'] = cfps
    df['cropped_labels'] = clabels
    
    print("done.")
    
    return df


def cluster(df, dbfn):
	print("> Clustering ...")
	fingerprint_column = 'cropped_fingerprints'
	sim = 0.5

	fingerprintdict = df.set_index('filename')[fingerprint_column].to_dict()
	# cluster and save files in folders
	ic.make_links(ic.cluster(fingerprintdict, sim), os.path.join(imagedir, ic_base_dir, 'clusters'))

	# cluster and save results in dataframe
	fps = df[fingerprint_column]
	dfps = distance.pdist(np.array(list(fps)), metric='euclidean')
	Z = hierarchy.linkage(dfps, method='average', metric='euclidean')
	cut = hierarchy.fcluster(Z, t=dfps.max()*(1.0-sim), criterion='distance')
	df['cluster'] = cut

	# save database to file
	co.write_pk(df, dbfn)
	print("done.")

def visualize_clusters(df):
	# save results on disk as jpgs
	clusterdir = os.path.join(imagedir, ic_base_dir, 'clusters')
	clusterlist = list(df['cluster'])
	unique_clusters = np.unique(df['cluster'])
	cut = df['cluster']

	for nclust in unique_clusters:
	    clustersize = clusterlist.count(nclust)
	    if clustersize > 1 and clustersize < 500:
	        print(nclust, clustersize)
	        clusterdf = df[df['cluster'] == nclust]
	        
	        labels = list(clusterdf['labels'])
	        #labels = [result[0] for result in [label[0] for label in clusterdf['labels']]]
	        #print(labels)
	        
	        clusterfile = os.path.join(clusterdir, str(clustersize) + '_' + str(nclust) + '.jpg')
	        plotfiles(list(clusterdf['filename']), plot=False, filename=clusterfile)
	        print(clusterfile)
	        #break

def plotfiles(files, plot = True, filename = '', labels=[]):
    nrows = max(2, int(np.ceil(np.sqrt(len(files)))))
    ncols = max(2, int(np.floor(np.sqrt(len(files)))))
    nimgs = nrows * nrows
    if len(files) < 3: nrows = 1
    f, axs = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3), dpi=300)
    for n in range(nimgs):
        row, col = (n)//(ncols), (n)%(ncols)
        if n < len(files):
            try:
                img=mpimg.imread(files[n])
                bbox_props = dict(boxstyle="circle", fc="w", ec="0.5", pad=0.2, alpha=0.9)
                if nrows == 1:
                    axs[n].imshow(img)
                    if len(labels)<=len(files): axs[n].text(0.05, 0.05, labels[n], transform=axs[n].transAxes, 
                                                            bbox={'facecolor':'white', 'alpha':0.8, 'pad':2}, fontsize=6)
                else:
                    axs[row, col].imshow(img)
                    if len(labels)<=len(files): axs[row, col].text(0.05, 0.05, labels[n], transform=axs[row, col].transAxes, 
                                                                   bbox={'facecolor':'white', 'alpha':0.8, 'pad':2}, fontsize=6)
            except:
                pass
        try:
            if nrows == 1:
                axs[n].axis('off')
            else:
                axs[row, col].axis('off')
        except:
            pass
        
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if len(filename) > 0:
        plt.savefig(filename)
    if plot:
        plt.show()


if __name__ == "__main__":
    main()
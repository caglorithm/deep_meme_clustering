import numpy as np
import re, pickle, os
import tqdm

def read_pk(fn):
    with open(fn, 'rb') as fd:
        ret = pickle.load(fd)
    return ret


def write_pk(obj, fn):
    with open(fn, 'wb') as fd:
        pickle.dump(obj, fd)


def get_files(dr, ext='jpg|jpeg|bmp|png'):
    rex = re.compile(r'^.*\.({})$'.format(ext), re.I)
    return [os.path.join(dr,base) for base in os.listdir(dr) if rex.match(base)]

def rename_files(files, imagedir):
	# todo: don't rename files that are already sha256?
	import phashlib as ph
	newfiles = []
	hashes = []
	for fromfile in tqdm.tqdm(files, total=len(files)):
		suffix = fromfile.split('.')[-1]
		sha256hash = ph.sha256_checksum(fromfile)
		tofile = sha256hash + '.' + suffix
		tofile = os.path.join(imagedir, tofile)

		newfiles.append(tofile)
		hashes.append(sha256hash)

		os.rename(fromfile, tofile)
	return newfiles, hashes


def crop_images(df):
    cropped_folder = os.path.join(imagedir, 'cropped/')
    if not os.path.exists(cropped_folder):
        os.makedirs(os.path.dirname(cropped_folder), exist_ok=True)
    if 'cropped_filename' not in df:
        df['cropped_filename'] = None
    for file in tqdm(df.index, total=len(df.index)):
        pil_img=Image.open(df.loc[file]['filename'])
        fhash = df.loc[file]['hash']
        cropped_fname = os.path.join(imagedir, 'cropped/', fhash + '.jpg')

        pil_img.thumbnail((input_size, input_size), Image.ANTIALIAS)
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
        pil_img = pil_img.crop((croplines_x[0], croplines_y[0], croplines_x[1], croplines_y[1]))
        pil_img = pil_img.convert('RGB') 

        pil_img.save(cropped_fname)

        df.loc[file]['cropped_filename'] = cropped_fname

        #plt.imshow(pil_img)
        #plt.show()
    return df
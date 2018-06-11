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


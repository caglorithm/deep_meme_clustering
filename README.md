About
=====
![Info graphic](assets/deep-meme-clustering.png)

Credit
=====
Core code is based on https://github.com/elcorto/imagecluster

Usage
=====
Runs on python 3.6

IF you don't know anything about programming and you have installed all the depencies, you can go into `pipeline.py` and change the `imagedir` directory to where your meme stash is and run the script on the terminal using the command `python pipeline.py` and hope for the best.

Depends these python packages: `numpy, scipy, sklean, pyplot, pickle, tqdm, imagehash, hashlib, PIL, keras, tensorflow`. I know that these are a lot but since this is in development, there are probably some dependencies I will get rid of along the way. 

Pipeline roughly does the following:

    1 - calculates sha256 hash and renames files to hash
    2 - crops out image from meme and copies into ./cropped/
    3 - calculates phash using the imagehash library
    4 - calculates dnn fingerprint using keras and tensorflow
    6 - does the same for cropped versions
    7 - applies a clustering algorithm on fingerprints of cropped images
    8 - plots all members of all clusters into a jpg file and saves results

Results
=====
![graphic](assets/america.jpg)
![graphic](assets/drake.jpg)
![graphic](assets/isthis.jpg)
![graphic](assets/woke.jpg)
![graphic](assets/comic.jpg)

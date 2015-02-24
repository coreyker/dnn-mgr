##Setup Tips!
If you want to run this code on a high performance cluster (HPC), or on a computer where you don't have root access, the following setup instructions may be helpful. They may also be helpful in other cases as well

## Instructions

### HPC Module loading
On the HPC I use there are various modules that must be loaded on demand (e.g., python and cuda). This is done by placing a the file .gbarrc in your home directory with the following line:
```
MODULES=python/2.7.3,cuda/6.5
```
This will probably differ from case to case depending on how your HPC is setup.

### Libraries
Libraries typically get installed to /usr/local/lib, but often times we don't have write access to that location. However, one can download and install libraries locally. First make a directory to hold the libraries (starting from your home directory):
```
mkdir .local
mkdir .local/lib
mkdir .local/include
```
Now download and install the libraries into those folders. The following "pseudocode" demonstrates:
```
url_list = [
"http://www.hdfgroup.org/HDF5/release/obtain5.html", # find your libhdf5 here
"http://sourceforge.net/projects/mad/files/libmad", # find libmad here (only need if you plan to read mp3 files)
"http://www.mega-nerd.com/SRC/download.html", # libsamplerate
"http://www.mega-nerd.com/libsndfile/#Download", #libsndfile
"http://pyyaml.org/download/libyaml" #libyaml
]

for url in url_list:
	wget url
	tar xvfz package_name.tar.gz
	cd package_name
	./configure --prefix=$HOME/.local
	make
	make install
```
the --prefix flag tells the Makefile to install the library locally. The hdf5 library might need additional configuration information (e.g., in order to allow multiple read access, which is required).


### Python virtual environment
I setup a python virtual environment as follows

```
pip install virtualenv
mkdir venv
cd venv
virtualenv venv
```

After that you must call 
```
source venv/bin/activate
```
each time you login to activate the virtual environment. I added the above line to my .bashrc file so it happens automagically.


Now it is time to install the python packages, but first you should create a file called .numpy-site.cfg in your home directory with the following lines:
```
[sndfile]
library_dirs = $HOME/.local/lib
include_dirs = $HOME/.local/include
[hdf5]
library_dirs = $HOME/.local/lib
include_dirs = $HOME/.local/include
[samplerate]
library_dirs = $HOME/.local/lib
include_dirs = $HOME/.local/include
```
This tells numpy's distutils where to find your locally installed libraries.

Now we can install the numpy packages (using pip, or from github sources, etc). The packages I have installed are:
```
numpy, 
scipy, 
theano, 
pylearn2, 
pytables, 
numexpr, 
cython, 
pyyaml, 
ipython, 
sklearn, 
matplotlib, 
scikits.audiolab, 
scikits.samplerate, 
pymad (if you need to read mp3's) 
```
You might try installing these first with "pip install package_name", and if that fails, download the source code and run "python setup.py install". 

### Theano setup
If you want theano to use your GPU (and you probably do if you have one), create a file called .theanorc in your home directory with the following lines:
```
[global]
floatX = float32
device = gpu0

[nvcc]
fastmath = True
```
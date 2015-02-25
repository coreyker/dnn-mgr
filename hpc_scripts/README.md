##Setup Tips!
If you want to run this code on a high performance cluster (HPC), or on a computer where you don't have root access, the following setup instructions may be helpful. They may also be helpful in other cases as well.

## Instructions

### HPC Module loading
On the HPC I use there are various modules that must be loaded on demand (e.g., python and cuda). This is done by placing a file called .gbarrc in your home directory with the following lines:
```
MODULES=python/2.7.3,cuda/6.5
```
This will probably differ from case to case depending on how your HPC is setup.

### Python virtual environment
I first setup a python virtual environment (which provides a clean copy of python without any packages) as follows:
```
mkdir venv
virtualenv venv
```

After this you must call:
```
source venv/bin/activate
```
every time you login to activate the virtual environment. I added the above line to the end of my .bashrc file so it happens automagically.

### Installation of libraries
Libraries typically get installed to /usr/local/lib, but often times we don't have write access to that location. However, we can download and install libraries locally. First make a directory to hold the libraries (starting from your home directory):
```
mkdir .local
mkdir .local/lib
mkdir .local/include
```

Add this location to your LD_LIBRARY_PATH too
```
export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH
```
(that line can be added to .bashrc so that it is excuted in every new shell).

Now download and install the libraries. The following "pseudocode" demonstrates the basic process (you will need to track down the correct urls for your platform):
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
the --prefix flag tells the Makefile to install the library locally (and therefore it is crucial that you include it).

Note on the HDF5 library: This library may already exist on your system, however it may be a good idea to install a local copy anyway, because some older versions do not support multiple read access (which is required), leading to an error (if you see 'FILE_OPEN_POLICY=strict' printed out, then you have probably just encountered this error). Also, to make sure Pytables links with the correct HDF5 library add the following to your .bashrc file:
```
export HDF5_DIR=$HOME/.local
```

Note on libmad: In the Makfile for libmad you may need to remove the "-fforce-mem" flag, which gcc no longer supports.

### Installation of Python packages
After the libraries have been installed, one can start installing the necessary python packages. First you should create a file called .numpy-site.cfg in your home directory with the following lines:
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
This tells numpy's distutils where to find your locally installed libraries. You will probably have to replace $HOME with the actual path to your home directory (e.g., "/home/a/user") since it does not seem to get properly exported as an environment variable by numpy.

Now we can install the python packages (using pip, or from github sources, etc). The packages I have installed are:
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

You might first try installing these using the "requirements.txt" file in this folder:
```
pip install -r path/to/requirements.txt
```
If that doesn't work you can try installing these with "pip install package_name", and finally, if that fails, download the source code for each module and run "python setup.py build" followed by "python setup.py install". 

### Theano setup
If you want theano to use your GPU (and you probably do if you have one), create a file called .theanorc in your home directory with the following lines:
```
[global]
floatX = float32
device = gpu0

[nvcc]
fastmath = True
```
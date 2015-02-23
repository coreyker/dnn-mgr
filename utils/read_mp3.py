# from http://www.iro.umontreal.ca/~eckdoug/python_for_courses/readmp3.py

import numpy as N
import sys,os

try  :
    import mad
except :
    print 'You need to install pymad from http://spacepants.org/src/pymad'
    
def read_mp3(fn) :
    """Reads an mp3 file using madplay library.  Returns (x,fs)
    You need pymad from http://spacepants.org/src/pymad. 
    To install pymad you will also need libmad"""
    mf=mad.MadFile(fn)
    fs=mf.samplerate()       
    mode_to_channel = { mad.MODE_SINGLE_CHANNEL:1, 
                        mad.MODE_DUAL_CHANNEL:2,
                        mad.MODE_JOINT_STEREO:2,
                        mad.MODE_STEREO:2}
    channels = mode_to_channel[mf.mode()]
    secs=int(mf.total_time()/1000.0)
    samps_per_channel = (mf.samplerate() * mf.total_time()) /1000.0
    samples = samps_per_channel * channels
    dat = N.zeros(((samples+fs)*2),'int16')  #we store 1 second at end of song
    st=0
    buf=mf.read()
    while buf :
        shortbuf = N.fromstring(buf,'int16')
        dat[st:st+len(shortbuf)]=shortbuf
        st += len(shortbuf)
        buf=mf.read()
    dat=dat[:st]
    x=dat/float(32768)

    #it seems that we always get a stereo channel even for mono files, 
    #but that values are duplicated e.g. [10 , 10, -4 , -4]
    #I don't have enough mono files around to check so this trap is left in place
    if channels==1 and len(dat) > samples * 1.5 :
        x=x[0::2] #grab every other sample
    if channels>1 :
        x=x.reshape(-1,channels)
    return (x,fs,'int16')
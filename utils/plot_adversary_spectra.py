import numpy as np
import scikits.audiolab as audiolab
from test_adversary import winfunc, compute_fft
from matplotlib import pyplot as plt


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='')
    parser.add_argument('--true_file')
    parser.add_argument('--adversary')
    args = parser.parse_args()

    # load sndfile 
    x,_,_ = audiolab.wavread(args.true_file)
    x_adv,_,_ = audiolab.wavread(args.adversary)
    
    # STFT
    X = compute_fft(x)[0][:,:513]
    X_adv = compute_fft(x_adv)[0][:,:513]

    rng = 5+np.arange(100)
    X = X[rng,:]
    X_adv = X_adv[rng,:]

    # Plotting...
    plt.ion()
    plt.figure()

    plt.subplot(2,3,1)
    plt.imshow(20*np.log10(X), extent=[0,11.025,len(rng),0])
    plt.axis('tight')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Time frame')

    plt.subplot(2,3,2)
    plt.imshow(20*np.log10(X_adv), extent=[0,11.025,len(rng),0])
    plt.axis('tight')
    plt.xlabel('Frequency (kHz)')

    plt.subplot(2,3,3)
    plt.imshow(20*np.log10(np.abs(X_adv-X)), extent=[0,11.025,len(rng),0])
    plt.axis('tight')
    plt.xlabel('Frequency (kHz)')

    plt.subplot(2,1,2)
    N = 10
    x_range = np.arange(513)/513.*(22.050/2)
    plt.plot(x_range, 20*np.log10(X[N,:]), color=(.4,.6,1,0.8), linewidth=2)
    
    plt.plot(x_range, 20*np.log10(X_adv[N,:]), color=(0,0,0,1), linewidth=1)
    
    plt.plot(x_range, 20*np.log10(np.abs(X[N,:]-X_adv[N,:])), '-', color=(1,0.6,0.1,0.6), linewidth=2)
    plt.axis('tight')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Magnitude (dB)')

    #plt.savefig('adversary_spectra.pdf', format='pdf')
from sklearn.decomposition import FastICA
import numpy
import matplotlib.pyplot as plt
import time
from scipy.io import wavfile
from Processor import Processor
from sklearn import preprocessing


class ICA:
    def __init__(self, mixed_sig, no_of_components):
        self.mixed_signal = mixed_sig
        self.num_components = no_of_components

    def train(self, opt, proc):
        print("Training ICA...")
        time_start = time.time()
        # whitening necessary

        if opt == 2:
            ica = FastICA(n_components=self.num_components, whiten=False)
        elif opt == 3:
            ica = FastICA(n_components=self.num_components, fun='exp')
        elif opt == 4:
            ica = FastICA(n_components=self.num_components, fun='cube')
        elif opt == 5:
            self.mixed_signal = preprocessing.scale(self.mixed_signal)
            ica = FastICA(n_components=self.num_components)
        else:
            ica = FastICA(n_components=self.num_components)

        # self.mixed_signal = preprocessing.scale(self.mixed_signal)
        reconstruct_signal = ica.fit_transform(self.mixed_signal)
        mixing_matrix = ica.mixing_

        time_stop = time.time() - time_start
        print("Training Complete under {} seconds".format(time_stop))

        if opt == 2:
            assert numpy.allclose(self.mixed_signal, numpy.dot(reconstruct_signal, mixing_matrix.T))
        else:
            assert numpy.allclose(self.mixed_signal, numpy.dot(reconstruct_signal, mixing_matrix.T) + ica.mean_)

        if opt == 2:
            remixed_mat = numpy.dot(reconstruct_signal, mixing_matrix.T)
        else:
            remixed_mat = numpy.dot(reconstruct_signal, mixing_matrix.T) + ica.mean_

        for q in range(len(remixed_mat.T)):
            y = remixed_mat.T[q] - self.mixed_signal.T[q]
            q = sum([i**2 for i in y])
            print("Residual value: {}".format(q))

        # for i in range(len(remixed_mat.T)):
        #     wavfile.write('aaa.wav', proc[0], numpy.asarray(remixed_mat.T[0], dtype=numpy.int16))
        return reconstruct_signal, mixing_matrix, remixed_mat

    def create_audio(self, reconstructed_mat, s_hat_names, remixed_mat, recon_names, rates):
        print("Creating recovered audio...")
        for count in range(len(s_hat)):
            wavfile.write(s_hat[count], rates[count], reconstructed_mat.T[count])
            wavfile.write(recon_names[count], rates[count], numpy.asarray(remixed_mat.T[count], dtype=numpy.int16))

    def plot_audio(self, mixed_sig, recovered_sig):
        print("Plotting graphs...")
        models = [mixed_sig, recovered_sig]
        names = ['Observations (mixed signal)',
                 'ICA recovered signals']
        colors = ['red', 'steelblue', 'orange', 'green', 'brown']

        plt.figure()

        for ii, (model, name) in enumerate(zip(models, names), 1):
            plt.subplot(2, 1, ii)
            plt.title(name)
            for sig, color in zip(model.T, colors):
                plt.plot(sig, color=color)

        plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
        plt.show()


if __name__ == '__main__':
    audios = ['mic1.wav', 'mic2.wav', 'mic3.wav', 'mic4.wav', 'mic5.wav']
    s_hat = ['shat1.wav', 'shat2.wav', 'shat3.wav', 'shat4.wav', 'shat5.wav']
    recon = ['recon1.wav', 'recon2.wav', 'recon3.wav', 'recon4.wav', 'recon5.wav']

    processor = Processor(audios)
    processor.generate_wav_data()

    # Options: 1: default, 2: whitening off, 3: exp function, 4: cube function, 5: centering
    option = 1
    ica_ = ICA(processor.mixed_signal, 5)
    recon_matrix, mixing_matrix, remixed = ica_.train(option, processor.rates)
    ica_.create_audio(recon_matrix, s_hat, remixed, recon, processor.rates)
    ica_.plot_audio(processor.mixed_signal, recon_matrix)

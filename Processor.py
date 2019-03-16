from scipy.io import wavfile
import matplotlib.pyplot as plt
import math
import numpy
from sklearn import preprocessing


class Processor:

    def __init__(self, audio_list):
        self.audio_array = audio_list

    def generate_wav_data(self):
        smallest_shape = math.inf
        wav_data = []
        rate_data = []
        for audio in self.audio_array:
            rate, voice = wavfile.read(audio)
            wav_data.append(voice)
            rate_data.append(rate)
            m, = voice.shape
            if m < smallest_shape:
                smallest_shape = m

        for count in range(len(wav_data)):
            wav_data[count] = wav_data[count][:smallest_shape]
            # wav_data[count] = preprocessing.scale(wav_data[count])

        mixed = numpy.c_[wav_data[0], wav_data[1], wav_data[2], wav_data[3], wav_data[4]]

        # print(len(data[0]))
        # mixing_matrix = numpy.random.uniform(size=(len(self.audio_array), len(self.audio_array)))
        # mixing_matrix = mixing_matrix / mixing_matrix.sum(axis=0)

        # mixed = numpy.dot(data, mixing_matrix)

        # plt.title("mixed")
        # plt.plot(numpy.arange(smallest_shape) / rate_data[0], self.mixed_signal)
        # plt.show()
        # print(len(mixed))
        # print(len(mixed[0]))
        self.wav_signal = wav_data
        self.rates = rate_data
        # self.original_signal = data
        self.mixed_signal = mixed

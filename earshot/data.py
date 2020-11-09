import librosa
import os
import pandas as pd

from earshot.audio import *
from numpy import ones
from scipy import sparse
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

# TODO include pass for precomputed input


def pad(data_2d, ref_shape, pad_val=-9999):
    '''
    Pads a 2D matrix "data" with the pad_value to make it have shape ref_shape.
    All the padded values occur "off the end" of the data array, so that the
    original array occupies the upper left of the resulting matrix.

    Refer to pad_nans method to pad with nan values
    '''
    padded_data = pad_val*ones(ref_shape)
    padded_data[:data_2d.shape[0], :data_2d.shape[1]] = data_2d
    return padded_data


def spectro_calc(audio_path):
    # TODO add pass for parameters
    sig = librosa.core.load(audio_path, sr=22050)[0]
    sig = librosa.effects.trim(sig, frame_length=32, hop_length=16)[0]
    spec = spectrogram(sig, samp_rate=22050, dimension=256,
                       frame_len=10, frame_shift=10)
    return np.transpose(spec).astype(np.float32)


class Manifest(object):
    '''
    Generates data manifest when passed a folder containing audio files with
    naming structure of word_talker.wav
    '''
    def __init__(self, audio_dir):
        '''
        audio_dir = path to directory containing desired audio files
        '''
        path_list = list()
        file_list = list()
        for (dirpath, dirnames, filenames) in os.walk(audio_dir):
            path_list += [os.path.join(dirpath, file) for file in filenames if (
                file.endswith('.wav') or file.endswith('.WAV'))]
            file_list += [file for file in filenames if (
                file.endswith('.wav') or file.endswith('.WAV'))]
        manifest = []
        for i in range(len(file_list)):
            word, talker = file_list[i][:-4].split('_')
            manifest.append([path_list[i], word, talker])
        self.manifest = pd.DataFrame(
            manifest, columns=['Path', 'Word', 'Talker'])
        uniques = self.manifest.nunique()
        print("Your dataset contains {} unique word(s) and {} unique talker(s).".format(
              uniques[1], uniques[2]))
        print("\n")
        print("Number of utterances by a unique talker.")
        print(self.manifest.groupby('Word').count()['Talker'])
        print("\n")
        print("Number of unique word utterances by each talker.")
        print(self.manifest.groupby('Talker').count()['Word'])

    def generate_srvs(self, target='Talker', target_len=300, target_on=10):
        '''
        target = manifest column containing desired targets
        target_len = desired length of target vector
        target_on = desired number of ones in target vector
        '''
        assert target in [
            'Talker', 'Word'], "Please use 'Word' or 'Talker' as your target."
        target_items = self.manifest[target].unique()
        target_pairs = []
        for i in range(len(target_items)):
            target_pairs.append([target_items[i],
                                 sparse.random(1, target_len,
                                               density=target_on/target_len, data_rvs=ones).toarray()[0]])
        target_pairs = pd.DataFrame(target_pairs, columns=[target, 'Target'])
        print("There are {} unique target patterns.".format(
            len(target_pairs['Target'])))
        self.manifest = self.manifest.merge(target_pairs)

    def calc_spec(self):
        # place holder for input
        self.manifest['Input'] = None
        for i in tqdm(range(len(self.manifest))):
            self.manifest.loc[i, 'Input'] = spectro_calc(
                self.manifest['Path'][i])


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, df, batch_size=32, pad_value=-9999):
        '''
        df = manifest dataframe
        batch_size = desired batching size
        pad_value = desired value to pad data with, pads to max of each batch
        '''
        self.df = df
        self.batch_size = batch_size
        self.pad_value = pad_value
        self.targets = np.array(self.df['Target'].tolist()).astype(np.float32)
        self.path_list = self.df['Path'].tolist()
        self.indexes = np.arange(len(self.path_list))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.path_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_pairs_temp = []
        for k in indexes:
            list_pairs_temp.append((self.path_list[k], self.targets[k]))

        # Generate data
        X, y = self.__data_generation(list_pairs_temp)

        return X, y

    def __data_generation(self, list_pairs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        spec = [spectro_calc(path[0]) for path in list_pairs_temp]

        M = max(len(a) for a in spec)
        padded_spec = [pad(s, (M, s.shape[1]), pad_val=self.pad_value)
                       for s in spec]
        X = np.empty((self.batch_size, M, spec[0].shape[1]))
        y = np.empty((self.batch_size, len(list_pairs_temp[0][1])), dtype=int)

        for i, pair in enumerate(list_pairs_temp):
            X[i, ] = padded_spec[i]
            y[i] = pair[1]

        return X, y

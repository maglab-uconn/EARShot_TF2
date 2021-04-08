import librosa
import math
import os
import pandas as pd
import numpy.matlib as npm
import matplotlib.pyplot as plt

from earshot.phonology import *
from earshot.audio import *
from numpy import ones,sqrt,dot,vstack
from scipy import sparse, fftpack
from tensorflow.keras.utils import Sequence
from tqdm import tqdm,trange

# TODO include a pass for precomputed input to the Batch Generator / Manifest

def closest_power(x):
    # credit user koffein
    # https://stackoverflow.com/questions/28228774/find-the-integer-that-is-closest-to-the-power-of-two/28229369
    possible_results = math.floor(math.log(x, 2)), math.ceil(math.log(x, 2))
    return min(possible_results, key= lambda z: abs(x-2**z))

def vector_cosine(x,y):
    '''
    Slow but correct - can improve efficiency later.
    '''
    mag_x = sqrt(dot(x,x))
    mag_y = sqrt(dot(y,y))
    return dot(x,y)/(mag_x*mag_y)


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


    def generate_category_dict(self, pronounce_text):
        '''
        For each word, determines its cohorts, rhymes, DAS neighbors, and unrelated words from the
        pronunciation file.

        Make sure your pronunciation file matches the one used in training, or there will be a
        mismatch between the category dict and the manifest you used to train.
        '''
        pronounce_df = pd.read_csv(pronounce_text, sep='\t', names=['Word','Pronounce'])
        self.category_dict = {}
        for idx, row in tqdm(pronounce_df.iterrows(), total=pronounce_df.shape[0]):
            self.category_dict[row['Word']] = {'Target': [], 'Cohort': [], 'Rhyme': [],
                                               'DAS_Neighborhood': [], 'Unrelated': []}

            for index, compare_row in pronounce_df.iterrows():
                compare_word_indx = compare_row['Word']

                # assume word is unrelated by default
                if row['Word'] == compare_row['Word']:
                    # word is the target; move on
                    self.category_dict[row['Word']]['Target'].append(compare_word_indx)
                    continue

                # words cannot be both cohorts and rhymes
                if are_cohorts(row['Pronounce'].split('.'), compare_row['Pronounce'].split('.')):
                    # words are in the same cohort
                    self.category_dict[row['Word']]['Cohort'].append(compare_word_indx)
                    # they may also be neighbors
                    if are_neighbors(row['Pronounce'].split('.'),compare_row['Pronounce'].split('.')):
                        self.category_dict[row['Word']]['DAS_Neighborhood'].append(compare_word_indx)
                    continue
                elif are_rhymes(row['Pronounce'].split('.'), compare_row['Pronounce'].split('.')):
                    # words are rhymes and (by this def'n) automatically neighbors
                    self.category_dict[row['Word']]['Rhyme'].append(compare_word_indx)
                    self.category_dict[row['Word']]['DAS_Neighborhood'].append(compare_word_indx)
                    continue

                # words may be neighbors but not cohorts or rhymes
                if are_neighbors(row['Pronounce'].split('.'),compare_row['Pronounce'].split('.')):
                    self.category_dict[row['Word']]['DAS_Neighborhood'].append(compare_word_indx)
                    continue

                # if we made it here, they must be unrelated
                self.category_dict[row['Word']]['Unrelated'].append(compare_word_indx)

    def generate_srvs(self, target='Word', target_len=300, target_on=10):
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

    def add_w2v_targets(self, path_to_w2vpickle):
        '''
        Currently exclusive with SRV targets.
        '''
        w2v_df = pd.DataFrame.from_dict(pd.read_pickle(path_to_w2vpickle))

        w2v_list = []
        for word in w2v_df.keys():
            w2v_list.append((word.upper(), w2v_df[word].to_list()))

        w2v_df = pd.DataFrame(w2v_list, columns=['Word', 'W2V Target'])

        w2v_training_df = self.manifest[self.manifest['Word'].isin(w2v_df['Word'])]
        self.manifest = w2v_training_df.merge(w2v_df)
        return self.manifest

    def calc_spec(self, subset=False):
        training_df = self.manifest
        try:
            training_df = training_df[training_df['Word'].isin(subset['Word'].tolist())]
        except:
            print("No subset given, using full manifest.")
        # place holder for input
        training_df['Input'] = None
        for index,row in tqdm(training_df.iterrows()):
            training_df.loc[index, 'Input'] = AudioTools(training_df['Path'][index]).sgram(0.010,0.010,8000)

        training_df['Padded Input'] = None
        # get max len of batch
        M = max(len(a) for a in training_df['Input'].tolist())
        for index,row in tqdm(training_df.iterrows()):
            s = training_df.loc[index, 'Input']
            training_df.loc[index, 'Padded Input'] = pad(s, (M, s.shape[1]))

        try:
            # pad output
            training_df['Target'] # force to except if not present
            training_df['Padded Target'] = None
            for index,row in tqdm(training_df.iterrows()):
                padded_target = npm.repmat(training_df.loc[index, 'Target'], M, 1)
                training_df.loc[index, 'Padded Target'] = padded_target
            try:
                # pad w2v output
                training_df['W2V Target'] # force to except if not present
                training_df['Padded W2V Target'] = None
                for index,row in tqdm(training_df.iterrows()):
                    padded_target = npm.repmat(training_df.loc[index, 'W2V Target'], M, 1)
                    training_df.loc[index, 'Padded W2V Target'] = padded_target
                return training_df
            except:
                return training_df
        except:
            # pad w2v output
            training_df['Padded W2V Target'] = None
            for index,row in tqdm(training_df.iterrows()):
                padded_target = npm.repmat(training_df.loc[index, 'W2V Target'], M, 1)
                training_df.loc[index, 'Padded W2V Target'] = padded_target
            return training_df

    def gen_predict(self, Talker, subset=False, num_samp=100, pad_value=-9999, window_len=0.010, skip_len=0.010, max_freq=8000):
        # pull a number of random items from a specific talker for model prediction
        predict_df = self.manifest[self.manifest['Talker'].isin(Talker)]
        try:
            predict_df = predict_df[predict_df['Word'].isin(subset['Word'].tolist())]
        except:
            print("No subset given, using full manifest.")
        predict_df = predict_df.sample(num_samp)
        predict_df['Input'] = None
        for index, row in tqdm(predict_df.iterrows()):
            predict_df.loc[index, 'Input'] = AudioTools(predict_df['Path'][index]).sgram(window_len,skip_len,max_freq)
        M = max(len(a) for a in predict_df['Input'].tolist())

        predict_df['Padded Input'] = None
        for index, row in predict_df.iterrows():
            s = predict_df.loc[index, 'Input']
            predict_df.loc[index, 'Padded Input'] = pad(s, (M, s.shape[1]), pad_val=pad_value)
        return predict_df



class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, df, batch_size=32, pad_value=-9999, return_seq=True, spec_calc='scipy'):
        '''
        df = manifest dataframe
        batch_size = desired batching size
        pad_value = desired value to pad data with, pads to max of each batch
        return_seq = determines generated label structure, related to model architecture
        spec_calc = Pass 'scipy' or 'librosa' to calc specs from wav files. Pass 'padding' if you are using precomputed values and only need to batch pad.
        '''
        assert spec_calc in [
            'scipy', 'librosa', 'padding'], "Please use 'scipy' or 'librosa' to calculate spectrograms."
        self._spec_calc = spec_calc
        self.df = df
        self.batch_size = batch_size
        self.pad_value = pad_value
        self.targets = np.array(self.df['Target'].tolist()).astype(np.float32)
        self.path_list = self.df['Path'].tolist()
        self.indexes = np.arange(len(self.path_list))
        self.return_seq = return_seq
        if self._spec_calc == 'padding':
            self.inputs = self.df['Input'].tolist()
            self.indexes = np.arange(len(self.inputs))

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self._spec_calc == 'padding':
            return int(np.floor(len(self.inputs) / self.batch_size))
        else:
            return int(np.floor(len(self.path_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_pairs_temp = []
        if self._spec_calc == 'padding':
            for k in indexes:
                list_pairs_temp.append((self.inputs[k], self.targets[k]))
        else:
            for k in indexes:
                list_pairs_temp.append((self.path_list[k], self.targets[k]))

        # Generate data
        X, y = self.__data_generation(list_pairs_temp)

        return X, y

    def __data_generation(self, list_pairs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        # calculate spectrograms for each item in batch
        if self._spec_calc == 'librosa':
            spec = [ spectro_calc(path[0]) for path in list_pairs_temp ]
        elif self._spec_calc == 'scipy':
            spec = [ AudioTools(path[0]).sgram(0.010,0.010,8000) for path in list_pairs_temp ]
        elif self._spec_calc == 'padding':
            spec = [ input[0] for input in list_pairs_temp ]
        # get max len of batch
        M = max(len(a) for a in spec)
        # pad all specs in batch to max length
        padded_spec = [ pad(s, (M, s.shape[1]), pad_val=self.pad_value)
                       for s in spec ]
        X = np.empty((self.batch_size, M, spec[0].shape[1]), dtype=np.float32)
        # pad targets if LSTM layer has return_sequence=True
        if self.return_seq:
            y = np.empty((self.batch_size, M, len(list_pairs_temp[0][1])), dtype=np.float32)
            targets = [ npm.repmat(pair[1],spec[i].shape[0],1) for i,pair in enumerate(list_pairs_temp) ]
            padded_targets = [ pad(i,(M,i.shape[1]),pad_val=self.pad_value) for i in targets ]
            for i, pair in enumerate(list_pairs_temp):
                X[i, ] = padded_spec[i]
                y[i, ] = padded_targets[i]
        else:
            y = np.empty((self.batch_size, len(list_pairs_temp[0][1])), dtype=np.float32)

            for i, pair in enumerate(list_pairs_temp):
                X[i, ] = padded_spec[i]
                y[i] = pair[1]

        return X, y

class AudioTools(object):
    '''
    A suite of tools to generate SPL spectrograms for a specified audio file.

    The reference for the majority of these functions can be found here:
    https://courses.engr.illinois.edu/ece590sip/sp2018/spectrograms1_wideband_narrowband.html

    '''
    def __init__(self, audiopath):
        '''
        audiopath: path to desired audio
        '''
        sig, self.fs = librosa.core.load(audiopath)
        self.signal = librosa.effects.trim(
            sig, frame_length=32, hop_length=16)[0]

    def _enframe(self, signal, skip_len, window_len):
        # generate time windows for subsequent stft calculations
        # w = 0.54*np.ones(L)
        # for n in range(0,L):
        #   w[n] = w[n] - 0.46*math.cos(2*math.pi*n/(L-1))
        w = np.hamming(window_len)
        frames = []
        nframes = 1+int((len(signal)-window_len)/skip_len)
        for t in range(0, nframes):
            frames.append(
                np.copy(signal[(t*skip_len):(t*skip_len+window_len)])*w)
        return frames

    def _stft(self, frames, n_fft, Fs):
        stft_frames = [fftpack.fft(x, n_fft) for x in frames]
        freq_axis = np.linspace(0, Fs, n_fft)
        return(stft_frames, freq_axis)

    def _stft2level(self, stft_spectra, max_freq_bin):
        magnitude_spectra = [np.abs(x) for x in stft_spectra]
        max_magnitude = max([max(x) for x in magnitude_spectra])
        min_magnitude = max_magnitude / 1000.0
        for t in range(0, len(magnitude_spectra)):
            for k in range(0, len(magnitude_spectra[t])):
                magnitude_spectra[t][k] /= min_magnitude
                if magnitude_spectra[t][k] < 1:
                    magnitude_spectra[t][k] = 1
        # convert to SPL spectra
        level_spectra = [20*np.log10(x[0:max_freq_bin])
                         for x in magnitude_spectra]
        return level_spectra

    def sgram(self, window_len, skip_len, max_freq, n_fft = False):
        '''
        window_len: length of window in seconds
        skip_len: length to skip in seconds
        max_freq: maximum desired frequency
        n_fft: False to automatically determine number of frequency bins based on window length, can override with number brought to the nearest power of 2
        '''
        if n_fft:
            n_fft = int(closest_power(n_fft))
            self.n_fft = pow(2, n_fft)
        else:
            self.n_fft = pow(2, int(math.log(int(self.fs*window_len), 2) + 0.5))  # calc NFFT suitable for window_len
        window_len = int(window_len*self.fs)  # convert to length in samples
        skip_len = int(skip_len*self.fs)  # convert to length in samples
        self.frames = self._enframe(self.signal, skip_len, window_len)
        spectra, self.freq_axis = self._stft(self.frames, self.n_fft, self.fs)
        self.spl_spec = np.array(self._stft2level(
            spectra, int(max_freq*self.n_fft/self.fs)))
        self.max_time = len(self.frames)*skip_len/self.fs
        self.max_freq = max_freq
        # returns max scaled spl spectrogram
        return self.spl_spec / self.spl_spec.max()

    def plot_spec(self):
        try:
            plt.imshow(self.spl_spec.transpose(),
                       origin='lower',
                       extent=(0, self.max_time, 0, self.max_freq),
                       aspect='auto')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.show()
        except:
            print("Please run the sgram method before plotting.")

class Prediction(object):
    '''
    Class for generating cosine simularity dfs.
    '''
    def __init__(self, fresh_model, checkpoint_path, prediction_df, full_manifest):
        '''
        fresh_model: untrained earshot model to load weights
        checkpoint_path: path to model checkpoint to evaluate
        prediction_set: set of data for prediction, easily generated using gen_predict method of the Manifest class
        full_manifest: full manifest df from Manifest class (.manifest method) that contains full set of words and targets
        '''
        self.prediction_df = prediction_df

        fresh_model.load_weights(checkpoint_path)

        self.predictions = fresh_model.predict(np.array(self.prediction_df['Padded Input'].tolist()),
                                               batch_size = 32,
                                               verbose=1)

        # creat df of unique words with associated labels
        self.unique_label_df = full_manifest[~full_manifest['Word'].duplicated()][['Word','Target']].reset_index()

        self.cosine_sim_dict = self._calc_cosines()


    def _calc_cosines(self):
        '''
        Calculates over-time cosine simularity of each prediction to every word in the training vocabulary.
        Slow but understandable and hopefully also correct.

        Returns a nested dictionary such that similarity_dict[w1][w2] is the similarity of the prediction of
        the model for w1 to the semantic pattern associated with w2.
        '''
        similarity_dict = {}
        # self.predictions is num_test_words x total time steps (+ pad) x semantic dim
        total_l = self.predictions.shape[1]
        for pred_dx in trange(self.predictions.shape[0]):
            pred_word = self.prediction_df['Word'].iloc()[pred_dx]
            similarity_dict[pred_word] = {}
            for lex_dx in range(len(self.unique_label_df)):
                test_word = self.unique_label_df['Word'].iloc()[lex_dx]
                similarity_dict[pred_word][test_word] = np.zeros(total_l)
                for t in range(total_l):
                    similarity_dict[pred_word][test_word][t] = vector_cosine(self.predictions[pred_dx,t,:],self.unique_label_df['Target'].iloc()[lex_dx])
        return similarity_dict


    def calc_category_cosine(self, target_word, category_dict):
        '''
        Computes over-time cosine similarities of a given target word (which must be in the lexicon and hence
        in the cosine similarity dictionary!) to different categories of word, definted by the category dict
        (i.e., Target, Cohort, Rhyme, etc.)
        '''
        cosine_dict = {}
        if '_' in target_word:
            word_talker = target_word.split('_')
            for k in category_dict[word_talker[0]].keys():
                cosine_dict[k] = [self.cosine_sim_dict[target_word][x] for x in category_dict[word_talker[0]][k]]
        else:
            for k in category_dict[target_word].keys():
                cosine_dict[k] = [self.cosine_sim_dict[target_word][x] for x in category_dict[target_word][k]]
        return cosine_dict


    def calc_category_flows(self,category_dict):
        '''
        Calculates grand mean category flows to targets, cohorts, rhymes, DAS neighbors, and unrelated words.
        '''
        grand_mean_dict = {}
        # initialize the grand mean dict
        first_word = list(category_dict.keys())[0]
        for k in category_dict[first_word]:
            grand_mean_dict[k] = list()
        # now loop over all words we made predictions for
        for word in self.cosine_sim_dict.keys():
            cat_dict = self.calc_category_cosine(word,category_dict)
            # some categories may be empty, so don't add them
            for k in cat_dict:
                if len(cat_dict[k]) != 0:
                    grand_mean_dict[k].append(cat_dict[k])
        # now store in a data frame
        grand_mean_df = pd.DataFrame()
        for k in grand_mean_dict:
            grand_mean_df[k] = vstack(grand_mean_dict[k]).mean(axis=0)
        return grand_mean_df


    """
    def _calc_cosine(self):

        # calculate cosine similarity of each prediction
        # at each time step
        # to every word in the training vocab

        # compute cosine simularity at each timestep against each word in vocab
        # output rows = timestep, columns = word, cell value = cosine simularity
        # this method currently overrides duplicate targets in dictionary with most recent in set
        # TODO tag duplicates before assigning key in dict; cat cat1 cat2 etc
        simularity_dict = {}
        Y = np.array(self.unique_label_df['Target'].to_list()).T
        for i in trange(len(self.predictions)):
            x = self.predictions[i][:self.prediction_df['Padded Input'].iloc()[i].shape[0]]
            cosine_sim = []
            for step in x:
                cosine_sim.append(np.dot(Y.T,step)/(np.sqrt(np.dot(step,step))*np.sqrt((Y*Y).sum(axis=0))))
            simularity_df = pd.DataFrame(np.array(cosine_sim).reshape(x.shape[0],Y.shape[1]))
            simularity_df.columns = np.array(self.unique_label_df['Word'].to_list()).T
            if self.prediction_df['Talker'].unique().shape[0] > 1:
                simularity_dict["{0}_{1}".format(self.prediction_df['Word'].iloc()[i], self.prediction_df['Talker'].iloc()[i])] = simularity_df
            else:
                simularity_dict["{0}".format(self.prediction_df['Word'].iloc()[i])] = simularity_df
        return simularity_dict


    def calc_category_cosine(self, target_word, category_dict):
        '''
        target_word: select word to compute similarities to
        category_dict: pass a category dictionary generated from Manifest.generate_category_dict()
        '''
        cosine_category_df = pd.DataFrame()
        if '_' in target_word:
            word_talker = target_word.split('_')
            for i in list(category_dict[word_talker[0]].keys()):
                cosine_category_df[i] = self.cosine_sim_dict[target_word][category_dict[word_talker[0]][i]].mean(axis=1)
        else:
            for i in list(category_dict[target_word].keys()):
                cosine_category_df[i] = self.cosine_sim_dict[target_word][category_dict[target_word][i]].mean(axis=1)
        return cosine_category_df


    def plot_category_cosine(self, target_word, cosine_category_df):
        '''
        target_word: select word to plot from .cosine_sim_dict.keys()
        category_dict: pass the output of .calc_category_cosine()
        '''
        lines = cosine_category_df.plot.line(xlabel='Time Steps',ylabel='Cosine Simularity', ylim=(0,1), title=target_word)
        return plt.show()


    def calc_cosine_grand_mean(self,category_dict):
        '''
        Category cosines/flows, averaged over all targets in the training lexicon.
        '''
        target_df = pd.DataFrame()
        cohort_df = pd.DataFrame()
        rhyme_df = pd.DataFrame()
        neighborhood_df = pd.DataFrame()
        unrelated_df = pd.DataFrame()

        for word in tqdm(self.cosine_sim_dict.keys()):
            if '_' in word:
                word_talker = word.split('_')
                category_df = pd.DataFrame()
                for i in list(category_dict[word_talker[0]].keys()):
                    category_df[i] = self.cosine_sim_dict[word][category_dict[word_talker[0]][i]].mean(axis=1)
                target_df[word] = category_df['Target']
                cohort_df[word] = category_df['Cohort']
                rhyme_df[word] = category_df['Rhyme']
                neighborhood_df[word] = category_df['DAS_Neighborhood']
                unrelated_df[word] = category_df['Unrelated']
            else:
                category_df = pd.DataFrame()
                for i in list(category_dict[word].keys()):
                    category_df[i] = self.cosine_sim_dict[word][category_dict[word][i]].mean(axis=1)
                target_df[word] = category_df['Target']
                cohort_df[word] = category_df['Cohort']
                rhyme_df[word] = category_df['Rhyme']
                neighborhood_df[word] = category_df['DAS_Neighborhood']
                unrelated_df[word] = category_df['Unrelated']

        mean_cosine_df = pd.DataFrame()
        mean_cosine_df['Target'] = target_df.mean(axis=1)
        mean_cosine_df['Cohort'] = cohort_df.mean(axis=1)
        mean_cosine_df['Rhyme'] = rhyme_df.mean(axis=1)
        mean_cosine_df['DAS Neighborhood'] = neighborhood_df.mean(axis=1)
        mean_cosine_df['Unrelated'] = unrelated_df.mean(axis=1)
        # this line was in Nick's original plotting function, but we never use self.mean_cosine_df, so
        #   I'm not sure it really needs to be promoted to a member variable
        # self.mean_cosine_df = mean_cosine_df
        return mean_cosine_df
        """

    def plot_cosine_grand_mean(self, mean_cosine_df):
        '''
        Makes the plot once you've produced mean_cosine_df.
        '''
        mean_cosine_df.plot.line(xlabel='Time Steps',ylabel='Cosine Simularity', ylim=(0,1), title="Cosine Simularity Grand Mean")
        return plt.show()


    def gen_RT(self, word, prediction_df, abs_threshold=0.4, rel_threshold=0.05, time_threshold=10):
        '''
        word = Target word to generate RTs for
        prediction_df = set of words for prediction, derived from a dataframe akin to the training dataframe
        abs_threshold = cosine threshold for target to hit to be considered accurate
        rel_threshold = relative distance of target to next highest cosine that must be reached to be considered accurate
        time_threshold = duration, in time steps, that relative threshold must be maintained between target and next highest cosine to be considered accurate
        '''
        # TODO Add next highest word when check fails
        # Accuracy/RT measures

        absolute_Criterion = abs_threshold
        relative_Criterion = rel_threshold
        time_Dependency_Criterion = (time_threshold, rel_threshold)

        if '_' in word:
            word_talker = word.split('_')
            # limit to true word length
            target_df = self.cosine_sim_dict[word][:prediction_df[prediction_df['Word']==word_talker[0]]['Input'].values[0].shape[0]][word_talker[0]]
            other_df = pd.DataFrame(self.cosine_sim_dict[word][:prediction_df[prediction_df['Word']==word_talker[0]]['Input'].values[0].shape[0]].drop(word_talker[0], axis=1).max(axis=1), columns=['Max Cosine'])
            other_df['Max Word'] = self.cosine_sim_dict[word][:prediction_df[prediction_df['Word']==word_talker[0]]['Input'].values[0].shape[0]].drop(word_talker[0], axis=1).idxmax(axis=1)
            rt_Dict = {
                ('Absolute'): np.nan,
                ('Relative'): np.nan,
                ('Time_Dependent'): np.nan
                }

            #Absolute threshold RT
            if not (other_df['Max Cosine'] > absolute_Criterion).any():
                absolute_Check_Array = target_df > absolute_Criterion
                for step in range(len(target_df)):
                    if absolute_Check_Array[step]:
                        rt_Dict['Absolute'] = (step, other_df.iloc()[step]['Max Word'])
                        break

            #Relative threshold RT
            relative_Check_Array = target_df > (other_df['Max Cosine'] + relative_Criterion)
            for step in range(len(target_df)):
                if relative_Check_Array[step]:
                    rt_Dict['Relative'] = (step, other_df.iloc()[step]['Max Word'])
                    break

            #Time dependent RT
            time_Dependency_Check_Array_with_Criterion = target_df > other_df['Max Cosine'] + time_Dependency_Criterion[1]
            time_Dependency_Check_Array_Sustainment = target_df > other_df['Max Cosine']
            for step in range(target_df.shape[0]- time_Dependency_Criterion[0]):
                if all(np.hstack([time_Dependency_Check_Array_with_Criterion[step:step + time_Dependency_Criterion[0]],
                                 time_Dependency_Check_Array_Sustainment[step + time_Dependency_Criterion[0]:]])):
                    rt_Dict['Time_Dependent'] = (step, other_df.iloc()[step]['Max Word'])
                    break

            self.rt_Dict = rt_Dict
        else:
            # limit to true word length
            target_df = self.cosine_sim_dict[word][:prediction_df[prediction_df['Word']==word]['Input'].values[0].shape[0]][word]
            other_df = pd.DataFrame(self.cosine_sim_dict[word][:prediction_df[prediction_df['Word']==word]['Input'].values[0].shape[0]].drop(word, axis=1).max(axis=1), columns=['Max Cosine'])
            other_df['Max Word'] = self.cosine_sim_dict[word][:prediction_df[prediction_df['Word']==word]['Input'].values[0].shape[0]].drop(word,axis=1).idxmax(axis=1)
            rt_Dict = {
                ('Absolute'): np.nan,
                ('Relative'): np.nan,
                ('Time_Dependent'): np.nan
                }

            #Absolute threshold RT
            if not (other_df['Max Cosine'] > absolute_Criterion).any():
                absolute_Check_Array = target_df > absolute_Criterion
                for step in range(len(target_df)):
                    if absolute_Check_Array[step]:
                        rt_Dict['Absolute'] = (step, other_df.iloc()[step]['Max Word'])
                        break

            #Relative threshold RT
            relative_Check_Array = target_df > (other_df['Max Cosine'] + relative_Criterion)
            for step in range(len(target_df)):
                if relative_Check_Array[step]:
                    rt_Dict['Relative'] = (step, other_df.iloc()[step]['Max Word'])
                    break

            #Time dependent RT
            time_Dependency_Check_Array_with_Criterion = target_df > other_df['Max Cosine'] + time_Dependency_Criterion[1]
            time_Dependency_Check_Array_Sustainment = target_df > other_df['Max Cosine']
            for step in range(target_df.shape[0]- time_Dependency_Criterion[0]):
                if all(np.hstack([time_Dependency_Check_Array_with_Criterion[step:step + time_Dependency_Criterion[0]],
                                 time_Dependency_Check_Array_Sustainment[step + time_Dependency_Criterion[0]:]])):
                    rt_Dict['Time_Dependent'] = (step, other_df.iloc()[step]['Max Word'])
                    break

            self.rt_Dict = rt_Dict
        return self.rt_Dict


# Cosines here may be busted as well
class PairPredictions(object):
    '''
    Generates cosine simularities for a selected word pair across all talkers in the set.
    '''
    def __init__(self, word1, word2, isi, training_df, fresh_model, checkpoint_path):
        '''
        word1: first word in the pair
        word2: second word in the pair
        isi: interstimulus interval between words in number of steps, defaults to None
        training_df: the dataframe that containins the training data to be subsetted
        fresh_model: an untrainined model of the same architecture to load the checkpoint onto
        checkpoint_path: desired checkpoint to load
        '''
        # save unique labels for cosine calcs
        self._unique_label_df = training_df[~training_df['Word'].duplicated()][['Word','Target']].reset_index()

        # combine two spectrograms and their targets with defined ISI // turn this into a class
        self.word1_df = training_df[training_df['Word'].isin([word1])]
        self.word2_df = training_df[training_df['Word'].isin([word2])]
        assert len(self.word1_df) == len(self.word2_df)
        word_pair_list = []
        if isi:
            for i in trange(len(self.word1_df)):
                words = self.word1_df.iloc()[i]['Word'] + '_' + self.word2_df.iloc()[i]['Word']
                isi_array_word = np.zeros((isi,self.word1_df.iloc()[i]['Input'].shape[1])) # assumes feature dimenion of equal length
                for j in range(len(self.word2_df)):
                    talkers = self.word1_df.iloc()[i]['Talker'] + '_' + self.word2_df.iloc()[j]['Talker']
                    word_pair_input = np.vstack((self.word1_df.iloc()[i]['Input'],
                                                 isi_array_word,
                                                 self.word2_df.iloc()[j]['Input']))
                    word_pair_list.append([words, talkers, word_pair_input])
        else:
            for i in trange(len(self.word1_df)):
                words = self.word1_df.iloc()[i]['Word'] + '_' + self.word2_df.iloc()[i]['Word']
                for j in range(len(self.word2_df)):
                    talkers = self.word1_df.iloc()[i]['Talker'] + '_' + self.word2_df.iloc()[j]['Talker']
                    word_pair_input = np.vstack((self.word1_df.iloc()[i]['Input'],
                                                 self.word2_df.iloc()[j]['Input']))
                    word_pair_list.append([words, talkers, word_pair_input])

        self.word_pair_df = pd.DataFrame(word_pair_list, columns=['Word Pair','Talkers', 'Pair Input'])

        M = max(len(a) for a in self.word_pair_df['Pair Input'].tolist())
        # pad all specs in batch to max length
        padded_spec = [ pad(s, (M, s.shape[1]), pad_val=-9999)
                        for s in self.word_pair_df['Pair Input'].tolist() ]


        fresh_model.load_weights(checkpoint_path)

        self.predictions = fresh_model.predict(np.array(padded_spec),
                                               batch_size = 32,
                                               verbose=1)
        self.cosine_dict = self._calc_cosines()


    def _calc_cosines(self):
        simularity_dict = {}
        Y = np.array(self._unique_label_df['Target'].to_list()).T
        for i in trange(len(self.predictions)):
            cosine_sim = []
            x = self.predictions[i]
            for step in x:
                cosine_sim.append(np.dot(Y.T,step)/(np.sqrt(np.dot(step,step))*np.sqrt((Y*Y).sum(axis=0))))
            simularity_df = pd.DataFrame(np.array(cosine_sim).reshape(x.shape[0],Y.shape[1]))
            simularity_df.columns = np.array(self._unique_label_df['Word'].to_list()).T
            simularity_dict[self.word_pair_df.iloc()[i]['Talkers']] = simularity_df
        return simularity_dict

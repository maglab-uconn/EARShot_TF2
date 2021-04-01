import os, io, librosa, pickle, json
import numpy as np
from random import sample
import concurrent.futures

from .audio import *

class PatternGenerator(object):
    '''
    Creates spectrogam/semantic vector pairs, using the parameters in pattern_params,
    and writes them to the directory specifed in the same parameters object.

    INPUT:
        pattern_params : parameters.PatternParameters object, required
            see parameters.PatternParameters for information about the various
            training/test pattern option/settings
    '''
    def __init__(self,pattern_params):
        self.pattern_params = pattern_params
        # pronounciation dictionary
        self.assemble_wordlist()
        # creates/reads in/massages semantic vectors
        self.assemble_semantic_vectors()
        # parsing of wav files to get word,talker,pronunciation,file tuples
        self.assemble_wavfile_list()
        # sets (or creates) the directory where the patterns will be dumped
        os.makedirs(self.pattern_params.pattern_path, exist_ok= True)


    def assemble_wordlist(self):
        '''
        Reads the pronounciation file and creates a word:list of phonemes
        dictionary.
        '''
        self.pron_dict = {}
        with open(self.pattern_params.lexicon, 'r') as f:
            lines = f.readlines()
            for line in lines:
                atoms = line.strip().split()
                self.pron_dict[atoms[0].upper()] = atoms[1].split('.')


    def assemble_semantic_vectors(self):
        '''
        Creates sparse random vectors (SRVs) for every word in the wordlist,
        or fetches pre-computed vector embeddings (from word2vec, LSI, or whatnot)
        and associates them with the words in the pronounciation dict.

        Words in the precomputed vector dictionary are assumed to be strictly lowercase.

        If the precomputed vector file is missing words in the model, the function
        raises an error, rather than inserting a random vector of some sort.
        '''
        self.semantic_patterns = {}
        # create new SRVs with the parameters specified in the options class
        if self.pattern_params.semantic_mode == 'SRV':
            # first bit just makes a vector of the desired size with the first num_nonzero
            #  elements equal to 1
            master_vec = np.zeros(self.pattern_params.semantic_dict['embedding_dim'])
            master_vec[:self.pattern_params.semantic_dict['num_nonzero']] = 1.0
            # assign a random permutation of the master_vec to each word
            for word in self.pron_dict.keys():
                self.semantic_patterns[word] = np.random.permutation(master_vec)
        # fetch word vectors/output patterns from specified location
        elif self.pattern_params.semantic_mode == 'VEC':
            precomputed_vectors = pickle.load(open(self.pattern_params.semantic_dict['vec_file'],'rb'))
            for word in self.pron_dict.keys():
                # rather than inserting a default value, if we don't have a word vector for
                #   a word in the model, we just raise an error and fail
                if word.lower() in precomputed_vectors:
                    self.semantic_patterns[word] = precomputed_vectors[word.lower()]
                else:
                    raise ValueError('word {} not found in precomputed vectors'.format(word))


    def assemble_wavfile_list(self):
        '''
        EARSHOT wav files all currently have the format WORD_TALKER.WAV.  This
        function recursively walks through the wav_path (there may be talker
        subdirectories) and extracts word/talker pairs to bind to semantic
        vectors.
        '''
        self.wav_files = []
        for root,_,files in os.walk(self.pattern_params.wav_path):
            for file in files:
                base_name,ext = os.path.splitext(file.upper())
                # enforce the rigid file format WORD_TALKER.{wav,WAV}
                if len(base_name.split('_')) != 2 or ext not in ['.wav','.WAV']:
                    continue
                word, talker = base_name.split('_')
                # don't bother with words not in the pronunciation dictionary
                if word not in self.pron_dict:
                    continue
                self.wav_files.append((word,self.pron_dict[word],talker,os.path.join(root,file).replace('\\', '/')))


    def generate_patterns(self,max_workers=10):
        '''
        Simple wrapper that allows multithreaded pattern generation. (N.B: Wasn't sure
        of any other clean way to move this out of main - KB)

        I've added proper exception handling from the executor calls; without this,
        the function silently succeeds no matter what happens because exceptions in the
        child processes are never caught.
        '''
        future_to_item = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for word,pron,talker,wav_file in self.wav_files:
                future_to_item[executor.submit(self.generate,word,pron,talker,wav_file)] = (word,pron,talker,wav_file)
            for future in concurrent.futures.as_completed(future_to_item):
                word,pron,talker,wav_file = future_to_item[future]
                try:
                    data = future.result()
                except:
                    print('{},{} pair generated an exception.'.format(word,talker))
                else:
                    print('{}:{} pair ({}) processed.'.format(word,talker,wav_file))


    def generate(self,word,pron,talker,wav_file):
        '''
        Generates a single talker/word spectrogram/semantic pattern example.
        This function is called many times (once per eacn word/talker combo)
        by generate_patterns.

        While you *could* use this to generate a specific single pattern,
        it's not recommended.  It mostly exists to wrap multithreaded execution.

        N.B: Key names in the pattern_dict are set to keys that Model.py will
        currently recognize; when we change these we have to be careful.
        '''
        pattern_dict = {'Word':word,'Pronunciation':pron,'Identifier':talker}
        # read/trim from the wav_file
        sig = librosa.core.load(wav_file,sr=self.pattern_params.spectrogram_dict['samp_rate'])[0]
        sig = librosa.effects.trim(sig,frame_length=32,hop_length=16)[0]
        # this block if we are using spectrograms
        if self.pattern_params.acoustic_mode == 'spectrogram':
            spec = spectrogram(sig,**self.pattern_params.spectrogram_dict)
            pattern_dict['Acoustic'] = np.transpose(spec).astype(np.float32)
        # using MEL frequency cepstral coefficients instead
        elif self.pattern_params.acoustic_mode == 'mel':
            spec = mel_spectrogram(sig,**self.pattern_params.mel_dict)
        else:
            raise ValueError('{} is not a valid choice for acoustic_mode.'.format(self.pattern_params.acoustic_mode == 'mel'))
        pattern_dict['Acoustic'] = np.transpose(spec).astype(np.float32)
        # add the semantic pattern
        pattern_dict['Semantic'] = self.semantic_patterns[word].astype(np.float32)
        pattern_filename = os.path.split(wav_file)[1].replace(os.path.splitext(wav_file)[1], '.pickle').upper()
        # N.B. - why are we fixing the protocol here and not using -1?
        with open(os.path.join(self.pattern_params.pattern_path, pattern_filename).replace('\\', '/'), 'wb') as f:
            pickle.dump(pattern_dict, f, protocol= 4)
        # status message
        #print('{}\t->\t{}'.format(wav_file, pattern_filename))
        # this is just to catch as exception in the thread pool executor?
        #return 1


    def generate_metadata(self):
        '''
        Generates the metadata file.  I'm not actually sure what this is even used for or where?
        (Presumably I'll find out at some point.) Plus, why aren't we logging this information
        when we create it?  This should go away or be drastically modified - right now it's
        basically just a copy of Heejo's original code.

        Also, this assumes a very rigid pattern format: WORD_TALKER.PICKLE
        '''
        metadata_dict = {'parameters':self.pattern_params}
        # word : pronunciation
        metadata_dict['Pronunciation_Dict'] = self.pron_dict
        # (word,talker) : pattern_filename
        metadata_dict['Pattern_Path_Dict'] = {}
        # this is just a reverse of metadata_dict['Pattern_Path_Dict']
        metadata_dict['Word_and_Identifier_Dict'] = {}
        # pattern_filename : pattern_step
        metadata_dict['Step_Dict'] = {}
        for root,_,files in os.walk(self.pattern_params.pattern_path):
            for file in files:
                # right now, pattern files have a very rigid format: WORD_TALKER.PICKLE.  Ignore
                #   any files not matching this pattern
                base_name,ext = os.path.splitext(file)
                if ext != '.PICKLE' or len(base_name.split('_')) != 2:
                    continue
                # file matches rigid pattern, carry on.
                with open(os.path.join(root, file).replace('\\', '/'), 'rb') as f:
                    pattern_dict = pickle.load(f)
                if not all([(key in pattern_dict.keys()) for key in ['Word', 'Identifier', 'Acoustic']]):
                    print('File \'{}\' is not a valid pattern. This file is ignored.'.format(file))
                    continue

                metadata_dict['Pattern_Path_Dict'][pattern_dict['Word'], pattern_dict['Identifier']] = file
                metadata_dict['Word_and_Identifier_Dict'][file] = (pattern_dict['Word'], pattern_dict['Identifier'])
                metadata_dict['Step_Dict'][file] = pattern_dict['Acoustic'].shape[0]

        metadata_dict['Target_Dict'] = {word: self.semantic_patterns[word] for word in self.pron_dict.keys()}

        with open(os.path.join(self.pattern_params.pattern_path, 'METADATA.PICKLE').replace('\\', '/'), 'wb') as f:
            pickle.dump(metadata_dict, f, protocol= 4)



    def generate_metadata_subset(self,word_list=None,talker_list=None,metadata_filename='METADATA.SUBSET.PICKLE'):
        '''
        I personally haven't been excluding any talkers, etc. so I don't need this to be super functional right now.
        It's just a pretty close copy of Heejo's function for now, to be tested/modified later.
        '''
        if word_list is not None:
            word_list = [x.upper() for x in word_list]
        if talker_list is not None:
            talker_list = [x.upper() for x in talker_list]

        with open(os.path.join(self.pattern_params.pattern_path, 'METADATA.PICKLE').replace('\\', '/'), 'rb') as f:
            metadata_dict = pickle.load(f)

        sub_metadata_dict = {'parameters':self.pattern_params,'Pronunciation_Dict':self.pron_dict}

        # filter on words
        if word_list is not None:
            word_filtered_pattern_files = [pattern_file for pattern_file, (word, talker) in metadata_dict['Word_and_Identifier_Dict'].items() if word in word_list]
        else:
            word_filtered_pattern_files = [pattern_file for pattern_file, (word, talker) in metadata_dict['Word_and_Identifier_Dict'].items()]
        # filter on talkers
        if talker_list is not None:
            talker_filtered_pattern_files = [pattern_file for pattern_file, (word, talker) in metadata_dict['Word_and_Identifier_Dict'].items() if talker in talker_list]
        else:
            talker_filtered_pattern_files = [pattern_file for pattern_file, (word, talker) in metadata_dict['Word_and_Identifier_Dict'].items()]

        # get a list of patterns in both filtered lists
        wordtalker_union = []
        for p in word_filtered_pattern_files:
            if p in talker_filtered_pattern_files:
                wordtalker_union.append(p)

        sub_metadata_dict['Pattern_Path_Dict'] = {(word,talker): pattern_file for (word,talker), pattern_file in metadata_dict['Pattern_Path_Dict'].items() if pattern_file in wordtalker_union}
        # the word and identifier dict is just a reverse of the pattern path dict
        for k,v in sub_metadata_dict['Pattern_Path_Dict']:
            sub_metadata_dict['Word_and_Identifier_Dict'][v] = k

        sub_metadata_dict['Step_Dict'] = {pattern_file:step for pattern_file, step in metadata_dict['Step_Dict'].items() if pattern_file in wordtalker_union}

        if word_list is not None:
            sub_metadata_dict['Target_Dict'] = {word:target_pattern for word, target_pattern in metadata_dict['Target_Dict'].items() if word in word_list}
        else:
            sub_metadata_dict['Target_Dict'] = metadata_dict['Target_Dict']

        with open(os.path.join(self.pattern_params.pattern_path, metadata_filename).replace('\\', '/'), 'wb') as f:
            pickle.dump(sub_metadata_dict, f, protocol= 4)

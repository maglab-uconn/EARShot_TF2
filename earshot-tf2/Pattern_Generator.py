import os, io, librosa, pickle, json
import numpy as np
from random import sample
from Audio import *
import concurrent.futures

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

#Global constants
with open(hp_Dict['Pattern']['Lexicon_File'], 'r') as f:
    lines = f.readlines()
    pronunciation_Dict = {
        word.upper(): pronunciation.split('.')
        for word, pronunciation in [line.strip().split('\t') for line in lines]
        }
    using_Word_List = list(pronunciation_Dict.keys())

if hp_Dict['Pattern']['Semantic']['Mode'].upper() == 'SRV':
    semantic_Index_Dict = {}
    for word in using_Word_List:
        unit_List = list(range(hp_Dict['Pattern']['Semantic']['SRV']['Size']))
        while True:
            index_Set =  set(sample(unit_List, hp_Dict['Pattern']['Semantic']['SRV']['Assign_Number']))
            if not index_Set in semantic_Index_Dict.values():
                semantic_Index_Dict[word] = index_Set
                break

    semantic_Dict = {}
    for word, index_Set in semantic_Index_Dict.items():
        new_Semantic_Pattern = np.zeros(shape= (hp_Dict['Pattern']['Semantic']['SRV']['Size']), dtype= np.float32)
        for unit_Index in index_Set:
            new_Semantic_Pattern[unit_Index] = 1
        semantic_Dict[word] = new_Semantic_Pattern
elif hp_Dict['Pattern']['Semantic']['Mode'].upper() == 'PGD':
    with open(hp_Dict['Pattern']['Semantic']['PGD']['Dict_File_Path'], 'rb') as f:
        semantic_Dict = {
            word.upper(): pattern
            for word, pattern in pickle.load(f).items()
            if word.upper() in using_Word_List
            }

    if len(semantic_Dict) < len(using_Word_List):
        raise ValueError('Some words are not in pre generated dict. {} : {}'.format(len(semantic_Dict), len(using_Word_List)))


def Pattern_File_Generate(
    word,
    pronunciation,
    identifier, #In paper, this is 'talker'.
    voice_File_Path,
    ):
    new_Pattern_Dict = {
        'Word': word,
        'Pronunciation': pronunciation,
        'Identifier': identifier
        }

    if hp_Dict['Pattern']['Acoustic']['Mode'].upper() == 'Spectrogram'.upper():
        sig = librosa.core.load(
            voice_File_Path,
            sr = hp_Dict['Pattern']['Acoustic']['Spectrogram']['Sample_Rate']
            )[0]
        sig = librosa.effects.trim( # Trim
            sig,
            frame_length = 32,
            hop_length=16
            )[0]
        spec = spectrogram(
            sig,
            num_freq= hp_Dict['Pattern']['Acoustic']['Spectrogram']['Dimension'],
            frame_shift_ms= hp_Dict['Pattern']['Acoustic']['Spectrogram']['Frame_Shift'],
            frame_length_ms= hp_Dict['Pattern']['Acoustic']['Spectrogram']['Frame_Length'],
            samp_rate= hp_Dict['Pattern']['Acoustic']['Spectrogram']['Sample_Rate'],
            )
        new_Pattern_Dict['Acoustic'] = np.transpose(spec).astype(np.float32)    # [Time, Dim]
    elif hp_Dict['Pattern']['Acoustic']['Mode'].upper() == 'Mel'.upper():
        sig = librosa.core.load(
            voice_File_Path,
            sr = hp_Dict['Pattern']['Acoustic']['Mel']['Sample_Rate']
            )[0]
        sig = librosa.effects.trim(   #Trim
            sig,
            frame_length = 32,
            hop_length=16
            )[0]
        mel = mel_spectrogram(
            sig,
            num_freq= hp_Dict['Pattern']['Acoustic']['Mel']['Spectrogram_Dimension'],
            frame_shift_ms= hp_Dict['Pattern']['Acoustic']['Mel']['Frame_Shift'],
            frame_length_ms= hp_Dict['Pattern']['Acoustic']['Mel']['Frame_Length'],
            num_mels= hp_Dict['Pattern']['Acoustic']['Mel']['Dimension'],
            samp_rate= hp_Dict['Pattern']['Acoustic']['Mel']['Sample_Rate'],
            max_abs_value= hp_Dict['Pattern']['Acoustic']['Mel']['Max_Abs']
            )
        new_Pattern_Dict['Acoustic'] = np.transpose(mel).astype(np.float32)
    else:
        assert False

    new_Pattern_Dict['Semantic'] = semantic_Dict[word].astype(np.float32)

    pattern_File_Name = os.path.split(voice_File_Path)[1].replace(os.path.splitext(voice_File_Path)[1], '.pickle').upper()

    with open(os.path.join(hp_Dict['Pattern']['Pattern_Path'], pattern_File_Name).replace('\\', '/'), 'wb') as f:
        pickle.dump(new_Pattern_Dict, f, protocol= 4)

    print('{}\t->\t{}'.format(voice_File_Path, pattern_File_Name))


def Metadata_Generate():
    new_Metadata_Dict = {}

    #Although we use the hyper parameter now, I insert several information about that for checking consistency.
    new_Metadata_Dict = {
        'Hyper_Parameter_Dict': hp_Dict['Pattern']
        }

    new_Metadata_Dict['Pronunciation_Dict'] = pronunciation_Dict   #key: word, value: pronunciation
    new_Metadata_Dict['Pattern_Path_Dict'] = {}    #key: (word, identifier), value: pattern_Path
    new_Metadata_Dict['Word_and_Identifier_Dict'] = {}     #key: pattern_Path, value: (word, identifier) #Reversed of pattern_Path_Dict
    new_Metadata_Dict['Step_Dict'] = {}     #key: pattern_Path, value: pattern step
    for root, _, files in os.walk(hp_Dict['Pattern']['Pattern_Path']):
        for file in files:
            if file.upper() == 'Metadata.pickle'.upper():
                continue
            with open(os.path.join(root, file).replace('\\', '/'), 'rb') as f:
                pattern_Dict = pickle.load(f)
            if not all([(key in pattern_Dict.keys()) for key in ['Word', 'Identifier', 'Acoustic']]):
                print('File \'{}\' is not correct pattern. This file is ignored.'.format(file))
                continue
            new_Metadata_Dict['Pattern_Path_Dict'][pattern_Dict['Word'], pattern_Dict['Identifier']] = file
            new_Metadata_Dict['Word_and_Identifier_Dict'][file] = (pattern_Dict['Word'], pattern_Dict['Identifier'])
            new_Metadata_Dict['Step_Dict'][file] = pattern_Dict['Acoustic'].shape[0]

    new_Metadata_Dict['Target_Dict'] ={word: semantic_Dict[word] for word in using_Word_List}

    with open(os.path.join(hp_Dict['Pattern']['Pattern_Path'], 'METADATA.PICKLE').replace('\\', '/'), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol= 4)


def Metadata_Subset_Generate(word_List = None, identifier_List = None, metadata_File_Name = 'METADATA.SUBSET.PICKLE'):
    if not word_List is None:
        word_List = [x.upper() for x in word_List]
    if not identifier_List is None:
        identifier_List = [x.upper() for x in identifier_List]

    with open(os.path.join(hp_Dict['Pattern']['Pattern_Path'], 'METADATA.PICKLE').replace('\\', '/'), 'rb') as f:
        metadata_Dict = pickle.load(f)

    new_Metadata_Dict = {}
    new_Metadata_Dict['Hyper_Parameter_Dict'] = metadata_Dict['Hyper_Parameter_Dict']
    new_Metadata_Dict['Pronunciation_Dict'] = metadata_Dict['Pronunciation_Dict']

    if not word_List is None:
        word_Filtered_Pattern_Path_List = [
            pattern_Path
            for pattern_Path, (word, identifier) in metadata_Dict['Word_and_Identifier_Dict'].items()
            if word in word_List
            ]
    else:
        word_Filtered_Pattern_Path_List = [
            pattern_Path
            for pattern_Path, (word, identifier) in metadata_Dict['Word_and_Identifier_Dict'].items()
            ]

    if not identifier_List is None:
        identifier_Filtered_Pattern_Path_List = [
            pattern_Path
            for pattern_Path, (word, identifier) in metadata_Dict['Word_and_Identifier_Dict'].items()
            if identifier in identifier_List
            ]
    else:
        identifier_Filtered_Pattern_Path_List = [
            pattern_Path
            for pattern_Path, (word, identifier) in metadata_Dict['Word_and_Identifier_Dict'].items()
            ]

    new_Metadata_Dict['Pattern_Path_Dict'] = {
        (word, identifier): pattern_Path
        for (word, identifier), pattern_Path in metadata_Dict['Pattern_Path_Dict'].items()
        if pattern_Path in word_Filtered_Pattern_Path_List and pattern_Path in identifier_Filtered_Pattern_Path_List
        }

    new_Metadata_Dict['Word_and_Identifier_Dict'] = {
        pattern_Path: (word, identifier)
        for pattern_Path, (word, identifier) in metadata_Dict['Word_and_Identifier_Dict'].items()
        if pattern_Path in word_Filtered_Pattern_Path_List and pattern_Path in identifier_Filtered_Pattern_Path_List
        }

    new_Metadata_Dict['Step_Dict'] = {
        pattern_Path: step
        for pattern_Path, step in metadata_Dict['Step_Dict'].items()
        if pattern_Path in word_Filtered_Pattern_Path_List and pattern_Path in identifier_Filtered_Pattern_Path_List
        }

    if not word_List is None:
        new_Metadata_Dict['Target_Dict'] = {
            word: target_Pattern
            for word, target_Pattern in metadata_Dict['Target_Dict'].items()
            if word in word_List
            }
    else:
        new_Metadata_Dict['Target_Dict'] = metadata_Dict['Target_Dict']

    with open(os.path.join(hp_Dict['Pattern']['Pattern_Path'], metadata_File_Name).replace('\\', '/'), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol= 4)


def Get_File_List():
    file_List = []
    for root, _, files in os.walk(hp_Dict['Pattern']['Wav_Path']):
        for file in files:
            name, ext = os.path.splitext(file.upper())
            if len(name.split('_')) != 2 or ext != '.wav'.upper():
                continue
            word, talker = name.split('_')

            if not word in using_Word_List:
                continue

            file_List.append((
                word,
                pronunciation_Dict[word],
                talker,
                os.path.join(root, file).replace('\\', '/')
                ))

    return file_List

if __name__ == '__main__':
    os.makedirs(hp_Dict['Pattern']['Pattern_Path'], exist_ok= True)
    max_workers = 10

    file_List = Get_File_List()

    
    future_to_item = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for word,pron,talker,wav_file in file_List:
            future_to_item[executor.submit(Pattern_File_Generate,word,pron,talker,wav_file)] = (word,pron,talker,wav_file)
        for future in concurrent.futures.as_completed(future_to_item):
            word,pron,talker,wav_file = future_to_item[future]
            try:
                data = future.result()
            except:
                print('{},{} pair generated an exception.'.format(word,talker))
            else:
                print('{}:{} pair ({}) processed.'.format(word,talker,wav_file))

    #word,pron,talker,wav_file = file_List[0]
    #Pattern_File_Generate(word,pron,talker,wav_file)

    Metadata_Generate()

    # Example
    #Metadata_Subset_Generate(
    #    identifier_List= ['Agnes', 'Alex', 'Allison', 'Ava',  'Bruce', 'Fred', 'Junior', 'Kathy', 'Princess', 'Ralph', 'Samantha', 'Susan', 'Tom', 'Vicki', 'Victoria'],
    #    metadata_File_Name = 'METADATA.1KW.15T.PICKLE'
    #    )

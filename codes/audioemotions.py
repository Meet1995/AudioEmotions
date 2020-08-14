# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:39:07 2020

@author: mg21929
"""
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity

class AudioSimilarityEmotions:
    def __init__(self, audio_database_path, database_sample_rate=16000):        
        self.database_sr = database_sample_rate
        self.audio_database = pd.read_pickle(audio_database_path)
        self.means, self.scaler, self.scaled_data = self.get_means()
        

    def __call__(self, arr, sr):
        """
        While inferencing the sample rate of the incoming audio should match the sample rate at which 
        the database was generated. If the sample rate of audio doesn't match database sample rate, then it will be 
        resampled to match it.
        """
        if sr != self.database_sr:
            f = sr/self.database_sr
            num = int(arr.shape[0]/f)
            arr_resampled = resample(arr, num)
        else:
            arr_resampled = arr
            
        vec = self.get_scaled_vec(arr_resampled,sr)
#         df = self.measure_similarity(vec)[:topn]
        dictt = self.measure_similarity(vec)
        return dictt


    def get_means(self):
        cols = [x for x in self.audio_database.columns if x != 'target']
        x = self.audio_database[cols].values
        sc = RobustScaler()
        sc.fit(x)
        xscaled = sc.transform(x)
        data = pd.DataFrame(xscaled, columns=cols)
        data['class'] = self.audio_database['target']
        classes = data['class'].unique().tolist()
        means={}
        for i,_cls in enumerate(classes):
            means[_cls] = np.array(data[data['class'] == classes[i]].mean()
                                   ).reshape((1, xscaled.shape[1]))
        return means, sc, data

    def get_scaled_vec(self, arr, sr):
        """
        n_chroma and n_mel are left to default 12 and 128, n_mfcc is calculated
        based on n_chroma, n_mel and audio_database(which should be generated
        using generate_audio_database static method).

        Parameters
        ----------
        arr : TYPE
            DESCRIPTION.
        sr : TYPE
            DESCRIPTION.

        Returns
        -------
        vec : TYPE
            DESCRIPTION.

        """
        window_size=2048
        window_stride=512
        arr = arr.astype('float32')
        lst = []

        mel = librosa.feature.melspectrogram(arr, n_fft=window_size, hop_length=window_stride)
        chroma = librosa.feature.chroma_stft(arr,sr=sr,n_fft=window_size, hop_length=window_stride)
        n_mfcc = self.audio_database.shape[1]-1 - (mel.shape[0] + chroma.shape[0])
        mfccs = librosa.feature.mfcc(arr, sr=sr, n_fft=window_size, hop_length=window_stride,
                                     n_mfcc=n_mfcc)
        lst.append(mel)
        lst.append(mfccs)
        lst.append(chroma)

        vec = np.concatenate(lst)
        vec = vec.mean(axis = 1).reshape((1,self.audio_database.shape[1]-1))
        vec = self.scaler.transform(vec)
        return vec


    def measure_similarity(self, feature_vec):
        dictt = {}
        for key, value in self.means.items():
            dictt[key] = cosine_similarity(value,feature_vec)[0,0]

#         dfc = pd.DataFrame(dictt.items(), columns=['class','similarity']).sort_values(
#             'similarity', ascending = False).reset_index(drop=True)
        return dictt

    @staticmethod
    def generate_audio_database(file_path_list, emotion_identifier_index, emotion_map,
                                sample_rate=16000, n_mfcc=40):
        """    
        For generating audio database: 
        Upsampling audio that was recorded at a lower sample rate will not give correct results. 
        One should always provide the sample rate at which the audio was recorded or a lower one. 
    
        The path of the audio files should look something like this:

            pathtofile\\file-{emotion_identifier}-someotherstuff.wav

        Parameters
        ----------
        file_path_list : list of strings
            DESCRIPTION: glob path list of all the audio files
        emotion_identifier_index : int
            DESCRIPTION: index of emotion_identifier in
            "file-{emotion_identifier}-someotherstuff.wav".split("-")
        emotion_map : dictionary
            DESCRIPTION: A dictionary mapping emotion_identifier with emotion name
        sample_rate : int, optional
            DESCRIPTION: The default is 16000.
        n_mfcc : int, optional
            DESCRIPTION: The default is 40.

        Returns
        -------
        df : DataFrame
            DESCRIPTION: A dataframe containing unscaled audio features along with a
            "target" column.
        """
        def get_vec(arr, sr, n_mfcc):
            window_size=2048
            window_stride=512
            arr = arr.astype('float32')
            lst = []

            mel = librosa.feature.melspectrogram(arr, n_fft=window_size, hop_length=window_stride)
            chroma = librosa.feature.chroma_stft(arr,sr=sr,n_fft=window_size,
                                                 hop_length=window_stride)
            mfccs = librosa.feature.mfcc(arr, sr=sr, n_fft=window_size, hop_length=window_stride,
                                         n_mfcc=n_mfcc)

            lst.append(mel)
            lst.append(mfccs)
            lst.append(chroma)

            vec = np.concatenate(lst)
            shape = mel.shape[0] + chroma.shape[0] + mfccs.shape[0]
            vec = vec.mean(axis = 1).reshape((1,shape))
            return vec

        x_list = []
        y_list = []
        for file_path in tqdm(file_path_list):
            arr, _ = librosa.load(file_path, sr=sample_rate)
            emo = emotion_map[file_path.split("\\")[-1].split("-")[emotion_identifier_index]]
            x_list.append(get_vec(arr, sample_rate,n_mfcc))
            y_list.append(emo)
        x = np.concatenate(x_list)
        cols = [f'f{i}' for i in range(x.shape[1])]
        df = pd.DataFrame(x, columns=cols)
        df['target'] = y_list
        df.to_pickle(f'AudioDatabase_{sample_rate}.pkl')
        print("Database created and saved")
        return df
    
    
import pyaudio
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

class Recorder:    
    def __init__(self, chunk=4000,sr=16000,seconds_to_wait=0.5,dtype='int16', channels=1, 
                 threshold=None):
        self.chunk = chunk
        self.sr = sr
        self.seconds_to_wait = seconds_to_wait
        self.dtype = dtype
        self.channels = channels
        self.chunks_per_sec = int(sr/chunk)
        
        self.p = pyaudio.PyAudio()
        
        if dtype == 'float32':
            pyaudio_const = pyaudio.paFloat32
            self.thresh = 0.1            
        elif dtype == 'int32':
            pyaudio_const = pyaudio.paInt32
            self.thresh = 2**26
        else:
            pyaudio_const = pyaudio.paInt16
            self.thresh = 2**13
            
        if threshold:
            self.thresh = threshold
            
        self.stream = self.p.open(rate=self.sr, channels=self.channels, format=pyaudio_const,
                             input=True, output=False, frames_per_buffer=self.chunk) 
            
    
    def print_audio_devices(self):
        for x in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(x)
            print(info['name'])
    
    def record(self):      
        analyzing = True
        start_recording = False        
        chunk_count=1
        while analyzing:            
            raw_tmp = self.stream.read(self.chunk)
            tmp = np.frombuffer(raw_tmp, self.dtype)
            
            if tmp.max() > self.thresh:                
                start_recording = True                
                arr = tmp
                raw_data = raw_tmp
                print('Recording Starts')
                
                while start_recording:                    
                    buffer = self.stream.read(self.chunk)
                    raw_data += buffer    
                    arr = np.concatenate((arr,np.frombuffer(buffer, self.dtype)),axis=0)
    
                    chunk_count +=1                    
                    if chunk_count > self.seconds_to_wait * self.chunks_per_sec:                        
                        idx = int(self.seconds_to_wait * self.sr)
                        max_value = arr[-idx:].max()
                        if max_value<self.thresh:
                            print('Recording Stops')
                            start_recording = False
                            analyzing = False         
        return arr, raw_data
        
                            
    def record_and_visualize(self):
        fig, ax = plt.subplots()
        x = np.arange(self.chunk)
        line, = ax.plot(x,np.random.randn(self.chunk))
        
        if self.dtype == 'int16':
            ax.set_ylim([-2**15,(2**15)-1])
        elif self.dtype == 'int32':
            ax.set_ylim([-2**31,(2**31)-1])
        elif self.dtype == 'float32':
            ax.set_ylim([-1,1])
            
        analyzing = True
        start_recording = False        
        chunk_count=1
        while analyzing:            
            raw_tmp = self.stream.read(self.chunk)
            tmp = np.frombuffer(raw_tmp, self.dtype)
            
            if tmp.max() > self.thresh:                
                start_recording = True                
                arr = tmp
                raw_data = raw_tmp
                print('Recording Starts')
                
                while start_recording:                    
                    buffer = self.stream.read(self.chunk)
                    raw_data += buffer                        
                    buffer_unpacked = np.frombuffer(buffer, self.dtype)
                    line.set_ydata(buffer_unpacked)
                    fig.canvas.draw()
                    fig.canvas.flush_events()                    
                    arr = np.concatenate((arr,buffer_unpacked),axis=0)
    
                    chunk_count +=1                    
                    if chunk_count > self.seconds_to_wait * self.chunks_per_sec:                        
                        idx = int(self.seconds_to_wait * self.sr)
                        max_value = arr[-idx:].max()
                        if max_value<self.thresh:
                            print('Recording Stops')
                            start_recording = False
                            analyzing = False      
                            plt.close()
        return arr, raw_data
    
    @staticmethod
    def write_to_wav(name, sr, arr):
        wavfile.write(name, sr, arr)
        print(f'file {name} saved to working directory')

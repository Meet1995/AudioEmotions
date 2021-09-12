import subprocess

def convert_mp4_to_wav(mp4_path, samplerate=16000):
    tmp = mp4_path.split(".")[0]
    command = f"ffmpeg -i {mp4_path} -ab 160k -ac 1 -ar {samplerate} -codec pcm_s16le -vn {tmp}_{samplerate}khz.wav"
    subprocess.call(command, shell=True)



import librosa

def get_beat_list(audio_file:str):
    # 加载音频文件
    y, sr = librosa.load(audio_file)

    # 提取音频的节奏和拍子信息
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr,tightness=0.01,hop_length=512)
    print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
    # 打印节奏和拍子信息
    print("Tempo:", tempo)
    print("Beat frames:", beat_frames)
    times = librosa.frames_to_time(beat_frames, sr=sr)
    return times

    # 将帧数转换为时间点
    # beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # # 打印节奏和拍子信息
    # print("Tempo:", tempo)
    # print("Beat times:", beat_times)

if __name__ == '__main__':
    print(get_beat_list('../Alex MakeMusic - I Am Good - Instrumental Version.mp3'))
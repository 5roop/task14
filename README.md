# Task 14: CrES dataset



## 2022-08-22T11:58:28

I downloaded the corpus and rsynced it to kt-gpu-vm-1TB. The audio is indeed sampled with a weird sample rate:

```
(base) peterr@kt-gpu-vm-1TB:~/macocu/task14$ soxi cres_v2.1/audio_files/1.wav 

Input File     : 'cres_v2.1/audio_files/1.wav'
Channels       : 1
Sample Rate    : 11025
Precision      : 16-bit
Duration       : 00:00:06.70 = 73867 samples ~ 502.497 CDDA sectors
File Size      : 148k
Bit Rate       : 176k
Sample Encoding: 16-bit Signed Integer PCM
```

A bash script for upsampling will be prepared to mitigate this.

# Addendum 2022-08-23T07:39:10

All of the work so far has been done in [the first notebook](001_dataset_introspection.ipynb). The target column should be 'discrete_emotion_annotation_phase', as agreed in an email by Branimir.


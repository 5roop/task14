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

# Addendum 2022-08-24T11:10:17

Model "classla/wav2vec2-large-slavic-parlaspeech-hr" does not train as the other two, due to Memory issues on the GPU. Also, when training it, the disk filled up, so I had to clean the disk a bit. To try to get around that I also implemented clipping to 10 seconds for this particular model. This combination then finally worked fine.




# Addendum 2022-08-25T10:33:01

Presumed optimal hyperparameters:

| model                                          | best epoch nr | best accuracy on dev @ presumed optimal nr of epochs |
|------------------------------------------------|---------------|------------------------------------------------------|
| "facebook/wav2vec2-large-960h-lv60-self",      | 9             | 0.73                                                 |
| "facebook/wav2vec2-large-slavic-voxpopuli-v2"  | 11            | 0.74                                                 |
| "classla/wav2vec2-large-slavic-parlaspeech-hr" | 7             | 0.75                                                 |

So far HuBERT has proved difficult to handle, there is a bunch of bugs being raised due to: `AttributeError: 'HubertConfig' object has no attribute 'add_adapter'`, although this attribute is not used anywhere in my code...

Upon analyzing the results, two things became obvious: 
* the calculated accuracy and macroF1 values are often repeated, and the confusion matrices often seem very much alike. Is this due overfitting?
* On both metrics test performs worse than dev. 

Meeting notes:
* Get Nikola model outputs for a random run in form of utterance_id, y_true, y_pred

# Addendum 2022-08-29T11:45:51

The outputs for Nikola are [here](005_analysis_of_outputs.jsonl)
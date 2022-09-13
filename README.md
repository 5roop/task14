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

# Addendum 2022-09-06T11:16:41

I implemented a fix for Hubert ("facebook/hubert-large-ls960-ft") but the results are shit (accuracy fluctuates between 0.17 and 0.3 during training). Searching for other options.

First I increased the number of training epochs, but the behaviour stayed the same. I will try another HuBERT model, if the poor performance issues persist, I'll have to debug deeper.

Notes so far:
* "facebook/hubert-large-ls960-ft": produces sub-par results
* "facebook/hubert-xlarge-ls960-ft": had to clean the caches due to low disk space, but now works. First results are not optimistic.

To make it work in the first place I had to manually add the line

```python
setattr(config, "add_adapter", False)
```
To try and improve results I changed this attribute to `True`, but that set off a cascade of other options I had to include. I found similar settings in the Hubert config files, but that does not mean I included the correct ones.

The final additions are:

```python
    if "hubert" in model_name_or_path:
        setattr(config, "add_adapter", True)
        setattr(config, "output_hidden_size", 1024)
        setattr(config, "num_adapter_layers", 7)
        setattr(config, "adapter_kernel_size", 3)
        setattr(config, "adapter_stride", 7)
```

This works for training both models, but when it comes to evaluation, only `hubert-large` works, while `hubert-xlarge` raises `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x1024 and 1280x1280)`

# Addendum 2022-09-07T08:07:47

The above error seems to be avoidable with additional `if` clause:
```python
    if "xlarge" in model_name_or_path:
        setattr(config, "output_hidden_size", 1280)
    else:
        setattr(config, "output_hidden_size", 1024)
```




# Addendum 2022-09-08T08:47:42

* Remove outliers by accuracy < 0.5 ✓
* Prepare Dummy classifier baselines for dev, test ✓-> max accuracy on any split with any strategy: 0.3, max macro F1: 0.2


# Addendum 2022-09-12T13:03:13
Research what layer the classification comes from, also from ASR.
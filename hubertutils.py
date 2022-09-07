
import numpy as np
example_eval_config = {
    "output_column": "Speaker_age_group",
    "model_name_or_path":  "./models/facebook_wav2vec2-large-slavic-voxpopuli-v2_age_clf_male_15epochs_v2_/checkpoint-2625",
    "eval_file": "005_testv2_males.csv"
}


def eval_model(config_dict):
    import torch
    import torch.nn as nn
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2PreTrainedModel,
        Wav2Vec2Model
    )
    from transformers.models.hubert.modeling_hubert import (
        HubertPreTrainedModel,
        HubertModel
    )
    import librosa
    from sklearn.metrics import classification_report
    from datasets import load_dataset

    output_column = config_dict.get("output_column")
    model_name_or_path = config_dict.get("model_name_or_path")
    test_dataset = load_dataset(
        "csv", data_files={"test": config_dict.get("eval_file")}, delimiter=",")["test"]
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoConfig, Wav2Vec2Processor
    import torch
    import torch.nn as nn
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2PreTrainedModel,
        Wav2Vec2Model
    )

    class Wav2Vec2ClassificationHead(nn.Module):
        """Head for wav2vec classification task."""

        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):
            x = features
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)
            return x

    class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.pooling_mode = config.pooling_mode
            self.config = config

            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = Wav2Vec2ClassificationHead(config)

            self.init_weights()

        def freeze_feature_extractor(self):
            self.wav2vec2.feature_extractor._freeze_parameters()

        def merged_strategy(
                self,
                hidden_states,
                mode="mean"
        ):
            if mode == "mean":
                outputs = torch.mean(hidden_states, dim=1)
            elif mode == "sum":
                outputs = torch.sum(hidden_states, dim=1)
            elif mode == "max":
                outputs = torch.max(hidden_states, dim=1)[0]
            else:
                raise Exception(
                    "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

            return outputs

        def forward(
                self,
                input_values,
                attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                labels=None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            outputs = self.wav2vec2(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            hidden_states = self.merged_strategy(
                hidden_states, mode=self.pooling_mode)
            logits = self.classifier(hidden_states)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SpeechClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    config = AutoConfig.from_pretrained(model_name_or_path)

    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
    except:
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            "./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" "
        )
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor, tokenizer=tokenizer)

    # model = Wav2Vec2ForSpeechClassification.from_pretrained(
    #     model_name_or_path).to(device)
    class HubertClassificationHead(nn.Module):
        """Head for hubert classification task."""

        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):
            x = features
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)
            return x

    
    class HubertForSpeechClassification(HubertPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.pooling_mode = config.pooling_mode
            self.config = config

            self.hubert = HubertModel(config)
            self.classifier = HubertClassificationHead(config)

            self.init_weights()

        def freeze_feature_extractor(self):
            self.hubert.feature_extractor._freeze_parameters()

        def merged_strategy(
                self,
                hidden_states,
                mode="mean"
        ):
            if mode == "mean":
                outputs = torch.mean(hidden_states, dim=1)
            elif mode == "sum":
                outputs = torch.sum(hidden_states, dim=1)
            elif mode == "max":
                outputs = torch.max(hidden_states, dim=1)[0]
            else:
                raise Exception(
                    "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

            return outputs

        def forward(
                self,
                input_values,
                attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                labels=None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            outputs = self.hubert(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            hidden_states = self.merged_strategy(
                hidden_states, mode=self.pooling_mode)
            logits = self.classifier(hidden_states)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SpeechClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
    model = HubertForSpeechClassification.from_pretrained(
        model_name_or_path).to(device)
    # %%
    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        speech_array = speech_array.squeeze().numpy()
        speech_array = librosa.resample(np.asarray(
            speech_array), sampling_rate, processor.feature_extractor.sampling_rate)


# /home/peterr/macocu/task11/utils.py:159: FutureWarning:
# Pass orig_sr=16000, target_sr=16000 as keyword args.
# From version 0.10 passing these as positional arguments will result in an error
        batch["speech"] = speech_array
        return batch

    def predict(batch):
        features = processor(
            batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

        input_values = features.input_values.to(device)
        attention_mask = features.attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        batch["predicted"] = pred_ids
        return batch

    # %%
    import torchaudio
    import numpy as np
    test_dataset = test_dataset.map(speech_file_to_array_fn)

    # %%
    from dataclasses import dataclass
    from typing import Optional, Tuple
    import torch
    from transformers.file_utils import ModelOutput

    @dataclass
    class SpeechClassifierOutput(ModelOutput):
        loss: Optional[torch.FloatTensor] = None
        logits: torch.FloatTensor = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None

    result = test_dataset.map(predict, batched=True, batch_size=2)

    # %%
    label_names = [config.id2label[i] for i in range(config.num_labels)]

    # %%
    y_true = [config.label2id[name] for name in result[output_column]]
    y_pred = result["predicted"]

    # %%

    y_true = [name for name in result[output_column]]
    y_pred = [config.id2label[name] for name in result["predicted"]]
    return y_true, y_pred


# %%
example_train_config = dict(
    model_name_or_path="facebook/wav2vec2-large-slavic-voxpopuli-v2",
    TASK="age_clf_female_15epochs_v2",
    NUM_EPOCH=15,
    data_files={
        "train": "005_trainv2_females.csv",
        "validation": "005_testv2_females.csv",
    },
    clip_seconds=2
)


def train_model(config_dict: dict) -> str:
    import numpy as np
    import pandas as pd

    from pathlib import Path
    from tqdm import tqdm

    import torchaudio
    from sklearn.model_selection import train_test_split

    import os
    import sys

    model_name_or_path = config_dict.get(
        "model_name_or_path", "facebook/wav2vec2-large-slavic-voxpopuli-v2")

    TASK = config_dict.get("TASK")
    OUTPUT_DIR = "models/" + \
        model_name_or_path.replace("/", "_") + "_" + TASK+"_"
    NUM_EPOCH = config_dict.get("NUM_EPOCH")
    # %%

    # %%
    import torchaudio
    import librosa
    import IPython.display as ipd
    import numpy as np

    from datasets import load_dataset, load_metric

    data_files = config_dict.get("data_files")

    dataset = load_dataset("csv", data_files=data_files)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    # %%
    input_column = config_dict.get("input_column", "path")
    output_column = config_dict.get("output_column")

    label_list = train_dataset.unique(output_column)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")

    # %%
    from transformers import AutoConfig, Wav2Vec2Processor, HubertConfig

    pooling_mode = "mean"

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, 'pooling_mode', pooling_mode)
    # if "hubert" in model_name_or_path:
    #     setattr(config, "add_adapter", True)
    #     if "xlarge" in model_name_or_path:
    #         setattr(config, "output_hidden_size", 1280)
    #     else:
    #         setattr(config, "output_hidden_size", 1024)
    #     setattr(config, "num_adapter_layers", 7)
    #     setattr(config, "adapter_kernel_size", 3)
    #     setattr(config, "adapter_stride", 7)

    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
    except:
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            "./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" "
        )
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor, tokenizer=tokenizer)
    feature_extractor = processor.feature_extractor
    target_sampling_rate = processor.feature_extractor.sampling_rate
    print(f"The target sampling rate: {target_sampling_rate}")

    def speech_file_to_array_fn(path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(
            sampling_rate, target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        clip_seconds = config_dict.get("clip_seconds", -1)
        if clip_seconds == -1:
            return speech
        else:
            return speech[0:clip_seconds * sampling_rate]

    def label_to_id(label, label_list):

        if len(label_list) > 0:
            return label_list.index(label) if label in label_list else -1

        return label

    def preprocess_function(examples):
        speech_list = [speech_file_to_array_fn(
            path) for path in examples[input_column]]
        target_list = [label_to_id(label, label_list)
                       for label in examples[output_column]]

        result = processor(speech_list, sampling_rate=target_sampling_rate)
        result["labels"] = list(target_list)

        return result

    # %%
    train_dataset = train_dataset.map(
        preprocess_function,
        batch_size=100,
        batched=True,
        num_proc=4
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batch_size=100,
        batched=True,
        num_proc=4
    )

    # %%
    from dataclasses import dataclass
    from typing import Optional, Tuple
    import torch
    from transformers.file_utils import ModelOutput

    @dataclass
    class SpeechClassifierOutput(ModelOutput):
        loss: Optional[torch.FloatTensor] = None
        logits: torch.FloatTensor = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None

    # %%
    import torch
    import torch.nn as nn
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2PreTrainedModel,
        Wav2Vec2Model
    )
    from transformers.models.hubert.modeling_hubert import (
        HubertPreTrainedModel,
        HubertModel
    )

    class Wav2Vec2ClassificationHead(nn.Module):
        """Head for wav2vec classification task."""

        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):
            x = features
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)
            return x

    class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.pooling_mode = config.pooling_mode
            self.config = config

            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = Wav2Vec2ClassificationHead(config)

            self.init_weights()

        def freeze_feature_extractor(self):
            self.wav2vec2.feature_extractor._freeze_parameters()

        def merged_strategy(
                self,
                hidden_states,
                mode="mean"
        ):
            if mode == "mean":
                outputs = torch.mean(hidden_states, dim=1)
            elif mode == "sum":
                outputs = torch.sum(hidden_states, dim=1)
            elif mode == "max":
                outputs = torch.max(hidden_states, dim=1)[0]
            else:
                raise Exception(
                    "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

            return outputs

        def forward(
                self,
                input_values,
                attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                labels=None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            outputs = self.wav2vec2(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            hidden_states = self.merged_strategy(
                hidden_states, mode=self.pooling_mode)
            logits = self.classifier(hidden_states)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SpeechClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    class HubertClassificationHead(nn.Module):
        """Head for hubert classification task."""

        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):
            x = features
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)
            return x

    class HubertForSpeechClassification(HubertPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.pooling_mode = config.pooling_mode
            self.config = config

            self.hubert = HubertModel(config)
            self.classifier = HubertClassificationHead(config)

            self.init_weights()

        def freeze_feature_extractor(self):
            self.hubert.feature_extractor._freeze_parameters()

        def merged_strategy(
                self,
                hidden_states,
                mode="mean"
        ):
            if mode == "mean":
                outputs = torch.mean(hidden_states, dim=1)
            elif mode == "sum":
                outputs = torch.sum(hidden_states, dim=1)
            elif mode == "max":
                outputs = torch.max(hidden_states, dim=1)[0]
            else:
                raise Exception(
                    "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

            return outputs

        def forward(
                self,
                input_values,
                attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                labels=None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            outputs = self.hubert(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            hidden_states = self.merged_strategy(
                hidden_states, mode=self.pooling_mode)
            logits = self.classifier(hidden_states)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SpeechClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    from dataclasses import dataclass
    from typing import Dict, List, Optional, Union
    import torch

    import transformers
    from transformers import Wav2Vec2FeatureExtractor

    @dataclass
    class DataCollatorCTCWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`)
                The feature_extractor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
            max_length_labels (:obj:`int`, `optional`):
                Maximum length of the ``labels`` returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
        """

        feature_extractor: Wav2Vec2FeatureExtractor
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": feature["input_values"]}
                              for feature in features]
            label_features = [feature["labels"] for feature in features]

            d_type = torch.long if isinstance(
                label_features[0], int) else torch.float

            batch = self.feature_extractor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            batch["labels"] = torch.tensor(label_features, dtype=d_type)

            return batch

    data_collator = DataCollatorCTCWithPadding(
        feature_extractor=feature_extractor, padding=True)

    import numpy as np
    from transformers import EvalPrediction

    is_regression = False

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(
            p.predictions, tuple) else p.predictions
        preds = np.squeeze(
            preds) if is_regression else np.argmax(preds, axis=1)

        if is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # %%
    model = HubertForSpeechClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )
    model.freeze_feature_extractor()

    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=NUM_EPOCH,
        # save_total_limit=2,
        fp16=True,
        # save_steps=500,
        # eval_steps=100,
        logging_steps=10,
        learning_rate=1e-4,
        save_total_limit=1,
        # Guarantee different seed at every execution
        seed=np.random.randint(0, 100),
    )

    from typing import Any, Dict, Union

    import torch
    from packaging import version
    from torch import nn

    from transformers import (
        Trainer,
        is_apex_available,
    )

    if is_apex_available():
        from apex import amp

    if version.parse(torch.__version__) >= version.parse("1.6"):
        _is_native_amp_available = True
        from torch.cuda.amp import autocast

    class CTCTrainer(Trainer):
        def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
            """
            Perform a training step on a batch of inputs.

            Subclass and override to inject custom behavior.

            Args:
                model (:obj:`nn.Module`):
                    The model to train.
                inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument :obj:`labels`. Check your model's documentation for all accepted arguments.

            Return:
                :obj:`torch.Tensor`: The tensor with training loss on this batch.
            """

            model.train()
            inputs = self._prepare_inputs(inputs)

            if self.use_amp:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.deepspeed:
                self.deepspeed.backward(loss)
            else:
                loss.backward()
            return loss.detach()

    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=feature_extractor,
    )

    # %%
    trainer.train()
    return OUTPUT_DIR
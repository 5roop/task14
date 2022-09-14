import pandas as pd

    
checkpoints = [
    # "facebook/wav2vec2-large-slavic-voxpopuli-v2",
    "facebook/wav2vec2-large-960h-lv60-self",
    # "classla/wav2vec2-large-slavic-parlaspeech-hr",
]
optimal_epochs = {
    # "facebook/wav2vec2-large-960h-lv60-self": 9,
    "facebook/wav2vec2-large-slavic-voxpopuli-v2": 11,
    "classla/wav2vec2-large-slavic-parlaspeech-hr": 7,  
}
from utils import train_model, eval_model
results = []
checkpoint = checkpoints[0]
output_dir = "models/facebook_wav2vec2-large-960h-lv60-self_emotion_optimal_epochs_"
import numpy as np
split = "test"
from pathlib import Path
found_path = str(list(Path(output_dir).glob("checkpoint-*"))[0])
print(f"For checkpoint: {checkpoint} a path was found: ", found_path)
eval_config = dict(
    output_column = "target",
    model_name_or_path= found_path ,
    eval_file= f"010_debug.csv"
)

y_true, y_pred = eval_model(eval_config)

print("Done")

# %%
import pandas as pd

checkpoints = [
    "facebook/wav2vec2-large-slavic-voxpopuli-v2",
    "classla/wav2vec2-large-slavic-parlaspeech-hr",
    "facebook/wav2vec2-large-960h-lv60-self"
]
from utils import train_model, eval_model
results = []
for checkpoint in checkpoints:
    train_config = dict(
        model_name_or_path = checkpoint,
        TASK = "emotion_15epochs",
        NUM_EPOCH = 15,
        output_column = "target",
        input_column = "path",
        data_files = {
            "train": "001_train.csv",
            "validation": "001_dev.csv",
        },
    # clip_seconds = -1
    )
    output_dir = train_model(train_config)
    import torch
    torch.cuda.empty_cache()
    for split in "dev test".split():
        eval_config = dict(
            output_column = "target",
            model_name_or_path= output_dir + "/checkpoint-2640",
            eval_file= f"001_{split}.csv"
        )
        y_true, y_pred = eval_model(eval_config)
        results.append({**eval_config, 
                        **train_config,
                        "split": split,
                        "y_true": y_true, 
                        "y_pred": y_pred})
        pd.DataFrame(data=results).to_json("003_results.jsonl", 
                                           orient="records",
                                           lines=True
                                           )
        import torch
        torch.cuda.empty_cache()

# %%




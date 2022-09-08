# %%
import pandas as pd    
checkpoints = [
    # "facebook/wav2vec2-large-slavic-voxpopuli-v2",
    # "facebook/wav2vec2-large-960h-lv60-self",
    # "classla/wav2vec2-large-slavic-parlaspeech-hr",
    # "facebook/hubert-large-ls960-ft",
    # "facebook/hubert-xlarge-ls960-ft",
    "facebook/hubert-base-ls960",
]
optimal_epochs = {
    "facebook/wav2vec2-large-960h-lv60-self": 9,
    "facebook/wav2vec2-large-slavic-voxpopuli-v2": 11,
    "classla/wav2vec2-large-slavic-parlaspeech-hr": 7,  
    "facebook/hubert-large-ls960-ft": 10,
    "facebook/hubert-xlarge-ls960-ft": 10,
    "facebook/hubert-base-ls960":10,
    
}
from utils import train_model, eval_model
from hubertutils import train_model as hu_train_model, eval_model as hu_eval_model
for checkpoint in checkpoints * 3:
    import os
    os.system("rm -r models/*")
    train_config = dict(
        model_name_or_path = checkpoint,
        TASK = "emotion_10_epochs",
        NUM_EPOCH = optimal_epochs.get(checkpoint),
        output_column = "target",
        input_column = "path",
        data_files = {
            "train": "007_train.csv",
            "validation": "007_dev.csv",
        },
        clip_seconds = 10
    )

    output_dir = hu_train_model(train_config)
    import numpy as np
    from pathlib import Path
    found_path = str(list(Path(output_dir).glob("checkpoint-*"))[0])
    for split in "dev test".split():
        results = []
        # print(f"For checkpoint: {checkpoint} a path was found: ", found_path)
        eval_config = dict(
            output_column = "target",
            model_name_or_path= found_path ,
            eval_file= f"007_{split}.csv"
        )

        y_true, y_pred = hu_eval_model(eval_config)
        import datetime
        time = str(datetime.datetime.now())
        results.append(
            {
                **eval_config, 
                **train_config,
                "split": split,
                "y_true": y_true, 
                "y_pred": y_pred,
                "time": time,
                        })
        content = pd.DataFrame(data=results).to_json(None, 
                                           orient="records",
                                           lines=True,
                                           )
        with open("009_hubert_hyperparams.jsonl", "a") as f:
            f.writelines(content)

# %%




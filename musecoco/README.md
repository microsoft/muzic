# MuseCoco: generating symbolic music from text for composition copilot

# Environment
```bash
conda create -n MuseCoco python=3.8
pip install -r requirements.txt
```

# Training

## Text-to-Attribute Understanding
### 1 Construct attribute-text pairs
1. Attribute: We provide attributes of the standard test set in [text.bin](1-text2attribute_datapreparetest/test.bin).
2. Construct Text:
```bash
cd 1-text2attribute_dataprepare
bash run.sh
```
3. Obtain attribute-text pairs (the input dataset for the text-to-attribute understanding model) including *att_key.json* and *test.json*.
We have provided the off-the-shelf standard test set in the [folder](1-text2attribute_model/text-attribute_understanding/data) too.
### 2. Train the model
```bash
cd 1-text2attribute_model
bash train.sh
```
The checkpoint of the fine-tuned model and *num_labels.json* are obtained.

## Attribute-to-Music Generation

### 1. Data processing
Switch to `2-attribute2music_dataprepare` folder, and set `midi_data_extractor_path` in `config.py` to the path that contains `midi_data_extractor`.

Then, in `data_tool` folder, run the following command to obtain the packed data.

```bash
python extract_data.py path/to/the/folder/containing/midi/files path/to/save/the/dataset
```



**Note:** The tool can only automatically extract the objective attributes' values from MIDI files. If you want to insert values for the subjective attributes' values, please input it manually at L40-L42 in `extract_data.py`.






Prepare  `Token.bin, Token_index.json, RID.bin, RID_index.json` in folder `data/`. Then run the following command to process  the data into `train, validation, test`.

```shell
cd data_process

# The following script splits the midi corpus into "train.txt", "valid.txt" and "test.txt", using "5120" as the maximum length of the token sequence.
python split_data.py

#The following script binarizes the data in fairseq format.
python util.py
```

### 2. Training

Run the following command to train a model with approximately 200M parameters.

```shell
bash train-xl.sh
```



# Inference
## I. Text-to-Attribute Understanding
Switch to `1-text2attribute_model` folder
1. Set *model_name_or_path* as the checkpoint path and *num_labels* as the path of *num_labels.json* in *predict.sh*.
2. Prepare the text, from which attribute values will be extracted, as the format in [predict.json](data/predict.json).
3. Set *test_file* as the path of *predict.json* in *predict.sh*.
4. Then,
    ```bash
    bash predict.sh
    ```
    The *predict_attributes.json* and *softmax_probs.json* are obtained.
5. Preprocess the input of the attribute-to-music generation stage for inference
    After inference, set the path of *predict.json*, *predict_attributes.json*, *softmax_probs.json* and *att_key.json* in *stage2_pre.py* and then,
    ```bash
    python stage2_pre.py
    ```
    The *stage1.bin* is obtained as the inference input of the attribute-to-music generation stage.
## II. Attribute-to-Music Generation
Switch to `2-attribute2music_model` folder
1. Prepare the model checkpoint:

   `checkpoint/linear_mask-xl-truncated_5120/checkpoint_best.pt`

2. Prepare the input for inference in the folder `data/infer_input` from the output of text-to-attribute understanding stage. 

3. Run the following command to generate 64 samples for each input.

```shell
# The following script takes "data/infer_input/infer_test.bin" as input.
bash interactive.sh 0 10 infer_test
# bash interactive.sh start_idx end_idx input_name
```

The generated results are located in the folder `generation/`
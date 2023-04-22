# CLaMP: Contrastive Language-Music Pre-training for Cross-Modal Symbolic Music Information Retrieval
The intellectual property of the CLaMP project is owned by the [Central Conservatory of Music](http://en.ccom.edu.cn/2020/).
## Model description

In [CLaMP: Contrastive Language-Music Pre-training for Cross-Modal Symbolic Music Information Retrieval](https://ai-muzic.github.io/clamp/), we introduce a solution for cross-modal symbolic MIR that utilizes contrastive learning and pre-training. The proposed approach, CLaMP: Contrastive Language-Music Pre-training, which learns cross-modal representations between natural language and symbolic music using a music encoder and a text encoder trained jointly with a contrastive loss. To pre-train CLaMP, we collected a large dataset of 1.4 million music-text pairs. It employed text dropout as a data augmentation technique and bar patching to efficiently represent music data which reduces sequence length to less than 10%. In addition, we developed a masked music model pre-training objective to enhance the music encoder's comprehension of musical context and structure. CLaMP integrates textual information to enable semantic search and zero-shot classification for symbolic music, surpassing the capabilities of previous models. To support the evaluation of semantic search and music classification, we publicly release [WikiMusicText](https://huggingface.co/datasets/sander-wood/wikimusictext) (WikiMT), a dataset of 1010 lead sheets in ABC notation, each accompanied by a title, artist, genre, and description. In comparison to state-of-the-art models that require fine-tuning, zero-shot CLaMP demonstrated comparable or superior performance on score-oriented datasets.

<p align="center"><img src="../img/clamp_clamp.png" width="500"><br/><i>The architecture of CLaMP, including two encoders - one for music and one for text - trained jointly with a contrastive loss to learn cross-modal representations.</i></p>

Two variants of CLaMP are introduced: [CLaMP-S/512](https://huggingface.co/sander-wood/clamp-small-512) and [CLaMP-S/1024](https://huggingface.co/sander-wood/clamp-small-1024). Both models consist of a 6-layer music encoder and a 6-layer text encoder with a hidden size of 768. While CLaMP-S/512 accepts input music sequences of up to 512 tokens in length, CLaMP-S/1024 allows for up to 1024 tokens. The maximum input length for the text encoder in both models is 128 tokens. These models are part of [Muzic](https://github.com/microsoft/muzic), a research initiative on AI music that leverages deep learning and artificial intelligence to enhance music comprehension and generation.

## Cross-Modal Symbolic MIR

CLaMP is capable of aligning symbolic music and natural language, which can be used for various cross-modal retrieval tasks, including semantic search and zero-shot classification for symbolic music.

<p align="center"><img src="../img/clamp_cross-modal tasks.png" width="800"><br/><i>The processes of CLaMP performing cross-modal symbolic MIR tasks, including semantic search and zero-shot classification for symbolic music, without requiring task-specific training data.</i></p>

Semantic search is a technique for retrieving music by open-domain queries, which differs from traditional keyword-based searches that depend on exact matches or meta-information. This involves two steps: 1) extracting music features from all scores in the library, and 2) transforming the query into a text feature. By calculating the similarities between the text feature and the music features, it can efficiently locate the score that best matches the user's query in the library.

Zero-shot classification refers to the classification of new items into any desired label without the need for training data. It involves using a prompt template to provide context for the text encoder. For example, a prompt such as "<i>This piece of music is composed by {composer}.</i>" is utilized to form input texts based on the names of candidate composers. The text encoder then outputs text features based on these input texts. Meanwhile, the music encoder extracts the music feature from the unlabelled target symbolic music. By calculating the similarity between each candidate text feature and the target music feature, the label with the highest similarity is chosen as the predicted one.

## Intended uses:

1. Semantic search and zero-shot classification for score-oriented symbolic music datasets.
2. Cross-modal representation learning between natural language and symbolic music.
3. Enabling research in music analysis, retrieval, and generation.
4. Building innovative systems and applications that integrate music and language.

## Limitations:

1. CLaMP's current version has limited comprehension of performance MIDI.
2. The model may not perform well on tasks outside its pre-training scope.
3. It may require fine-tuning for some specific tasks.

### How to use

To use CLaMP, you can follow these steps:

1. Clone the CLaMP repository by running the following command in your terminal:
```
git clone https://github.com/microsoft/muzic.git
```
This will create a local copy of the repository on your computer.

2. Navigate to the CLaMP directory by running the following command:
```
cd muzic/clamp
```

3. Install the required dependencies by running the following command:
```
pip install -r requirements.txt
```

4. If you are performing a music query, save your query as `inference/music_query.mxl`. For music keys, ensure that all the music files are in the MusicXML (.mxl) format, and are saved in the `inference/music_keys` folder.

5. If you are performing a text query, save your query as `inference/text_query.txt`. For text keys, save all the keys in the `inference/text_keys.txt` file, where each line corresponds to a key.

6. Run the following command to perform the query:
```
python clamp.py -clamp_model_name [MODEL NAME] -query_modal [QUERY MODAL] -key_modal [KEY MODAL] -top_n [NUMBER OF RESULTS]
```
Replace [MODEL NAME] with the name of the CLaMP model you want to use (either `sander-wood/clamp-small-512` or `sander-wood/clamp-small-1024`), [QUERY MODAL] with either `music` or `text` to indicate the type of query you want to perform, [KEY MODAL] with either `music` or `text` to indicate the type of key modal you want to use, and [NUMBER OF RESULTS] with the number of top results you want to return.

For example, to perform semantic music search with the `sander-wood/clamp-small-512` model and return the top 5 results, run:
```
python clamp.py -clamp_model_name sander-wood/clamp-small-512 -query_modal text -key_modal music -top_n 5
```
Note that the first time you run the CLaMP script, it will automatically download the model checkpoint from Hugging Face. This may take a few minutes, depending on your internet speed.

7. After running the command, the script will generate a list of the top results for the given query. Each result correspond to a music file in the `music_keys` folder or a line in the `text_keys.txt` file, depending on the type of key modal you used.

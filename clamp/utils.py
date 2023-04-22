import re
import os
import torch
import requests
from tqdm import tqdm
from unidecode import unidecode
from transformers import AutoModel, AutoConfig, BertModel, PreTrainedModel

# Constants for patch length and number of features in a patch
PATCH_LENGTH = 64
PATCH_FEATURES = 98

class MusicPatchilizer:
    """
    Class for converting music data to patches and vice-versa.

    Attributes:
        delimiters (tuple): A tuple of strings containing the delimiters used for splitting bars.
        regexPattern (str): A regular expression pattern for splitting bars.
        pad_id (int): The id of the padding token.
        mask_id (int): The id of the mask token.
        eos_id (int): The id of the end-of-sequence token.

    Methods:
        split_bars(body): Splits a body of music into individual bars using the delimiters specified in `self.delimiters`.
        bar2patch(bar, patch_length): Encodes a single bar as a patch of specified length.
        patch2bar(patch): Converts a patch to a bar string.
        encode(music, music_length, patch_length=PATCH_LENGTH, add_eos_patch=False): Encodes the input music string as a list of patches.
        decode(patches): Decodes a sequence of patches into a music score.
    """
    def __init__(self):
        # Delimiters used for splitting bars
        self.delimiters = "|:", "::", ":|", "[|", "||", "|]", "|"
        # Regular expression pattern for splitting bars
        self.regexPattern = '('+'|'.join(map(re.escape, self.delimiters))+')'
        # Padding, mask, and end-of-sequence token ids
        self.pad_id = 0
        self.mask_id = 96
        self.eos_id = 97

    def split_bars(self, body):
        """
        Splits a body of music into individual bars using the delimiters specified in `self.delimiters`.

        Args:
            body (str): A string containing the body of music to be split into bars.
        
        Returns:
            list: A list of strings containing the individual bars.
        """
        body = "".join(body)
        bars = re.split(self.regexPattern, body)
        while("" in bars):
            bars.remove("")
        if bars[0] in self.delimiters:
            bars[1] = bars[0]+bars[1]
            bars = bars[1:]
        bars = [bars[i*2]+bars[i*2+1] for i in range(int(len(bars)/2))]

        return bars
    
    def bar2patch(self, bar, patch_length):
        """
        Encodes a single bar as a patch of specified length.
        
        Args:
            bar (str): A string containing the bar to be encoded.
            patch_length (int): An integer indicating the length of the patch to be returned.
        
        Returns:
            list: A list of integer-encoded musical tokens.
        """
        patch = [self.pad_id] * patch_length

        for i in range(min(patch_length, len(bar))):
            chr = bar[i]
            idx = ord(chr)
            if idx>=32 and idx<127:
                patch[i] = idx-31
        
        if i+1<patch_length:
            patch[i+1] = self.eos_id

        return patch
    
    def patch2bar(self, patch):
        """
        Converts a patch to a bar string.

        Args:
            patch (list): A list of integer-encoded musical tokens.

        Returns:
            str: A string containing the decoded bar.
        """
        bar = ""

        for idx in patch:
            if idx>0 and idx<96:
                bar += chr(idx+31)
            else:
                break

        return bar
    
    def encode(self, music, music_length, patch_length=PATCH_LENGTH, add_eos_patch=False):
        """
        Encodes the input music string as a list of patches.

        Args:
            music (str): A string containing the music to be encoded.
            music_length (int): An integer indicating the maximum number of patches to be returned.
            patch_length (int): An integer indicating the length of each patch.
            add_eos_patch (bool): A boolean indicating whether to add an extra patch consisting of all EOS tokens at the end of the encoded music.

        Returns:
            list: A list of integer-encoded patches.
        """
        # Convert to ASCII and split into lines
        music = unidecode(music)
        lines = music.split('\n')
        try:
            lines.remove('')
        except:
            pass

        body = ""
        patches = []

        # Iterate over lines, splitting bars and encoding each one as a patch
        for line in lines:
            # check if the line is a music score line or not
            if len(line)>1 and ((line[0].isalpha() and line[1] == ':') or line.startswith('%%score')):
                # if the current line is a music score line, encode the previous body as patches
                if body!="":
                    bars = self.split_bars(body)
                    
                    for bar in bars:
                        # encode each bar in the body as a patch and append to the patches list
                        patch = self.bar2patch(bar, patch_length)
                        patches.append(patch)
                    # reset the body variable
                    body = ""
                # encode the current line as a patch and append to the patches list
                patch = self.bar2patch(line, patch_length)
                patches.append(patch)
            else:
                # if the line is not a music score line, append to the body variable
                body += line
        
        if body!="":
            bars = self.split_bars(body)

            for bar in bars:
                # encode each bar in the body as a patch and append to the patches list
                patch = self.bar2patch(bar, patch_length)
                patches.append(patch)

        # add an extra patch consisting of all EOS tokens, if required
        if add_eos_patch:
            eos_patch = [self.eos_id] * patch_length
            patches = patches + [eos_patch]

        return patches[:music_length]

    def decode(self, patches):
        """
        Decodes a sequence of patches into a music score.

        Args:
            patches (list): A list of integer-encoded patches.
        
        Returns:
            str: A string containing the decoded music score.
        """
        music = ""
        for patch in patches:
            music += self.patch2bar(patch)+'\n'
        
        return music
    

class MusicEncoder(PreTrainedModel):
    """
    MusicEncoder model for encoding music patches into a sequence of hidden states.

    Args:
        config (:obj:`BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
    
    Attributes:
        patch_embedding (:obj:`torch.nn.Linear`): A linear layer to convert the one-hot encoded patches to the hidden size of the model.
        enc (:obj:`BertModel`): The BERT model used to encode the patches.        
    """
    def __init__(self, config):
        super(MusicEncoder, self).__init__(config)
        self.patch_embedding = torch.nn.Linear(PATCH_LENGTH*PATCH_FEATURES, config.hidden_size)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.enc = BertModel(config=config)
        
    def forward(self, input_musics, music_masks):
        """
        Args:
            input_musics (:obj:`torch.LongTensor` of shape :obj:`(batch_size, music_length, patch_length)`):
                Tensor containing the integer-encoded music patches.
            music_masks (:obj:`torch.LongTensor` of shape :obj:`(batch_size, music_length)`):
                Tensor containing the attention masks for the music patches.
        
        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, music_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
        """
        # One-hot encode the input music patches
        input_musics = torch.nn.functional.one_hot(input_musics, num_classes=PATCH_FEATURES)

        # Reshape the input music patches to feed into the linear layer
        input_musics = input_musics.reshape(len(input_musics), -1, PATCH_LENGTH*PATCH_FEATURES).type(torch.FloatTensor)
        
        # Apply the linear layer to convert the one-hot encoded patches to hidden features
        input_musics = self.patch_embedding(input_musics.to(self.device))
        
        # Apply the BERT model to encode the music data
        output = self.enc(inputs_embeds=input_musics, attention_mask=music_masks.to(self.device))
        
        return output
    
        
class CLaMP(PreTrainedModel):
    """
    CLaMP model for joint text and music encoding.

    Args:
        config (:obj:`BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
        text_model_name (:obj:`str`, `optional`, defaults to :obj:`"distilroberta-base"`):
            The name of the pre-trained text model to be used for text encoding.

    Attributes:
        text_enc (:obj:`AutoModel`): The pre-trained text model used for text encoding.
        text_proj (:obj:`torch.nn.Linear`): A linear layer to project the text encoding to the hidden size of the model.
        music_enc (:obj:`MusicEncoder`): The music encoder model used for music encoding.
        music_proj (:obj:`torch.nn.Linear`): A linear layer to project the music encoding to the hidden size of the model.
    """
    def __init__(self, config, text_model_name="distilroberta-base"):
        super(CLaMP, self).__init__(config)
        self.text_enc = AutoModel.from_pretrained(text_model_name)
        self.text_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        torch.nn.init.normal_(self.text_proj.weight, std=0.02)

        self.music_enc = MusicEncoder(config=config)
        self.music_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        torch.nn.init.normal_(self.music_proj.weight, std=0.02)

    def forward(self, input_texts, text_masks, input_musics, music_masks):
        """
        Args:
            input_texts (:obj:`torch.LongTensor` of shape :obj:`(batch_size, text_length)`):
                Tensor containing the integer-encoded text.
            text_masks (:obj:`torch.LongTensor` of shape :obj:`(batch_size, text_length)`):
                Tensor containing the attention masks for the text.
            input_musics (:obj:`torch.LongTensor` of shape :obj:`(batch_size, music_length, patch_length)`):
                Tensor containing the integer-encoded music patches.
            music_masks (:obj:`torch.LongTensor` of shape :obj:`(batch_size, music_length)`):
                Tensor containing the attention masks for the music patches.
        
        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            music_features (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
                The music features extracted from the music encoder.
            text_features (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
                The text features extracted from the text encoder.
        """
        # Encode input texts
        text_features = self.text_enc(input_texts.to(self.device), attention_mask=text_masks.to(self.device))['last_hidden_state']
        text_features = self.avg_pooling(text_features, text_masks)
        text_features = self.text_proj(text_features)

        # Encode input musics
        music_features = self.music_enc(input_musics, music_masks)['last_hidden_state']
        music_features = self.avg_pooling(music_features, music_masks)
        music_features = self.music_proj(music_features)

        return music_features, text_features
    
    def avg_pooling(self, input_features, input_masks):
        """
        Applies average pooling to the input features.
        
        Args:
            input_features (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length, hidden_size)`):
                Tensor containing the input features.
            input_masks (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_length)`):
                Tensor containing the attention masks for the input features.
        
        Returns:
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`:
                The pooled features.
        """
        input_masks = input_masks.unsqueeze(-1).to(self.device)
        input_features = input_features * input_masks
        avg_pool = input_features.sum(dim=1) / input_masks.sum(dim=1)
        
        return avg_pool
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Instantiate a CLaMP model from a pre-trained model configuration.

        Args:
            pretrained_model_name_or_path (:obj:`str`):
                This can be either:
                    "clamp-small-512" for the small CLaMP model with 512 max sequence length.
                    "clamp-small-1024" for the small CLaMP model with 1024 max sequence length.
        
        Returns:
            :class:`~transformers.CLaMP`: The CLaMP model.
        """
        model_dir = pretrained_model_name_or_path

        # If the pre-trained model is not found locally, download it from Hugging Face
        if not os.path.exists(model_dir):
            # Create the model directory and download the config and pytorch model files
            os.makedirs(model_dir)
            config_url = f"https://huggingface.co/{pretrained_model_name_or_path}/raw/main/config.json"
            model_url = f"https://huggingface.co/{pretrained_model_name_or_path}/resolve/main/pytorch_model.bin"
            chunk_size = 1024 * 1024  # 1MB

            # download config file
            with requests.get(config_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(model_dir+"/config.json", 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading config') as pbar:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            f.write(chunk)
                            pbar.update(len(chunk))

            # download pytorch model file
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(model_dir+"/pytorch_model.bin", 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading model') as pbar:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            f.write(chunk)
                            pbar.update(len(chunk))

        # Load the model weights and configuration
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = cls(config)
        model.load_state_dict(torch.load(pretrained_model_name_or_path+str('/pytorch_model.bin')))

        return model
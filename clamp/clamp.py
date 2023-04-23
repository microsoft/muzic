import argparse
import subprocess
from utils import *
from transformers import AutoTokenizer

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def get_args(parser):
    parser.add_argument('-clamp_model_name', type=str, default='sander-wood/clamp-small-512', help='The CLaMP model name, either "sander-wood/clamp-small-512" or "sander-wood/clamp-small-1024"')
    parser.add_argument('-query_modal', type=str, default='music', help="The query modal, either 'music' or 'text'. If 'music', the input is \"music_query.mxl\", if 'text', the input is \"text_query.txt\"")
    parser.add_argument('-key_modal', type=str, default='text', help="The key modal, either 'music' or 'text'. If 'music', the inputs are all mxl in the \"music_keys\" folder, if 'text', the inputs are all lines in the \"text_keys.txt\" file")
    parser.add_argument('-top_n', type=int, default=10, help="The number of top results to return") 

    return parser

# parse arguments
args = get_args(argparse.ArgumentParser()).parse_args()
CLAMP_MODEL_NAME = args.clamp_model_name
QUERY_MODAL = args.query_modal
KEY_MODAL = args.key_modal
TOP_N = args.top_n
TEXT_MODEL_NAME = 'distilroberta-base'
TEXT_LENGTH = 128

# load CLaMP model
model = CLaMP.from_pretrained(CLAMP_MODEL_NAME)
music_length = model.config.max_length
model = model.to(device)
model.eval()

# initialize patchilizer, tokenizer, and softmax
patchilizer = MusicPatchilizer()
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
softmax = torch.nn.Softmax(dim=1)

def compute_values(Q_e, K_e, t=1):
    """
    Compute the values for the attention matrix

    Args:
        Q_e (torch.Tensor): Query embeddings
        K_e (torch.Tensor): Key embeddings
        t (float): Temperature for the softmax
    
    Returns:
        values (torch.Tensor): Values for the attention matrix
    """
    # Normalize the feature representations
    Q_e = torch.nn.functional.normalize(Q_e, dim=1)
    K_e = torch.nn.functional.normalize(K_e, dim=1)

    # Scaled pairwise cosine similarities [1, n]
    logits = torch.mm(Q_e, K_e.T) * torch.exp(torch.tensor(t))
    values = softmax(logits)
    return values.squeeze()


def encoding_data(data, modal):
    """
    Encode the data into ids

    Args:
        data (list): List of strings
        modal (str): "music" or "text"
    
    Returns:
        ids_list (list): List of ids
    """
    ids_list = []
    if modal=="music":
        for item in data:
            patches = patchilizer.encode(item, music_length=music_length, add_eos_patch=True)
            ids_list.append(torch.tensor(patches).reshape(-1))
    else:
        for item in data:
            text_encodings = tokenizer(item, 
                                        return_tensors='pt', 
                                        truncation=True, 
                                        max_length=TEXT_LENGTH)
            ids_list.append(text_encodings['input_ids'].squeeze(0))

    return ids_list


def abc_filter(lines):
    """
    Filter out the metadata from the abc file

    Args:
        lines (list): List of lines in the abc file
    
    Returns:
        music (str): Music string
    """
    music = ""
    for line in lines:
        if line[:2] in ['A:', 'B:', 'C:', 'D:', 'F:', 'G', 'H:', 'N:', 'O:', 'R:', 'r:', 'S:', 'T:', 'W:', 'w:', 'X:', 'Z:'] \
        or line=='\n' \
        or (line.startswith('%') and not line.startswith('%%score')):
            continue
        else:
            if "%" in line and not line.startswith('%%score'):
                line = "%".join(line.split('%')[:-1])
                music += line[:-1] + '\n'
            else:
                music += line + '\n'
    return music


def load_music(filename):
    """
    Load the music from the xml file

    Args:
        filename (str): Path to the xml file

    Returns:
        music (str): Music string
    """
    p = subprocess.Popen('cmd /u /c python inference/xml2abc.py -m 2 -c 6 -x "'+filename+'"', stdout=subprocess.PIPE)
    result = p.communicate()
    output = result[0].decode('utf-8').replace('\r', '')
    music = unidecode(output).split('\n')
    music = abc_filter(music)

    return music


def get_features(ids_list, modal):
    """
    Get the features from the CLaMP model

    Args:
        ids_list (list): List of ids
        modal (str): "music" or "text"
    
    Returns:
        features_list (torch.Tensor): Tensor of features with a shape of (batch_size, hidden_size)
    """
    features_list = []
    print("Extracting "+modal+" features...")
    with torch.no_grad():
        for ids in tqdm(ids_list):
            ids = ids.unsqueeze(0)
            if modal=="text":
                masks = torch.tensor([1]*len(ids[0])).unsqueeze(0)
                features = model.text_enc(ids.to(device), attention_mask=masks.to(device))['last_hidden_state']
                features = model.avg_pooling(features, masks)
                features = model.text_proj(features)
            else:
                masks = torch.tensor([1]*(int(len(ids[0])/PATCH_LENGTH))).unsqueeze(0)
                features = model.music_enc(ids, masks)['last_hidden_state']
                features = model.avg_pooling(features, masks)
                features = model.music_proj(features)

            features_list.append(features[0])
    
    return torch.stack(features_list).to(device)


if __name__ == "__main__":
    # load query
    if QUERY_MODAL=="music":
        query = load_music("inference/music_query.mxl")
    else:
        with open("inference/text_query.txt", 'r', encoding='utf-8') as f:
            query = f.read()
    query = unidecode(query)

    # load keys
    keys = []
    key_filenames = []

    if KEY_MODAL=="music":
        # load music keys
        for root, dirs, files in os.walk("inference/music_keys"):
            for file in files:
                filename = root+"/"+file
                if filename.endswith(".mxl"):
                    key_filenames.append(filename)
        print("Loading music...")

        # load keys if the pth file exists
        if os.path.exists("inference/cache/"+KEY_MODAL+"_key_cache_"+str(music_length)+".pth"):
            with open("inference/cache/"+KEY_MODAL+"_key_cache_"+str(music_length)+".pth", 'rb') as f:
                key_cache = torch.load(f)
            cached_keys = key_cache["keys"]
            cached_key_filenames = key_cache["filenames"]
            cached_key_features = key_cache["features"]
            
            # remove cache that are not in the key_filenames
            files_to_remove = []
            for i, key_filename in enumerate(cached_key_filenames):
                if key_filename not in key_filenames:
                    files_to_remove.append(i)
            
            cached_keys = [key for i, key in enumerate(cached_keys) if i not in files_to_remove]
            cached_key_filenames = [filename for i, filename in enumerate(cached_key_filenames) if i not in files_to_remove]
            cached_key_features = [feature for i, feature in enumerate(cached_key_features) if i not in files_to_remove]

            if len(cached_key_features) > 0:
                cached_key_features = torch.stack(cached_key_features).to(device)

            # only keep files that are not in the cache
            key_filenames = [filename for filename in key_filenames if filename not in cached_key_filenames]

        for filename in tqdm(key_filenames):
            key = unidecode(load_music(filename))
            keys.append(key)
            
        non_empty_keys = []
        non_empty_filenames = []

        for key, filename in zip(keys, key_filenames):
            if key.strip()!="":
                non_empty_keys.append(key)
                non_empty_filenames.append(filename)
            else:
                print("File %s not successfully loaded" %(filename))
        
        keys = non_empty_keys
        key_filenames = non_empty_filenames

    else:
        with open("inference/text_keys.txt", 'r', encoding='utf-8') as f:
            inference_text = unidecode(f.read())
        for key in inference_text.split("\n"):
            if key.strip()!="":
                keys.append(key.strip())
        
        # load text keys
        if os.path.exists("inference/cache/"+KEY_MODAL+"_key_cache_"+str(music_length)+".pth"):
            with open("inference/cache/"+KEY_MODAL+"_key_cache_"+str(music_length)+".pth", 'rb') as f:
                key_cache = torch.load(f)
            cached_keys = key_cache["keys"]
            cached_key_filenames = key_cache["filenames"]
            cached_key_features = key_cache["features"]
        
            # remove cache that are not in the keys
            files_to_remove = []
            for i, key in enumerate(cached_keys):
                if key not in keys:
                    files_to_remove.append(i)
            
            cached_keys = [key for i, key in enumerate(cached_keys) if i not in files_to_remove]
            cached_key_filenames = [filename for i, filename in enumerate(cached_key_filenames) if i not in files_to_remove]
            cached_key_features = [feature for i, feature in enumerate(cached_key_features) if i not in files_to_remove]

            if len(cached_key_features)>0:
                cached_key_features = torch.stack(cached_key_features).to(device)
            
            # only keep keys that are not in the cache
            keys = [key for key in keys if key not in cached_keys]
            
    # encode keys
    if len(keys)>0:
        key_ids = encoding_data(keys, KEY_MODAL)
        key_features = get_features(key_ids, KEY_MODAL)

    # merge cache with new keys
    if os.path.exists("inference/cache/"+KEY_MODAL+"_key_cache_"+str(music_length)+".pth"):
        if len(keys)>0:
            keys = cached_keys + keys
            key_filenames = cached_key_filenames + key_filenames
            if len(cached_key_features)>0:
                key_features = torch.cat((cached_key_features, key_features), dim=0)
        else:
            keys = cached_keys
            key_filenames = cached_key_filenames
            key_features = cached_key_features
    key_cache = {"keys": keys, "filenames": key_filenames, "features": key_features}
        
    # save key cache as pth file
    if not os.path.exists("inference/cache"):
        os.makedirs("inference/cache")
    with open("inference/cache/"+KEY_MODAL+"_key_cache_"+str(music_length)+".pth", 'wb') as f:
        torch.save(key_cache, f)

    # encode query
    query_ids = encoding_data([query], QUERY_MODAL)
    query_feature = get_features(query_ids, QUERY_MODAL)

    # compute values
    values = compute_values(query_feature, key_features)
    sims = torch.cosine_similarity(query_feature, key_features)

    # sort and print results
    print("\n")

    if TOP_N==0:
        TOP_N = len(values)

    for idx in torch.argsort(values)[-TOP_N:]:
        prob = values[idx].item()*100
        sim = sims[idx].item()
        if KEY_MODAL=="text":
            content = keys[idx]
        else:
            content = key_filenames[idx]

        print("Prob: %.2f%% - Sim: %.4f:\n%s\n" % ((prob, sim, content)))
    
    # print query if text
    if QUERY_MODAL=="text":
        print("Query:\n%s" %(query.strip()))
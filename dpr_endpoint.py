import faiss
import pickle
import numpy as np
import torch
from torch import nn
import json
from flask import Flask
from flask import request
from argparse import ArgumentParser
from dpr.models import init_biencoder_components
from dpr.options import (
    add_encoder_params,
    setup_args_gpu,
    print_args,
    set_encoder_params_from_state,
    add_tokenizer_params,
    add_cuda_params,
)
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    load_states_from_checkpoint,
)

app = Flask(__name__)
q_model: nn.Module = None
tokenizer: Tensorizer = None
indexer = None
indexer_meta = None
args = None


@app.route('/', methods=['POST'])
def dpr_search():
    r = request.json
    question = r['question']

    with torch.no_grad():
        input_ids = tokenizer.text_to_tensor(question).view(1, -1).to(args.device)
        token_type_ids = torch.zeros_like(input_ids).to(args.device)
        attention_mask = tokenizer.get_attn_mask(input_ids).to(args.device)

        _, question_embedding, _ = q_model(input_ids=input_ids,
                                           token_type_ids=token_type_ids,
                                           attention_mask=attention_mask)
        question_embedding = question_embedding.detach().cpu().numpy().reshape(1, -1)

    dists, idxs = indexer.search(question_embedding, args.topn)
    result = [indexer_meta[idx] for idx in idxs[0]]
    return json.dumps(result)


def load_model(args):
    global q_model
    global tokenizer
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    tokenizer, encoder, _ = init_biencoder_components(
        args.encoder_model_type, args, inference_only=True
    )
    tokenizer.set_pad_to_max(False)

    q_model = encoder.question_model
    q_model, _ = setup_for_distributed_mode(
        q_model, None, args.device, args.n_gpu, args.local_rank, args.fp16
    )
    q_model.eval()

    # load state dict
    # take care of distributed mode (very unlikely)
    model = q_model.module if hasattr(q_model, 'module') else q_model
    prefix = 'question_model.'
    prefix_len = len(prefix)
    question_model_state_dict = {
        param_name[prefix_len:]: state for param_name, state in saved_state.model_dict.items()
        if param_name.startswith(prefix)
    }
    model.load_state_dict(question_model_state_dict)


if __name__ == '__main__':
    parser = ArgumentParser()
    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)
    parser.add_argument('--port', default=8000, type=int)
    parser.add_argument('--indexer_path', default=None, type=str)
    parser.add_argument('--embedding_path', default=None, type=str)
    parser.add_argument('--topn', default=20, type=int)
    parser.add_argument('--save_indexer', action='store_true')
    parser.add_argument('--saved_indexer_path', default='flat.index', type=str)
    args = parser.parse_args()

    setup_args_gpu(args)
    print_args(args)

    load_model(args)

    if args.indexer_path:
        indexer = faiss.read_index(args.indexer_path)
        with open(f'{args.indexer_path}.meta', 'rb') as f:
            indexer_meta = pickle.load(f)
    elif args.embedding_path:
        indexer = faiss.IndexFlatIP(q_model.get_out_size())
        with open(args.embedding_path, 'rb') as f:
            indexer_meta, vectors = list(zip(*pickle.load(f)))
            vectors = np.array(vectors)
            indexer.add(vectors)
        if args.save_indexer:
            faiss.write_index(indexer, args.saved_indexer_path)
            with open(f'{args.saved_indexer_path}.meta', 'wb') as f:
                pickle.dump(indexer_meta, f)
    else:
        raise ValueError('Either indexer_path or embedding_path should be set')

    app.run(host='0.0.0.0', port=args.port)

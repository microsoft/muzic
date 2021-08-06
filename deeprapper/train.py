# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License. 
#
 # -*- coding: utf-8 -*

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
from tqdm import tqdm
from utils import swap_value


def main():
    parser = argparse.ArgumentParser()
    
    # 各种数据路径
    parser.add_argument('--model_dir', default='model', type=str, required=False, help='模型存放位置')
    parser.add_argument('--root_path', default='data/lyrics/', type=str, required=False, help='根目录')
    parser.add_argument('--raw_data_dir', default='lyric_with_final_small', type=str, required=False, help='原始数据目录名称')
    parser.add_argument('--model_sign', default='1a', type=str, required=False, help='模型签名: 区分模型和log存储子目录')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    
    # 各种语料库
    parser.add_argument('--tokenizer_path', default='tokenizations/chinese_dicts.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--finalizer_path', default='tokenizations/finals.txt', type=str, required=False, help='选择韵母词库')
    parser.add_argument('--sentencer_path', default='tokenizations/sentences.txt', type=str, required=False, help='选择句子词库')
    parser.add_argument('--poser_path', default='tokenizations/sentences.txt', type=str, required=False, help='选择相对位置词库')
    parser.add_argument('--beater_path', default='tokenizations/beats.txt', type=str, required=False, help='选择鼓点词库')
    
    # 训练参数
    parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--init_device', default=0, type=int, required=False, help='设置使用主显卡')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--start_epoch', default=0, type=int, required=False, help='从哪个epoch开始训练')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=10, type=int, required=False,
                        help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=1024, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=1, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=0, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    
    # 数据处理方式
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json" , required=False)
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe" , required=False)
    parser.add_argument('--raw', action='store_true', help='是否从preprocessing开始', required=False)
    parser.add_argument('--tokenize', action='store_true', help='是否作tokenize', required=False)
    parser.add_argument('--segment', action='store_true', help='中文以词为单位', required=False)
    parser.add_argument('--bpe_token', action='store_true', help='subword', required=False)
    parser.add_argument('--enable_final', action='store_true', help='是否加入韵母embedding', required=False)
    parser.add_argument('--enable_sentence', action='store_true', help='是否加入sentence embedding', required=False)
    parser.add_argument('--enable_relative_pos', action='store_true', help='是否加入inner-sentence positional embedding', required=False)
    parser.add_argument('--enable_beat', action='store_true', help='是否加入beat embedding', required=False)
    parser.add_argument('--reverse', action='store_true', help='是否采用反向生成', required=False)
    parser.add_argument('--with_beat', action='store_true', help='是否同时生成beat', required=False)
    parser.add_argument('--beat_mode', default=0, type=int, help='beat控制模式：0.不控制；1.global；2.local', required=False)

    args = parser.parse_args()
    print('args:\n' + args.__repr__())
    
    
    ########################################################    
    # basic settings
    ###################################
    # set envs and import related packages
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    import torch
    import transformers
    from torch.nn import DataParallel
    from torch.utils.tensorboard import SummaryWriter
    from prepare_train_data import build_files_separate, read_lyrics, prepare_lyrics, get_shuffled_samples
    from tokenizations.bpe_tokenizer import get_encoder
    from module import GPT2Config, GPT2Model, GPT2LMHeadModel
    
    # choose tokenizer
    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert
    # set tokenizer
    if args.bpe_token:
        full_tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
        full_tokenizer.max_len = 999999
    else:
        full_tokenizer = tokenization_bert.BertTokenizer(
            vocab_file=args.tokenizer_path, 
            do_lower_case=False
        )
        full_finalizer = tokenization_bert.BertTokenizer(
            vocab_file=args.finalizer_path, 
            tokenize_chinese_chars=False, 
            do_lower_case=False
        )
        full_sentencer = tokenization_bert.BertTokenizer(
            vocab_file=args.sentencer_path, 
            tokenize_chinese_chars=False, 
            do_lower_case=False
        )
        full_poser = tokenization_bert.BertTokenizer(
            vocab_file=args.poser_path, 
            tokenize_chinese_chars=False, 
            do_lower_case=False
        )
        full_beater = tokenization_bert.BertTokenizer(
            vocab_file=args.beater_path, 
            tokenize_chinese_chars=False, 
            do_lower_case=False
        )
              
    ############################################    
    # run tokenizeing
    ###################################
    # dataset root key
    key = args.root_path.rstrip('/').split('/')[-1] 
    # processed data root path
    processed_path = os.path.join(args.root_path, args.raw_data_dir, 'processed')
    
    tokenized_path = os.path.join(processed_path, 'tokenized')
    reverse_str = '_reverse' if args.reverse else ''
    tokenized_data_path = os.path.join(tokenized_path, f'tokenized{reverse_str}')
    finalized_data_path = os.path.join(tokenized_path, f'finalized{reverse_str}')
    sentenced_data_path = os.path.join(tokenized_path, f'sentenced{reverse_str}')
    posed_data_path = os.path.join(tokenized_path, f'posed{reverse_str}')
    beated_data_path = os.path.join(tokenized_path, f'beated{reverse_str}')
            
    if args.tokenize:
        # prepare data
        if args.raw:  
            print('Processing from raw data...') 
            prepare_fn = {
                'lyrics': prepare_lyrics
            }
            prepare_fn[key](
                ins_path=os.path.join(args.root_path, args.raw_data_dir, 'raw'), # demo: data/lyrics/lyrics_22w/raw
                out_path=processed_path,  # demo: data/lyrics/lyrics_22w/processed
                with_beat=args.with_beat, 
                beat_mode=args.beat_mode
            )

        print('Loading processed data for training...')
        read_fn = {
            'lyrics': read_lyrics,
        } 
        train_lines, train_finals, train_sentences, train_pos, train_beats = read_fn[key](processed_path, reverse=args.reverse)
        
        print('Tokenizing processed data for training...')
        build_files_separate(num_pieces=args.num_pieces,
                    stride=args.stride,
                    min_length=args.min_length,
                    lines=train_lines, 
                    finals=train_finals,
                    sentences=train_sentences,
                    pos=train_pos,
                    beats=train_beats,
                    tokenized_data_path=tokenized_data_path,
                    finalized_data_path=finalized_data_path,
                    sentenced_data_path=sentenced_data_path,
                    posed_data_path=posed_data_path,
                    beated_data_path=beated_data_path,
                    full_tokenizer=full_tokenizer,
                    full_finalizer=full_finalizer,
                    full_sentencer=full_sentencer,
                    full_poser=full_poser,
                    full_beater=full_beater,
                    enable_final=args.enable_final,
                    enable_sentence=args.enable_sentence,
                    enable_pos=args.enable_relative_pos,
                    enable_beat=args.enable_beat,
                    segment=args.segment)
        
        print('End')

        
    ######################################
    # Training settings
    ################################
    # calculate total training steps
    full_len = 0
    print('calculating total steps')
    for i in tqdm(range(args.num_pieces)):
        with open(os.path.join(tokenized_data_path, 'tokenized_train_{}.txt'.format(i)), 'r') as f:
            full_len += len([int(item) for item in f.read().strip().split()])
    total_steps = int(full_len / args.stride * args.epochs / args.batch_size / args.gradient_accumulation)
    print('total steps = {}'.format(total_steps))

    # build model
    model_config = GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())
    if not args.pretrained_model:
        model = GPT2LMHeadModel(config=model_config)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.train()
    # set whether to use cuda
    gpu_count = torch.cuda.device_count()
    if gpu_count > 0:
        device_ids = [int(i) for i in range(gpu_count)]
        swap_value(device_ids, 0, args.init_device)
        device = f'cuda:{device_ids[0]}'
    else:
        device = 'cpu'
    print('using device:', device)
    model.to(device)
    

    # check parameters number of the built model
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    # set optimizer
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    # change WarmupLinearSchedule to get_linear_schedule_with_warmup for current version of Transformers
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                                             num_warmup_steps=args.warmup_steps,
                                                             num_training_steps=total_steps)
    
    # set whether to use 16-bits parameters to save GPU memory if your GPU support the operations of 16-bits number
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    
    # set whether to use multi GPUs
    multi_gpu = False
    if gpu_count > 1:
        print("Let's use", gpu_count, "GPUs!", device_ids)
        model = DataParallel(model, device_ids=device_ids)
        multi_gpu = True
        
    # set log info
    log_dir = os.path.join(args.writer_dir, key, f'{args.raw_data_dir}{reverse_str}', args.model_sign)
    tb_writer = SummaryWriter(log_dir=log_dir)
    assert args.log_step % args.gradient_accumulation == 0
    
    output_dir = os.path.join(args.model_dir, key, f'{args.raw_data_dir}{reverse_str}', args.model_sign)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print('starting training')
    overall_step = 0
    running_loss = 0
    for epoch in range(args.start_epoch, args.epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))

        # shuffle pieces of data
        x = np.linspace(0, args.num_pieces - 1, args.num_pieces, dtype=np.int32)
        random.shuffle(x)

        piece_num = 0
        
        # enumerate data pieces
        for i in x:
            # prepare training sentences
            with open(os.path.join(tokenized_data_path, 'tokenized_train_{}.txt'.format(i)), 'r') as f:
                line = f.read().strip()
            tokens = line.split()
            # print(len(tokens))
            tokens = [int(token) for token in tokens]
            # tokens = torch.Tensor(tokens)
            
            if args.enable_final:
                with open(os.path.join(finalized_data_path, 'tokenized_train_{}.txt'.format(i)), 'r') as f:
                    final = f.read().strip()
                finals = final.split()
                # print(len(finals))
                finals = [int(final) for final in finals]
                # finals = torch.Tensor(finals)
                
            if args.enable_sentence:
                with open(os.path.join(sentenced_data_path, 'tokenized_train_{}.txt'.format(i)), 'r') as f:
                    sentence = f.read().strip()
                sentences = sentence.split()
                # print(len(sentences))
                sentences = [int(sentence) for sentence in sentences]
                # sentences = torch.Tensor(sentences)
            
            if args.enable_relative_pos:
                with open(os.path.join(posed_data_path, 'tokenized_train_{}.txt'.format(i)), 'r') as f:
                    p = f.read().strip()
                pos = p.split()
                # print(len(sentences))
                pos = [int(p) for p in pos]
                # sentences = torch.Tensor(sentences)
            
            if args.enable_beat:
                with open(os.path.join(beated_data_path, 'tokenized_train_{}.txt'.format(i)), 'r') as f:
                    beat = f.read().strip()
                beats = beat.split()
                # print(len(sentences))
                beats = [int(beat) for beat in beats]
                # sentences = torch.Tensor(sentences)
            # print('training: ', len(tokens), len(finals), len(sentences))
            
            start_point = 0
            samples_token, samples_final, samples_sentence, samples_pos, samples_beat = [], [], [], [], []
            n_ctx = model_config.n_ctx  # the length of a sentence for training
            stride = args.stride
            print(len(tokens))
            while start_point < len(tokens) - stride:
                samples_token.append(tokens[start_point: start_point + stride])
                if args.enable_final:
                    samples_final.append(finals[start_point: start_point + stride])
                if args.enable_sentence:
                    samples_sentence.append(sentences[start_point: start_point + stride])
                if args.enable_relative_pos:
                    samples_pos.append(pos[start_point: start_point + stride])
                if args.enable_beat:
                    samples_beat.append(beats[start_point: start_point + stride])
                start_point += stride
            if start_point < len(tokens):
                samples_token.append(tokens[len(tokens) - stride:])
                if args.enable_final:
                    samples_final.append(finals[len(tokens) - stride:])
                if args.enable_sentence:
                    samples_sentence.append(sentences[len(tokens) - stride:])
                if args.enable_relative_pos:
                    samples_pos.append(pos[len(tokens) - stride:])
                if args.enable_beat:   
                    samples_beat.append(beats[len(tokens) - stride:])
            
            
            samples_token, samples_final, samples_sentence, samples_pos, samples_beat = get_shuffled_samples(
                samples_token, samples_final, 
                samples_sentence, samples_pos, samples_beat
            )
#             print(len(samples_token), len(samples_final), len(samples_sentence), len(samples_))

            # enumerate batch data
            for step in range(len(samples_token) // args.batch_size):  # drop last

                #  prepare batch data
                batch_token = samples_token[step * args.batch_size: (step + 1) * args.batch_size]
                batch_inputs_token = torch.Tensor(batch_token).long().to(device)
                
                if samples_final is not None:
                    batch_final = samples_final[step * args.batch_size: (step + 1) * args.batch_size]
                    batch_inputs_final = torch.Tensor(batch_final).long().to(device)
                else:
                    batch_inputs_final = None
                
                if samples_sentence is not None:
                    batch_sentence = samples_sentence[step * args.batch_size: (step + 1) * args.batch_size]
                    batch_inputs_sentence = torch.Tensor(batch_sentence).long().to(device)
                else:
                    batch_inputs_sentence = None
                
                if samples_pos is not None:
                    batch_pos = samples_pos[step * args.batch_size: (step + 1) * args.batch_size]
                    batch_inputs_pos = torch.Tensor(batch_pos).long().to(device)
                else:
                    batch_inputs_pos = None
                
                if samples_beat is not None:
                    batch_beat = samples_beat[step * args.batch_size: (step + 1) * args.batch_size]
                    batch_inputs_beat = torch.Tensor(batch_beat).long().to(device)
                else:
                    batch_inputs_beat = None
 
                #  forward pass
                # Notes: Using Transformers, the labels are shifted inside the model,
                #           i.e. you can set labels = input_ids
                outputs = model.forward(input_ids=batch_inputs_token, 
                                        sentence_ids=batch_inputs_sentence, 
                                        final_ids=batch_inputs_final,
                                        pos_ids=batch_inputs_pos,
                                        beat_ids=batch_inputs_beat,
                                        labels=batch_inputs_token)
                loss, logits = outputs[:2]

                #  get loss
                if multi_gpu:
                    loss = loss.mean()
                    '''
                    running_loss += loss
                    overall_step += 1
                    '''
                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation

                #  loss backward
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                #  optimizer step
                if (overall_step + 1) % args.gradient_accumulation == 0:
                    running_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                # log info of training process
                if (overall_step + 1) % args.log_step == 0:
                    loss_log = running_loss * args.gradient_accumulation / (args.log_step / args.gradient_accumulation)
                    tb_writer.add_scalar('loss', loss_log, overall_step)
                    print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {}'.format(datetime.now().hour, 
                                                                                             datetime.now().minute,
                                                                                             step + 1, piece_num, 
                                                                                             epoch + 1, loss_log))
                    running_loss = 0

                overall_step += 1

            piece_num += 1

        # save model per epoch
        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(os.path.join(output_dir, 'model_epoch{}'.format(epoch + 1))):
            os.mkdir(os.path.join(output_dir, 'model_epoch{}'.format(epoch + 1)))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(os.path.join(output_dir, 'model_epoch{}'.format(epoch + 1)))
        # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
        # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    # save final model
    print('training finished')
    if not os.path.exists(os.path.join(output_dir, 'final_model')):
        os.mkdir(os.path.join(output_dir, 'final_model'))
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(os.path.join(output_dir, 'final_model'))
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    main()

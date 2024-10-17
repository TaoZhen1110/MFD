import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import os
import glob
from datetime import datetime
import socket
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
from transformers import AlbertTokenizer
import json
from dataloader import data_encoder, dataset_loader
from model import ALBERT_Encoder
import torch.nn.functional as F


def get_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='0,1,6,7')
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-epochs', type=int, default=30)
    parser.add_argument('-resume_epoch', type=int, default=0)
    parser.add_argument('-log_every', type=int, default=20)
    parser.add_argument('-naver_grad', type=int, default=1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-encoder_model_path', type=str,
                        default='/mnt/data132/taozhen/AI_Thesis_Detection/Framework/Contrastive_Model/Different_Encoder/ALBERT/albert_pretrained/')
    parser.add_argument('-train_dataset_path', type=str,
                        default='/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/HPPT/Llama3.1/train.json')
    parser.add_argument('-val_dataset_path', type=str,
                        default='/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/HPPT/Llama3.1/val.json')

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


#################  Loss function  #################
# def twice_triple_loss(anchor, positive1, positive2, negative, margin=1.0):
#     pos_dist1 = 1 - nn.functional.cosine_similarity(anchor, positive1)
#     pos_dist2 = 1 - nn.functional.cosine_similarity(anchor, positive2)
#     neg_dist = 1 - nn.functional.cosine_similarity(anchor, negative)
#
#     loss1 = torch.clamp(margin + pos_dist1 - neg_dist, min=0.0)
#     loss2 = torch.clamp(margin + pos_dist2 - neg_dist, min=0.0)
#
#     loss = loss1 + loss2
#
#     return loss.mean()


def twice_triple_loss(anchor, positive1, positive2, negative, margin=1.0):
    # 计算 anchor 和两个正样本的余弦距离 (1 - cosine_similarity)
    pos_dist1 = 1 - F.cosine_similarity(anchor, positive1)
    pos_dist2 = 1 - F.cosine_similarity(anchor, positive2)

    # 计算 anchor 和负样本的余弦距离
    neg_dist = 1 - F.cosine_similarity(anchor, negative)

    # 计算正样本与负样本之间的余弦距离，增强负样本效用
    pos_neg_dist1 = 1 - F.cosine_similarity(positive1, negative)
    pos_neg_dist2 = 1 - F.cosine_similarity(positive2, negative)

    # 计算损失，使用 margin 和 clamp 避免负损失
    loss1 = torch.clamp(margin + pos_dist1 - neg_dist, min=0.0)
    loss2 = torch.clamp(margin + pos_dist2 - neg_dist, min=0.0)

    # 添加正样本和负样本之间的距离损失
    loss1 += torch.clamp(margin + pos_dist1 - pos_neg_dist1, min=0.0)
    loss2 += torch.clamp(margin + pos_dist2 - pos_neg_dist2, min=0.0)

    # 平均两个正样本的损失
    loss = (loss1 + loss2) / 2

    return loss.mean()



def main(args):
    setup_seed(1234)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    #################  Train model save setting  #################
    save_dir_root = os.path.dirname(os.path.abspath(__file__))
    # os.path.abspath:取当前文件的绝对路径（完整路径）
    # os.path.dirname:去掉文件名，返回目录

    if args.resume_epoch != 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'albert_save', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'albert_save', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    save_dir = os.path.join(save_dir_root, 'albert_save', 'run_' + str(run_id))
    log_dir = os.path.join(save_dir, datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)  # 将数据以特定的格式存储到上面得到的那个日志文件夹中


    ###################  Dataset prepare #################
    tokenizer = AlbertTokenizer.from_pretrained(args.encoder_model_path)

    train_data = []
    with open(args.train_dataset_path) as f:
        for line in f:
            train_data.append(json.loads(line))  # 逐行加载 JSON 对象
    train_loader = dataset_loader(dataset=data_encoder(jsondata=train_data, tokenizer=tokenizer),
                                  batch_size=args.batch_size, shuffle=True)

    val_data = []
    with open(args.val_dataset_path) as f:
        for line in f:
            val_data.append(json.loads(line))
    val_loader = dataset_loader(dataset=data_encoder(jsondata=val_data, tokenizer=tokenizer),
                                batch_size=1, shuffle=False)

    num_iter_tr = len(train_loader)
    num_iter_ts = len(val_loader)

    nitrs = args.resume_epoch * num_iter_tr
    nsamples = 0


    ########################  Model ###########################
    model = ALBERT_Encoder(args.encoder_model_path)
    model = nn.DataParallel(model).cuda()

    ##################  优化 Setting #################
    parameters_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters_update, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


    ##################  Train Model  ###################
    sample_num = 0
    recent_losses = []
    aveGrad = 0
    best_f, curf = 10000.0, 0.0
    start_time = time.time()


    for epoch in tqdm(range(args.resume_epoch, args.epochs)):
        model.train()
        epoch_losses = []

        for step, sample_batched in enumerate(train_loader):
            sample_batched = {key: value.cuda() for key, value in sample_batched.items()}
            input_ids_1 = sample_batched['input_ids_1']
            attention_mask_1 = sample_batched['attention_mask_1']
            token_type_ids_1 = sample_batched['token_type_ids_1']
            input_ids_2 = sample_batched['input_ids_2']
            attention_mask_2 = sample_batched['attention_mask_2']
            token_type_ids_2 = sample_batched['token_type_ids_2']
            input_ids_3 = sample_batched['input_ids_3']
            attention_mask_3 = sample_batched['attention_mask_3']
            token_type_ids_3 = sample_batched['token_type_ids_3']
            input_ids_4 = sample_batched['input_ids_4']
            attention_mask_4 = sample_batched['attention_mask_4']
            token_type_ids_4 = sample_batched['token_type_ids_4']

            sample_num += input_ids_1.shape[0]

            Human_text_encoder = model(input_ids=input_ids_1, attention_mask=attention_mask_1, token_type_ids=token_type_ids_1)
            AI_text_encoder = model(input_ids=input_ids_2, attention_mask=attention_mask_2, token_type_ids=token_type_ids_2)
            AI_rewrite_encoder = model(input_ids=input_ids_3, attention_mask=attention_mask_3, token_type_ids=token_type_ids_3)
            AI_humanlike_encoder = model(input_ids=input_ids_4, attention_mask=attention_mask_4, token_type_ids=token_type_ids_4)

            loss = twice_triple_loss(AI_text_encoder, AI_rewrite_encoder, AI_humanlike_encoder, Human_text_encoder)
            trainloss = loss.item()
            epoch_losses.append(trainloss)

            if len(recent_losses) < args.log_every:  # args.log_every=40
                recent_losses.append(trainloss)
            else:
                recent_losses[nitrs % len(recent_losses)] = trainloss


            # Backward the averaged gradient
            loss.backward()
            aveGrad += 1
            nitrs += 1
            nsamples += args.batch_size

            if aveGrad % args.naver_grad == 0:  # args.naver_grad =1
                optimizer.step()  # 这个方法会更新所有的参数
                optimizer.zero_grad()
                aveGrad = 0

            if nitrs % args.log_every == 0:  # log_every=40
                meanloss1 = sum(recent_losses) / len(recent_losses)
                print('epoch: %d step: %d trainloss: %.5f timecost:%.2f secs' %
                      (epoch, step, meanloss1, time.time() - start_time))
                writer.add_scalar('data/trainloss', meanloss1, nsamples)


        meanloss2 = sum(epoch_losses) / len(epoch_losses)
        print('epoch: %d meanloss: %.5f' % (epoch, meanloss2))
        writer.add_scalar('data/epochloss', meanloss2, nsamples)

        scheduler.step()

        ######################## eval model ###########################
        print("######## val data ########")

        sum_valloss = 0.0
        count = 0
        model.eval()

        for step, sample_batched in enumerate(val_loader):
            sample_batched = {key: value.cuda() for key, value in sample_batched.items()}
            input_ids_1 = sample_batched['input_ids_1']
            attention_mask_1 = sample_batched['attention_mask_1']
            token_type_ids_1 = sample_batched['token_type_ids_1']
            input_ids_2 = sample_batched['input_ids_2']
            attention_mask_2 = sample_batched['attention_mask_2']
            token_type_ids_2 = sample_batched['token_type_ids_2']
            input_ids_3 = sample_batched['input_ids_3']
            attention_mask_3 = sample_batched['attention_mask_3']
            token_type_ids_3 = sample_batched['token_type_ids_3']
            input_ids_4 = sample_batched['input_ids_4']
            attention_mask_4 = sample_batched['attention_mask_4']
            token_type_ids_4 = sample_batched['token_type_ids_4']

            with torch.no_grad():
                Human_text_encoder = model(input_ids=input_ids_1, attention_mask=attention_mask_1, token_type_ids=token_type_ids_1)
                AI_text_encoder = model(input_ids=input_ids_2, attention_mask=attention_mask_2, token_type_ids=token_type_ids_2)
                AI_rewrite_encoder = model(input_ids=input_ids_3, attention_mask=attention_mask_3, token_type_ids=token_type_ids_3)
                AI_humanlike_encoder = model(input_ids=input_ids_4, attention_mask=attention_mask_4, token_type_ids=token_type_ids_4)

            loss = twice_triple_loss(AI_text_encoder, AI_rewrite_encoder, AI_humanlike_encoder, Human_text_encoder)
            sum_valloss += loss.item()
            count += 1

            if step % num_iter_ts == num_iter_ts - 1:
                mean_valLoss = sum_valloss / num_iter_ts

                print('Validation:')
                print('epoch: %d, numImages: %d valLoss: %.5f' % (epoch, count, mean_valLoss))
                writer.add_scalar('data/valloss', mean_valLoss, count)

                ################  Save Pth  ################
                cur_f = mean_valLoss
                if cur_f < best_f:
                    save_path = os.path.join(save_dir, 'contra_albert_model' + '.pth')
                    torch.save(model.state_dict(), save_path)
                    print("Save model at {}\n".format(save_path))
                    best_f = cur_f


if __name__ == '__main__':
    args = get_argument()
    main(args)








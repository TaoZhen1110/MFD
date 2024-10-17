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
from transformers import RobertaTokenizer
import json
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from model import Multi_Level_Framework
from dataloader import Data_Loader, dataset_loader


def get_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='2,3,4,5')
    parser.add_argument('-train_batch_size', type=int, default=512)
    parser.add_argument('-val_batch_size', type=int, default=64)
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-resume_epoch', type=int, default=0)
    parser.add_argument('-log_every', type=int, default=200)
    parser.add_argument('-naver_grad', type=int, default=1)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-weight_decay', type=float, default=1e-4)
    parser.add_argument('-Roberta_Path', type=str,
                        default="/mnt/data132/taozhen/AI_Thesis_Detection/Framework/roberta_pretrained/")
    parser.add_argument('-Thesis_Roberta_Encoder_Path', type=str,
                        default="/mnt/data132/taozhen/AI_Thesis_Detection/Framework/Contrastive_Model/contras_model_save/run_0/contras_Roberta_model.pth")
    parser.add_argument('-train_dataset_path', type=str,
                        default="/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/PASTED/Llama3.1/train.json")
    parser.add_argument('-val_dataset_path', type=str,
                        default="/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/PASTED/Llama3.1/val.json")

    return parser.parse_args()


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):

    #################  Setting Gpu and seed  #################
    setup_seed(1234)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    #################  Train model save setting  #################
    save_dir_root = os.path.dirname(os.path.abspath(__file__))
    # os.path.abspath:取当前文件的绝对路径（完整路径）
    # os.path.dirname:去掉文件名，返回目录

    if args.resume_epoch != 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'class1_save', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'class1_save', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    save_dir = os.path.join(save_dir_root, 'class1_save', 'run_' + str(run_id))
    log_dir = os.path.join(save_dir, datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)  # 将数据以特定的格式存储到上面得到的那个日志文件夹中


    ###################  Dataset prepare #################
    tokenizer = RobertaTokenizer.from_pretrained(args.Roberta_Path)

    train_data = []
    with open(args.train_dataset_path) as f:
        for line in f:
            train_data.append(json.loads(line))  # 逐行加载 JSON 对象
    train_loader = dataset_loader(dataset=Data_Loader(jsondata=train_data, tokenizer=tokenizer),
                                  batch_size=args.train_batch_size, shuffle=True)


    val_data = []
    with open(args.val_dataset_path) as f:
        for line in f:
            val_data.append(json.loads(line))
    val_loader = dataset_loader(dataset=Data_Loader(jsondata=val_data, tokenizer=tokenizer),
                                batch_size=args.val_batch_size, shuffle=False)

    num_iter_tr = len(train_loader)
    # num_iter_va = len(val_loader)
    nitrs = args.resume_epoch * num_iter_tr
    nsamples = 0

    ########################  Model ###########################
    model = Multi_Level_Framework(args.Roberta_Path, args.Thesis_Roberta_Encoder_Path)
    model.apply(init_weights)
    model = nn.DataParallel(model).cuda()

    ##################  优化 Setting #################
    parameters_update = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(parameters_update, lr=args.lr)
    optimizer = torch.optim.AdamW(parameters_update, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    #################  Loss function  #################
    loss_function = nn.MSELoss()

    #################  Train Model  ##################
    recent_losses = []
    aveGrad = 0
    best_f = 100000.0
    start_time = time.time()


    for epoch in tqdm(range(args.resume_epoch, args.epochs)):
        model.train()
        count1 = 0
        epoch_train_losses = []

        for step, sample_batched in enumerate(train_loader):
            sample_batched = {key: value.cuda() for key, value in sample_batched.items()}

            Third_level_feature = sample_batched["Third_level_feature"]

            Label = sample_batched['Label']


            output = model(Third_level_feature)

            loss1 = loss_function(output, Label)
            final_loss = loss1
            count1 += Label.size(0)

            trainloss = final_loss.item()
            epoch_train_losses.append(trainloss)

            if len(recent_losses) < args.log_every:
                recent_losses.append(trainloss)
            else:
                recent_losses[nitrs % len(recent_losses)] = trainloss

            # Backward the averaged gradient
            final_loss.backward()
            aveGrad += 1
            nitrs += 1
            nsamples += args.train_batch_size

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % args.naver_grad == 0:  # args.naver_grad =1
                optimizer.step()  # 这个方法会更新所有的参数
                optimizer.zero_grad()
                aveGrad = 0

            if nitrs % args.log_every == 0:  # log_every=40
                meanloss1 = sum(recent_losses) / len(recent_losses)
                print('epoch: %d step: %d count: %d trainloss: %.5f timecost:%.2f secs' %
                      (epoch, step, count1, meanloss1, time.time() - start_time))
                writer.add_scalar('data/trainloss', meanloss1, nsamples)

        meanloss2 = sum(epoch_train_losses) / len(epoch_train_losses)
        print('epoch: %d meanloss: %.5f' % (epoch, meanloss2))
        writer.add_scalar('data/epochloss', meanloss2, nsamples)

        scheduler.step()


        ######################## eval model ###########################
        print("######## val data ########")
        epoch_val_losses = []
        count2 = 0
        all_val_outputs = []
        all_val_labels = []

        model.eval()

        for step, sample_batched in enumerate(val_loader):
            sample_batched = {key: value.cuda() for key, value in sample_batched.items()}

            Third_level_feature = sample_batched["Third_level_feature"]

            Label = sample_batched['Label']

            with torch.no_grad():
                output = model(Third_level_feature)

            loss1 = loss_function(output, Label)
            final_loss = loss1

            valloss = final_loss.item()
            epoch_val_losses.append(valloss)


            all_val_outputs.append(output.cpu().numpy())
            all_val_labels.append(Label.cpu().numpy())
            count2 += Label.size(0)


        ########## 计算损失函数 ##########
        mean_valLoss = sum(epoch_val_losses) / len(epoch_val_losses)

        ########## 计算 mse 三个指标 ##########
        numpy_all_val_outputs = np.concatenate(all_val_outputs, axis=0)
        numpy_all_val_labels = np.concatenate(all_val_labels, axis=0)

        Lexical_outputs = numpy_all_val_outputs[:, 0]
        Lexical_labels = numpy_all_val_labels[:, 0]

        Syntax_outputs = numpy_all_val_outputs[:, 1]
        Syntax_labels = numpy_all_val_labels[:, 1]

        Semantic_outputs = numpy_all_val_outputs[:, 2]
        Semantic_labels = numpy_all_val_labels[:, 2]

        Lexical_mse = mean_squared_error(Lexical_labels, Lexical_outputs)
        Syntax_mse = mean_squared_error(Syntax_labels, Syntax_outputs)
        Semantic_mse = mean_squared_error(Semantic_labels, Semantic_outputs)

        Mean_mse = (Lexical_mse + Syntax_mse + Semantic_mse) / 3


        print('Validation:')
        print('epoch: %d, numtext: %d valLoss: %.5f MeanMSE: %.5f' % (
            epoch, count2, mean_valLoss, Mean_mse))
        writer.add_scalar('data/valloss', mean_valLoss, count2)
        writer.add_scalar('data/MeanMSE', Mean_mse, count2)


        ################   Save Pth  ################

        if Mean_mse < best_f:  # 如果当前MSE小于之前的最佳值
            save_path = os.path.join(save_dir, 'class1_model' + '.pth')
            torch.save(model.state_dict(), save_path)
            print("Save model at {}\n".format(save_path))
            best_f = Mean_mse  # 更新最佳的MSE值



if __name__ == "__main__":
    args = get_argument()
    main(args)
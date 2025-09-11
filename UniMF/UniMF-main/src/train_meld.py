import torch
import torch.nn.functional as F
from torch import nn
import sys
import csv
from src import models
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import eval_meld


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    """初始化模型、优化器和训练设置，集成GAN组件"""
    # 初始化GAN组件
    generators = {}
    discriminators = {}
    gen_optimizers = {}
    dis_optimizers = {}

    if hyp_params.modalities != 'LA':
        if hyp_params.modalities == 'L':
            # 仅文本模态，生成音频模态
            generators['A'] = models.Generator(hyp_params, missing='A')
            discriminators['A'] = models.Discriminator(input_dim=hyp_params.orig_d_a)

            gen_optimizers['A'] = optim.Adam(
                generators['A'].parameters(),
                lr=hyp_params.gen_lr
            )
            dis_optimizers['A'] = optim.Adam(
                discriminators['A'].parameters(),
                lr=hyp_params.dis_lr
            )

        elif hyp_params.modalities == 'A':
            # 仅音频模态，生成文本模态
            generators['L'] = models.Generator(hyp_params, missing='L')
            discriminators['L'] = models.Discriminator(input_dim=hyp_params.orig_d_l)

            gen_optimizers['L'] = optim.Adam(
                generators['L'].parameters(),
                lr=hyp_params.gen_lr
            )
            dis_optimizers['L'] = optim.Adam(
                discriminators['L'].parameters(),
                lr=hyp_params.dis_lr
            )
        else:
            raise ValueError('Unknown modalities type for MELD dataset')

        # 移动到GPU
        if hyp_params.use_cuda:
            for g in generators.values():
                g = g.cuda()
            for d in discriminators.values():
                d = d.cuda()

    # 初始化主模型（多模态理解模型）
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()

    # 主模型优化器
    if hyp_params.use_bert:
        # BERT参数特殊处理（区分衰减和不衰减参数）
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.text_model.named_parameters())
        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': hyp_params.weight_decay_bert, 'lr': hyp_params.lr_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': hyp_params.lr_bert},
            {'params': model_params_other, 'weight_decay': 0.0, 'lr': hyp_params.lr}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
    else:
        optimizer = getattr(optim, hyp_params.optim)(
            model.parameters(),
            lr=hyp_params.lr
        )

    criterion = getattr(nn, hyp_params.criterion)()  # 主任务损失函数
    gan_criterion = nn.BCELoss()  # GAN二分类损失函数

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=hyp_params.when,
        factor=0.1
    )

    # 整理训练设置
    settings = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'scheduler': scheduler,
        'generators': generators,
        'discriminators': discriminators,
        'gen_optimizers': gen_optimizers,
        'dis_optimizers': dis_optimizers,
        'gan_criterion': gan_criterion
    }

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    """训练主函数，包含GAN对抗训练循环和评估过程"""
    # 提取训练组件
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    gan_criterion = settings['gan_criterion']

    generators = settings['generators']
    discriminators = settings['discriminators']
    gen_optimizers = settings['gen_optimizers']
    dis_optimizers = settings['dis_optimizers']

    def train(model, generators, discriminators, optimizer, gen_optimizers, dis_optimizers, criterion, gan_criterion):
        """训练一个epoch的函数，包含GAN对抗训练逻辑"""
        model.train()
        for g in generators.values():
            g.train()
        for d in discriminators.values():
            d.train()

        epoch_loss = 0
        start_time = time.time()

        for i_batch, (audio, text, masks, labels) in enumerate(train_loader):
            # 梯度清零
            model.zero_grad()
            for opt in gen_optimizers.values():
                opt.zero_grad()
            for opt in dis_optimizers.values():
                opt.zero_grad()

            # 数据移至GPU并应用掩码
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, masks, labels = text.cuda(), audio.cuda(), masks.cuda(), labels.cuda()

            # 应用掩码（处理变长序列）
            masks_text = masks.unsqueeze(-1).expand(-1, 33, 600)
            if hyp_params.dataset == 'meld_senti':
                masks_audio = masks.unsqueeze(-1).expand(-1, 33, 600)  # 情感分析数据集
            else:
                masks_audio = masks.unsqueeze(-1).expand(-1, 33, 300)  # 情感分类数据集
            text = text * masks_text
            audio = audio * masks_audio
            batch_size = text.size(0)

            # 1. 生成缺失模态并计算GAN损失
            fake_modal = None
            gen_loss = 0
            dis_loss = 0

            if hyp_params.modalities == 'L':
                # 文本生成音频
                gen = generators['A']
                dis = discriminators['A']

                # 生成假音频
                fake_audio = gen(text, audio, phase='train')

                # 训练判别器
                real_pred = dis(audio)
                fake_pred = dis(fake_audio.detach())  # 不更新生成器梯度
                dis_loss = (gan_criterion(real_pred, torch.ones_like(real_pred)) +
                            gan_criterion(fake_pred, torch.zeros_like(fake_pred))) / 2
                dis_loss.backward(retain_graph=True)
                dis_optimizers['A'].step()

                # 训练生成器（对抗损失）
                fake_pred = dis(fake_audio)
                gen_gan_loss = gan_criterion(fake_pred, torch.ones_like(fake_pred))
                fake_modal = fake_audio

            elif hyp_params.modalities == 'A':
                # 音频生成文本
                gen = generators['L']
                dis = discriminators['L']

                # 生成假文本
                fake_text = gen(audio, text, phase='train')

                # 训练判别器
                real_pred = dis(text)
                fake_pred = dis(fake_text.detach())  # 不更新生成器梯度
                dis_loss = (gan_criterion(real_pred, torch.ones_like(real_pred)) +
                            gan_criterion(fake_pred, torch.zeros_like(fake_pred))) / 2
                dis_loss.backward(retain_graph=True)
                dis_optimizers['L'].step()

                # 训练生成器（对抗损失）
                fake_pred = dis(fake_text)
                gen_gan_loss = gan_criterion(fake_pred, torch.ones_like(fake_pred))
                fake_modal = fake_text

            # 2. 主模型前向传播
            net = nn.DataParallel(model) if hyp_params.distribute else model
            if hyp_params.modalities == 'L':
                preds, _ = net(text, fake_modal)
            elif hyp_params.modalities == 'A':
                preds, _ = net(fake_modal, audio)
            else:  # 'LA'
                preds, _ = net(text, audio)

            # 3. 计算总损失并反向传播
            task_loss = criterion(preds.transpose(1, 2), labels)

            if hyp_params.modalities != 'LA':
                # 联合损失 = 主任务损失 + GAN损失（带权重）
                total_loss = task_loss + hyp_params.gan_weight * gen_gan_loss
            else:
                total_loss = task_loss

            total_loss.backward()

            # 参数更新
            if hyp_params.modalities != 'LA':
                torch.nn.utils.clip_grad_norm_(generators[list(generators.keys())[0]].parameters(), hyp_params.clip)
                gen_optimizers[list(gen_optimizers.keys())[0]].step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            epoch_loss += total_loss.item() * batch_size

        # 返回平均损失
        return epoch_loss / hyp_params.n_train

    def evaluate(model, generators, criterion, test=False):
        """评估函数（验证或测试）"""
        model.eval()
        for g in generators.values():
            g.eval()

        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []
        mask = []

        with torch.no_grad():
            for i_batch, (audio, text, masks, labels) in enumerate(loader):
                # 数据处理
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, masks, labels = text.cuda(), audio.cuda(), masks.cuda(), labels.cuda()

                # 应用掩码
                masks_text = masks.unsqueeze(-1).expand(-1, 33, 600)
                if hyp_params.dataset == 'meld_senti':
                    masks_audio = masks.unsqueeze(-1).expand(-1, 33, 600)
                else:
                    masks_audio = masks.unsqueeze(-1).expand(-1, 33, 300)
                text = text * masks_text
                audio = audio * masks_audio
                batch_size = text.size(0)

                # 生成缺失模态
                fake_modal = None
                if hyp_params.modalities != 'LA':
                    gen = generators[list(generators.keys())[0]]
                    if not test:
                        # 验证阶段直接生成
                        if hyp_params.modalities == 'L':
                            fake_modal = gen(text, audio, phase='valid')
                        elif hyp_params.modalities == 'A':
                            fake_modal = gen(audio, text, phase='valid')
                    else:
                        # 测试阶段自回归生成
                        if hyp_params.modalities == 'L':
                            fake_modal = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                            for i in range(hyp_params.a_len):
                                if i == 0:
                                    token = gen(text, audio, phase='test', eval_start=True)[:, [-1]]
                                else:
                                    token = gen(text, fake_modal, phase='test')[:, [-1]]
                                fake_modal = torch.cat((fake_modal, token), dim=1)
                        elif hyp_params.modalities == 'A':
                            fake_modal = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                            for i in range(hyp_params.l_len):
                                if i == 0:
                                    token = gen(audio, text, phase='test', eval_start=True)[:, [-1]]
                                else:
                                    token = gen(audio, fake_modal, phase='test')[:, [-1]]
                                fake_modal = torch.cat((fake_modal, token), dim=1)

                # 主模型预测
                net = nn.DataParallel(model) if hyp_params.distribute else model
                if hyp_params.modalities == 'L':
                    preds, _ = net(text, fake_modal)
                elif hyp_params.modalities == 'A':
                    preds, _ = net(fake_modal, audio)
                else:  # 'LA'
                    preds, _ = net(text, audio)

                # 计算损失
                raw_loss = criterion(preds.transpose(1, 2), labels)
                total_loss += raw_loss.item() * batch_size

                # 收集结果
                results.append(preds)
                truths.append(labels)
                mask.append(masks)

        # 计算平均损失并整理结果
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        mask = torch.cat(mask)
        return avg_loss, results, truths, mask

    # 打印模型参数信息
    if hyp_params.modalities != 'LA':
        gen_params = sum([sum(p.numel() for p in g.parameters()) for g in generators.values()])
        dis_params = sum([sum(p.numel() for p in d.parameters()) for d in discriminators.values()])
        print(f'Trainable Parameters for Generators: {gen_params}')
        print(f'Trainable Parameters for Discriminators: {dis_params}')

    model_params = sum([param.nelement() for param in model.parameters()])
    print(f'Trainable Parameters for Main Model: {model_params}')

    # 训练循环
    best_valid = float('inf')
    loop = tqdm(range(1, hyp_params.num_epochs + 1), leave=False)

    for epoch in loop:
        loop.set_description(f'Epoch {epoch:2d}/{hyp_params.num_epochs}')

        # 训练
        train_loss = train(model, generators, discriminators, optimizer, gen_optimizers, dis_optimizers, criterion,
                           gan_criterion)

        # 验证
        val_loss, _, _, _ = evaluate(model, generators, criterion, test=False)

        # 学习率调整
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_valid:
            # 保存生成器
            for name, g in generators.items():
                save_model(hyp_params, g, name=f'GENERATOR_{name}')
            # 保存主模型
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    # 加载最佳模型进行测试
    for name in generators:
        generators[name] = load_model(hyp_params, name=f'GENERATOR_{name}')
    model = load_model(hyp_params, name=hyp_params.name)

    # 测试
    _, results, truths, mask = evaluate(model, generators, criterion, test=True)

    # 评估并返回结果
    acc = eval_meld(results, truths, mask)
    return acc

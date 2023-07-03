import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler #Change
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds


from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/50%scribble_semi_weakly/Rotation_CPS_MT', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
# parser.add_argument('--model', type=str,
#                     default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, #changed 12 to 24
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
##TO BE APPLIED TO OTHER METHODS
##TO BE APPLIED TO OTHER METHODS
parser.add_argument('--patch_size', type=list,  default=[224, 224], 
                    help='patch size of network input')
#SAME SEED [2022]
#Specify equity
# #At Implementation Details Section 
parser.add_argument('--seed', type=int,  default=2022, help='random seed')



parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
##########Change
# label and unlabel and alpha
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--ict_alpha', type=int, default=0.2,
                    help='ict_alpha')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
##########Change end

args = parser.parse_args()

config = get_config(args)
#Change begins
# def patients_to_slices(dataset, patiens_num):
#     ref_dict = None
#     if "ACDC" in dataset:
#         ref_dict = {'1':14,'2':28,"3": 68, "7": 136,
#                     "14": 256, "21": 396, "28": 512, "35": 664,"60": 786,"70": 917,"80": 1048,"90": 1179, "140": 1300}
#     else:
#         print("Error")
#     return ref_dict[str(patiens_num)]
#Change ends
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    #return 1.0 * ramps.sigmoid_rampup(epoch, 60)
    #Change
    #linear_rampup, cosine_rampdown

    # linear_rampup, cosine_rampdown
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)



def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args, snapshot_path):
    #Assign Training Parameters Values
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

 

    #1Create Network
    model1 = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    model2 = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    ema_model1 = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    
    # load pretrain weights
    # model1_weights = torch.load('/home/yliu/CV/CV-WSL-MIS/Ours/ACDC/50%scribble_semi_weakly/All_test10_fold1/model1_iter_9000.pth', map_location=device)
    # model1.load_state_dict(model1_weights)

    # model2_weights = torch.load('/home/yliu/CV/CV-WSL-MIS/Ours/ACDC/50%scribble_semi_weakly/All_test10_fold1/model2_iter_9000.pth', map_location=device)
    # model2.load_state_dict(model2_weights)

    # ema_weights = torch.load('/home/yliu/CV/CV-WSL-MIS/Ours/ACDC/50%scribble_semi_weakly/All_test10_fold1/ema_iter_9000.pth', map_location=device)
    # ema_model1.load_state_dict(ema_weights)

    #Load pretrained model weights
    # #TODO load different weights
    model1.load_from(config)
    model2.load_from(config)
    ema_model1.load_from(config)

    for param in ema_model1.parameters():
        param.detach_()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # Change begins
    # db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
    #     RandomGenerator(args.patch_size)
    # ]), fold=args.fold, sup_type=args.sup_type)
    # db_val = BaseDataSets(base_dir=args.root_path,
    #                       fold=args.fold, split="val")
    #Change ends
    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([#"num=None," is added, but voided
        RandomGenerator(args.patch_size) # $th Consistency, random zoom
    ]), fold=args.fold, sup_type=args.sup_type) #", fold=args.fold, sup_type=args.sup_type" is added
    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val") #"fold=args.fold, "" is added
    total_slices = len(db_train)
    labeled_slice = 756 #patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    print("unlabeled_idxs is:  {}, {}".format(batch_size, batch_size-args.labeled_bs))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)


    #Look into Training Loader
    #Modify to Semi+Weak Labels
    #Change
    trainloader = DataLoader(db_train,  batch_sampler=batch_sampler, #"batch_sampler=batch_sampler" is added
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn) #num_workers=8->4, batch_size=batch_size, shuffle=True, is removed
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model1.train()
    model2.train()

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4) # To be Searched
    #Change
    # dice_loss = losses.DiceLoss(num_classes)

    

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    start_epoch = iter_num // len(trainloader)
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            #Change - Jun 3

            outputs1 = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            rot_times = random.randrange(0, 4)
            rotated_volume_batch = torch.rot90(volume_batch, rot_times, [2, 3])

            # unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            # labeled_volume_batch = volume_batch[:args.labeled_bs]


            # """ICT Start"""
            # # ICT mix factors
            # ict_mix_factors = np.random.beta(
            #     args.ict_alpha, args.ict_alpha, size=(args.labeled_bs//2, 1, 1, 1))
            # ict_mix_factors = torch.tensor(
            #     ict_mix_factors, dtype=torch.float).cuda()
            # unlabeled_volume_batch_0 = unlabeled_volume_batch[0:args.labeled_bs//2, ...]
            # unlabeled_volume_batch_1 = unlabeled_volume_batch[args.labeled_bs//2:, ...]

            # #Maybe also mix scribble labeled data

            # Mix images
            # batch_ux_mixed = unlabeled_volume_batch_0 * \
            #     (1.0 - ict_mix_factors) + \
            #     unlabeled_volume_batch_1 * ict_mix_factors
            # input_volume_batch = torch.cat(
            #     [labeled_volume_batch, batch_ux_mixed], dim=0)
            # outputs_of_mix1 = model1(input_volume_batch) #ema_model1(input_volume_batch)
            # outputs_soft_of_mix1 = torch.softmax(outputs_of_mix1, dim=1)
            # pred_of_mix = outputs_soft_of_mix1[args.labeled_bs:]
            # with torch.no_grad():
            #     ema1_output_ux0 = torch.softmax(
            #         ema_model1(unlabeled_volume_batch_0), dim=1)
            #     ema1_output_ux1 = torch.softmax(
            #         ema_model1(unlabeled_volume_batch_1), dim=1)
            #     mixed_batch_pred1 = ema1_output_ux0 * \
            #         (1.0 - ict_mix_factors) + ema1_output_ux1 * ict_mix_factors

            #     mixed_batch_pred1 = ema1_output_ux0 * \
            #         (1.0 - ict_mix_factors) + ema1_output_ux1 * ict_mix_factors
            # consistency_weight_ICT = get_current_consistency_weight(
            #     iter_num // 1000)
            # # # Mixup loss - MSE
            # Mixup_loss1 =  1 * consistency_weight_ICT * torch.mean(
            #     (pred_of_mix - mixed_batch_pred1) ** 2)
            # Mixup_loss1 = torch.mean(
            #     (pred_of_mix - mixed_batch_pred1) ** 2)


            # """ICT End"""

            # """Mean Teacher Start"""
            # noise = torch.clamp(torch.randn_like(
            #     unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            
            # ema_inputs1 = unlabeled_volume_batch + noise

            # outputs1 = model1(volume_batch)
            # outputs_soft1 = torch.softmax(outputs1, dim=1)
            # outputs2 = model2(volume_batch)
            # outputs_soft2 = torch.softmax(outputs2, dim=1)

            # with torch.no_grad():
            #     ema_output1 = ema_model1(ema_inputs1)
            #     # SSL CASE MEAN TEACHER
            #     # ema_output_soft1 = torch.softmax(ema_output1, dim=1)
            #     # MeanT_loss1 = torch.mean(
            #     #     (outputs_soft1[args.labeled_bs:]-ema_output_soft1)**2)
            # T = 8
            # _, _, w, h = unlabeled_volume_batch.shape
            # volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
            # stride = volume_batch_r.shape[0] // 2
            # preds = torch.zeros([stride * T, num_classes, w, h]).cuda()
            # for i in range(T//2):
            #     ema_inputs = volume_batch_r + \
            #         torch.clamp(torch.randn_like(
            #             volume_batch_r) * 0.1, -0.2, 0.2)
            #     with torch.no_grad():
            #         preds[2 * stride * i:2 * stride *
            #               (i + 1)] = ema_model1(ema_inputs)
            # preds = F.softmax(preds, dim=1)
            # preds = preds.reshape(T, stride, num_classes, w, h)
            # preds = torch.mean(preds, dim=0)
            # uncertainty = -1.0 * \
            #     torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)
            # MT_consistency_dist1 = losses.softmax_mse_loss(
            #     outputs1[args.labeled_bs:], ema_output1)  # (batch, 2, 112,112,80)
            # threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num,
            #                                             max_iterations))*np.log(2)
            # mask = (uncertainty < threshold).float()
            # MeanT_loss1 = torch.sum(
            #     mask*MT_consistency_dist1)/(2*torch.sum(mask)+1e-16)

            noise = torch.clamp(torch.randn_like(
                rotated_volume_batch) * 0.1, -0.2, 0.2)

            with torch.no_grad():
                ema_inputs1_r = rotated_volume_batch + noise
                ema_output1_r = ema_model1(ema_inputs1_r)
            T = 8
            _, _, w, h = volume_batch.shape
            volume_batch_r = rotated_volume_batch.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, num_classes, w, h]).cuda()
            for i in range(T//2):
                ema_inputs = volume_batch_r + \
                    torch.clamp(torch.randn_like(
                        volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride *
                          (i + 1)] = ema_model1(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, num_classes, w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * \
                torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)
            
            rotated_ouputs = torch.rot90(outputs1, rot_times, [2, 3])
            MT_consistency_dist1 = losses.softmax_mse_loss(
                rotated_ouputs, ema_output1_r)  # (batch, 2, 112,112,80)
            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num,
                                                        max_iterations))*np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_weight_MT = get_current_consistency_weight(
                iter_num // 1000)
            MeanT_loss1 = 1 * consistency_weight_MT * torch.sum(
                mask * MT_consistency_dist1) / (2 * torch.sum(mask) + 1e-16)
            # """Mean Teacher End"""


            """CPS Start"""

            # outputs_soft1_r = torch.rot90(outputs_soft1.detach(), rot_times, [2, 3])
            # outputs_soft2_r = torch.rot90(outputs_soft2.detach(), rot_times, [2, 3])

            pseudo_outputs1 = torch.argmax(outputs_soft1.detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2.detach(), dim=1, keepdim=False)

            pseudo_supervision1 = ce_loss(outputs1, pseudo_outputs2)
            pseudo_supervision2 = ce_loss(outputs2, pseudo_outputs1)
            consistency_weight_CPS = get_current_consistency_weight(
                iter_num // 1000)
            loss_CPS =  consistency_weight_CPS * (pseudo_supervision1 + pseudo_supervision2)

            # """CPS End"""

            #Supervised Loss Start ONLY THIS PART HAS TO BE SEMI SUPERVISED
            #  ONLY THIS PART HAS TO BE SEMI SUPERVISED
            #  ONLY THIS PART HAS TO BE SEMI SUPERVISED
            #  ONLY THIS PART HAS TO BE SEMI SUPERVISED
            ##Model1 supervised loss
            loss_ce1 = ce_loss(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())
            # loss_dice1 = dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss1 = 1 * loss_ce1 #+ 1.2 * loss_dice1

                ##Model2 supervised loss
            loss_ce2 = ce_loss(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())
            # loss_dice2 = dice_loss(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss2 = 1 * loss_ce2 #+ 1.2 * loss_dice2
                #Overall SUperivsed Loss
            supervised_loss = 0.5 * (loss1 + loss2)
            #Supervised Loss End
            ## ##MAKE USE OF SCRIBBLE LABELS IN CONSISTENCY LOSS
            #Consistency Loss:
                ##1, Consistency loss is Mean Teacher loss
                ##2, Mixup loss is ICT Mixup loss
                ##3, loss_CPS is CPS loss

            consistency_loss =  loss_CPS + MeanT_loss1
            if iter_num < 1000:
                consistency_loss = 0.0
            loss = supervised_loss + consistency_loss


            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            update_ema_variables(model1, ema_model1, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            for param_group1 in optimizer1.param_groups:
                param_group1['lr'] = lr_
            for param_group2 in optimizer2.param_groups:
                param_group2['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce1, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce2, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            # writer.add_scalar('info/consistency_weight',
            #                   consistency_weight, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_ce1: %f, loss_ce2: %f' %
                (iter_num, loss.item(), loss_ce1.item(), loss_ce2.item()))

            
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                _outputs1 = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                _outputs2 = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 _outputs1[1, ...] * 50, iter_num)
                writer.add_image('train/model2_Prediction',
                                 _outputs2[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 100 == 0:
                print("validation")
                model1.eval()
                metric_list1 = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes, patch_size=args.patch_size)
                    metric_list1 += np.array(metric_i)
                metric_list1 = metric_list1 / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list1[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list1[class_i, 1], iter_num)

                performance1 = np.mean(metric_list1, axis=0)[0]

                mean_hd951 = np.mean(metric_list1, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_best = os.path.join(snapshot_path,
                                                  'best_model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    torch.save(model1.state_dict(), save_best)
                    save_ema_path = os.path.join(snapshot_path,
                                            'best_ema.pth'.format(iter_num))
                    torch.save(ema_model1.state_dict(), save_ema_path)
                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_best = os.path.join(snapshot_path,
                                                  'best_model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))

                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path1 = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path1)
                logging.info("save model1 to {}".format(save_mode_path1))

                save_mode_path2 = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path2)
                logging.info("save model2 to {}".format(save_mode_path2))
                #Save the ema_model
                save_ema_path = os.path.join(snapshot_path,
                                            'ema_iter_{}.pth'.format(iter_num))
                torch.save(ema_model1.state_dict(), save_ema_path)
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../Ours/{}_{}/".format(
        args.exp, args.fold)#, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code_prostate'):
        shutil.rmtree(snapshot_path + '/code_prostate')
    shutil.copytree('.', snapshot_path + '/code_prostate',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

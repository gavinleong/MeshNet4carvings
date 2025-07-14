import os
import re
import time
import copy
import yaml
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
from config import get_train_config, get_test_config
from data import ModelNet40
from models import MeshNet
from utils import append_feature, calculate_map

# Experiment configuration
timesrun = list(range(1, 33))
subsets = [1, 2, 3, 4]
momentum = 0.9
max_epoch = 100
milestones = [30, 60]
batch_size_array = [40]
weight_decay_array = [0.0005, 0.001, 0.002, 0.003]
lr_array = [0.0005, 0.001, 0.002, 0.003]

home_dir = 'XXXXXXXXXX'

# AdamW optimiser used instead of stochastic gradient descent 
# train and testing combined into one script to run grid search over hyperparameters

# Main loop for repeated experiments
for timesrun_number in timesrun:
    for subset_number in subsets:
        saved_nets = f'adam_optimiser_results{timesrun_number}_{subset_number}_pred_crossen/'
        data_root_new = f'XXXXXXXXXX{subset_number}/'
        os.makedirs(saved_nets, exist_ok=True)

        # Output file for summary scores
        score_output = open(f"{saved_nets}score_output", "w")

        for lr in lr_array:
            for weight_decay in weight_decay_array:
                for batch_size in batch_size_array:
                    t_start = time.time()

                    ckpt_root = f"{saved_nets}pkl_lr{lr}wd{weight_decay}b{batch_size}_adam_final/"
                    os.makedirs(ckpt_root, exist_ok=True)
                    success = open(f"{ckpt_root}lr{lr}wd{weight_decay}", "w")

                    print(f"lr = {lr} \nweight_decay = {weight_decay}")

                    os.environ['CUDA_VISIBLE_DEVICES'] = get_train_config(lr, weight_decay)['cuda_devices']
                    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

                    # Setup data loaders
                    cfg = get_train_config(lr, weight_decay)
                    cfg["dataset"]["data_root"] = data_root_new
                    dataset = {x: ModelNet40(cfg=cfg['dataset'], part=x) for x in ['train', 'test']}
                    loader = {x: DataLoader(dataset[x], batch_size=batch_size, num_workers=4, shuffle=True) for x in ['train', 'test']}

                    cfg2 = get_test_config()
                    cfg2["dataset"]["data_root"] = data_root_new
                    test_dataset = ModelNet40(cfg=cfg2['dataset'], part='test')
                    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=True)

                    # Load or train model
                    load_model_path = os.path.join(home_dir, saved_nets, f'pkl_lr{lr}wd{weight_decay}b{batch_size}_adam_final/MeshNet_best.pkl')

                    if __name__ == '__main__':
                        # Initialize model and training tools
                        model = MeshNet(cfg=cfg['MeshNet'], require_fea=True).cuda()
                        model = nn.DataParallel(model)
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

                        # Train model
                        def train_model(model, criterion, optimizer, scheduler, cfg):
                            best_acc, best_map = 0.0, 0.0
                            best_wts = copy.deepcopy(model.state_dict())
                            best_opt = copy.deepcopy(optimizer.state_dict())
                            train_loss, test_loss, train_acc, test_acc = [], [], [], []

                            for epoch in range(1, max_epoch + 1):
                                print(f"{'-'*60}\nEpoch: {epoch} / {max_epoch}\n{'-'*60}")
                                for phase in ['train', 'test']:
                                    model.train() if phase == 'train' else model.eval()
                                    if phase == 'train': scheduler.step()

                                    running_loss, correct = 0.0, 0
                                    ft_all, lbl_all = None, None

                                    for centers, corners, normals, neighbors, targets, _ in loader[phase]:
                                        centers, corners, normals = [Variable(x.cuda()) for x in (centers, corners, normals)]
                                        neighbors, targets = [Variable(x.cuda()) for x in (neighbors, targets)]
                                        
                                        with torch.set_grad_enabled(phase == 'train'):
                                            outputs, feats = model(centers, corners, normals, neighbors)
                                            loss = criterion(outputs, targets)
                                            if phase == 'train':
                                                optimizer.zero_grad()
                                                loss.backward()
                                                optimizer.step()

                                            if phase == 'test':
                                                ft_all = append_feature(ft_all, feats.detach())
                                                lbl_all = append_feature(lbl_all, targets.detach(), flaten=True)

                                            _, preds = torch.max(outputs, 1)
                                            running_loss += loss.item() * centers.size(0)
                                            correct += torch.sum(preds == targets)

                                    epoch_loss = running_loss / len(dataset[phase])
                                    epoch_accuracy = correct.double() / len(dataset[phase])

                                    if phase == 'train':
                                        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}")
                                        train_loss.append(epoch_loss)
                                        train_acc.append(epoch_accuracy)
                                    else:
                                        epoch_map = calculate_map(ft_all, lbl_all)
                                        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f} mAP: {epoch_map:.4f}")
                                        test_loss.append(epoch_loss)
                                        test_acc.append(epoch_accuracy)
                                        if epoch_accuracy > best_acc:
                                            best_acc, best_wts, best_opt = epoch_accuracy, copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
                                        best_map = max(best_map, epoch_map)

                            return best_wts, best_opt, train_loss, train_acc, test_loss, test_acc

                        best_wts, best_opt, train_loss, train_acc, test_loss, test_acc = train_model(model, criterion, optimizer, scheduler, cfg)

                        # Save best model
                        torch.save({'model_state_dict': best_wts, 'optimizer_state_dict': best_opt}, os.path.join(ckpt_root, 'MeshNet_best.pkl'))

                        # Load and evaluate model
                        model2 = MeshNet(cfg=cfg['MeshNet'], require_fea=True).cuda()
                        model2 = nn.DataParallel(model2)
                        checkpoint = torch.load(load_model_path)
                        model2.load_state_dict(checkpoint['model_state_dict'])
                        model2.eval()

                        # Test model
                        def test_model(model, ckpt_root):
                            correct, ft_all, lbl_all = 0, None, None
                            for centers, corners, normals, neighbors, targets, path in test_loader:
                                centers, corners, normals = [Variable(x.cuda()) for x in (centers, corners, normals)]
                                neighbors, targets = [Variable(x.cuda()) for x in (neighbors, targets)]

                                outputs, feats = model(centers, corners, normals, neighbors)
                                _, preds = torch.max(outputs, 1)
                                probs = nn.functional.softmax(outputs, dim=1)
                                max_prob = torch.max(probs)

                                result = "correct" if preds[0] == targets else "incorrect"
                                prob_str = np.array2string(((max_prob.cpu().detach().numpy() - 0.5) / 0.5) if result == "correct" else max_prob.cpu().detach().numpy())
                                success.write(f"{result}: {path[0]} , {prob_str}\n")

                                if result == "correct":
                                    correct += 1
                                ft_all = append_feature(ft_all, feats.detach())
                                lbl_all = append_feature(lbl_all, targets.detach(), flaten=True)

                            acc = correct / len(test_dataset)
                            print(f'Accuracy: {acc:.4f} mAP: {calculate_map(ft_all, lbl_all):.4f}')

                            # Parse log file for metrics
                            with open(os.path.join(ckpt_root, f"lr{lr}wd{weight_decay}")) as f:
                                flat, carve = 0, 0
                                for line in f:
                                    if 'incorrect' not in line:
                                        match = re.search(r'/test/1024_(.+?)\.npz', line)
                                        if match:
                                            if 'flat' in match.group(1): flat += 1
                                            else: carve += 1

                            TP, TN = carve, flat
                            FP = len(test_dataset)/2 - TN
                            FN = len(test_dataset)/2 - TP
                            precision = TP / (TP + FP) if TP else 0.0
                            recall = TP / (TP + FN) if TP else 0.0
                            f1 = 2 * precision * recall / (precision + recall) if TP else 0.0
                            print(f'Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}')
                            score_output.write(f"{acc:.4f} {precision:.4f} {recall:.4f} {f1:.4f}\n")

                        test_model(model2, ckpt_root)

                        # Plot results
                        plt.plot(train_loss, label='train loss', color='blue', linestyle='dashed')
                        plt.plot(test_loss, label='test loss', color='orange', linestyle='dashed')
                        plt.plot(torch.stack(train_acc).cpu().numpy(), label='train acc', color='blue')
                        plt.plot(torch.stack(test_acc).cpu().numpy(), label='test acc', color='orange')
                        plt.ylim(0, 4)
                        plt.legend()
                        plt.savefig(os.path.join(home_dir, saved_nets, f'pkl_lr{lr}wd{weight_decay}b{batch_size}_adam_final/train_test_graph.png'), dpi=1000)
                        plt.close()

                    print(f"Elapsed time: {time.time() - t_start:.2f} sec")

        score_output.close()

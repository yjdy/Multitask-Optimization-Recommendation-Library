import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import os
import numpy as np

from utils import get_dataset, get_model, EarlyStopper, set_seed

from torch.utils.tensorboard import SummaryWriter


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
        categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
            device), labels.to(device)
        y = model(categorical_fields, numerical_fields)
        loss_list = [criterion(y[i], labels[:, i].float()) for i in range(labels.size(1))]
        loss = 0
        for item in loss_list:
            loss += item
        loss /= len(loss_list)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def metatrain(model, optimizer, data_loader, device, log_interval=100):
    model.train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y = list(), list(), list(), list(), list(), list()
    for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
        categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
            device), labels.to(device)
        batch_size = int(categorical_fields.size(0) / 2)
        list_sup_categorical.append(categorical_fields[:batch_size])
        list_qry_categorical.append(categorical_fields[batch_size:])
        list_sup_numerical.append(numerical_fields[:batch_size])
        list_qry_numerical.append(numerical_fields[batch_size:])
        list_sup_y.append(labels[:batch_size])
        list_qry_y.append(labels[batch_size:])

        if (i + 1) % 2 == 0:
            loss = model.global_update(list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical,
                                       list_qry_numerical, list_qry_y)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y = list(), list(), list(), list(), list(), list()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, task_num, device):
    model.eval()
    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    for i in range(task_num):
        labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
    with torch.no_grad():
        for categorical_fields, numerical_fields, labels in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
                device), labels.to(device)
            y = model(categorical_fields, numerical_fields)
            for i in range(task_num):
                labels_dict[i].extend(labels[:, i].tolist())
                predicts_dict[i].extend(y[i].tolist())
                loss_dict[i].extend(
                    torch.nn.functional.binary_cross_entropy(y[i], labels[:, i].float(), reduction='none').tolist())
    auc_results, loss_results = list(), list()
    for i in range(task_num):
        auc_results.append(roc_auc_score(labels_dict[i], predicts_dict[i]))
        loss_results.append(np.array(loss_dict[i]).mean())
    return auc_results, loss_results


def main(dataset_name,
         dataset_path,
         task_num,
         expert_num,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         embed_dim,
         weight_decay,
         device,
         save_dir):
    device = torch.device(device)
    train_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/train.csv')
    test_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/test.csv')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    field_dims = train_dataset.field_dims
    numerical_num = train_dataset.numerical_num
    model = get_model(model_name, field_dims, numerical_num, task_num, expert_num, embed_dim).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    save_path = f'{save_dir}/{dataset_name}_{model_name}.pt'
    early_stopper = EarlyStopper(num_trials=2, save_path=save_path)
    for epoch_i in range(epoch):
        if model_name == 'metaheac':
            metatrain(model, optimizer, train_data_loader, device)
        else:
            train(model, optimizer, train_data_loader, criterion, device)
        auc, loss = test(model, test_data_loader, task_num, device)
        print('epoch:', epoch_i, 'test: auc:', auc)
        for i in range(task_num):
            print('task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
        if not early_stopper.is_continuable(model, np.array(auc).mean()):
            print(f'test: best auc: {early_stopper.best_accuracy}')
            break

    model.load_state_dict(torch.load(save_path))
    auc, loss = test(model, test_data_loader, task_num, device)
    f = open('{}_{}.txt'.format(model_name, dataset_name), 'a', encoding='utf-8')
    f.write('learning rate: {}\n'.format(learning_rate))
    for i in range(task_num):
        print('task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
        f.write('task {}, AUC {}, Log-loss {}\n'.format(i, auc[i], loss[i]))
    print('\n')
    f.write('\n')
    f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='AliExpress_NL',
                        choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US'])
    parser.add_argument('--dataset_path', default='./data/')
    parser.add_argument('--model_name', default='metaheac',
                        choices=['singletask', 'sharedbottom', 'omoe', 'mmoe', 'ple', 'aitm'])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--mto_type', default='base',
                        choices=['base','mgda','mgdafw','paretomtl','nashmtl','bandit','imtl','imtlg','uncertainty','pcgrad','cagrad','famo'])
    parser.add_argument('--mto_normalization_type', default='none', choices=['l2','none','loss','loss+'])
    args = parser.parse_args()

    if args.seed:
        set_seed(args.seed)

    main(args.dataset_name,
         args.dataset_path,
         args.task_num,
         args.expert_num,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.embed_dim,
         args.weight_decay,
         args.device,
         args.save_dir)
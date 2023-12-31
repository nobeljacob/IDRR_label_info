import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import copy
import argparse
import logging
import gc
import datetime
import pprint
from collections import OrderedDict
from functools import partial
from torch.optim import Adam

torch.cuda.empty_cache()
try:
    from torch.utils.tensorboard import SummaryWriter
except BaseException as e:
    from tensorboardX import SummaryWriter
from model import BMGFModel
from dataset import Dataset, Sampler
from evaluate import evaluate_accuracy, evaluate_precision_recall_f1
from util import *

INF = 1e30
_INF = -1e30



def lcm_loss(true_labels, predicted_labels, label_similarity_dist, alpha):
    simulated_true_labels = F.softmax(F.softmax(label_similarity_dist) + alpha * F.one_hot(true_labels, args.num_rels))
    loss = nn.KLDivLoss(F.log_softmax(predicted_labels), simulated_true_labels)
    return loss


def eval_epoch(args, logger, writer, model, data_type, data_loader, device, epoch, lcm_setting):
    model.eval()
    epoch_step = len(data_loader)
    total_step = args.epochs * epoch_step
    total_cnt = 0
    total_ce = 0.0
    total_mlce = 0.0
    total_loss = 0.0

    results = {"data": {"id": list(), "relation": list(), "prefered_relation": list()},
               "prediction": {"prob": list(), "pred": list()},
               "error": {"ce": list(), "mlce": list(), "mean_ce": INF, "mean_mlce": INF},
               "evaluation": {"accuracy": dict(), "precision_recall_f1": dict()}}

    with torch.no_grad():
        for batch_id, batch in enumerate(data_loader):
            step = epoch * epoch_step + batch_id
            _id, arg1, arg1_mask, arg2, arg2_mask, relation, prefered_relation = batch
            bsz = len(_id)
            total_cnt += bsz

            results["data"]["id"].extend(_id)
            results["data"]["relation"].extend(relation)
            results["data"]["prefered_relation"].extend(prefered_relation)

            arg1 = arg1.to(device)
            arg2 = arg2.to(device)
            if arg1_mask is not None:
                arg1_mask = arg1_mask.to(device)
            if arg2_mask is not None:
                arg2_mask = arg2_mask.to(device)
            relation = relation.to(device)
            prefered_relation = prefered_relation.to(device)

            if lcm_setting:
                X_Label = []
                X_Label = np.array([np.array(range(args.num_rels)) for i in range(len(arg1))])
                X_Label = torch.IntTensor(X_Label).to(device)
                output, label_sim_dict = model(arg1=arg1, arg2=arg2, arg1_mask=arg1_mask, arg2_mask=arg2_mask,
                                               lcm_setting=lcm_setting,
                                               X_Label=X_Label)
                computed_lcm_loss = lcm_loss(true_labels=prefered_relation, predicted_labels=output, label_similarity_dist=label_sim_dict,
                                             alpha=args.alpha)


                logp = F.log_softmax(output, dim=1)
                prob = logp.exp()

            if not lcm_setting:
                output = model(arg1=arg1, arg2=arg2, arg1_mask=arg1_mask, arg2_mask=arg2_mask, lcm_setting=lcm_setting)
                logp = F.log_softmax(output, dim=-1)
                prob = logp.exp()

            results["prediction"]["prob"].extend(prob.cpu().detach())
            results["prediction"]["pred"].extend(prob.cpu().argmax(dim=1).detach())

            ce = F.nll_loss(logp, prefered_relation, reduction="none")
            mlce = F.multilabel_soft_margin_loss(output, relation, reduction="none")

            results["error"]["ce"].extend(ce.cpu().detach())
            results["error"]["mlce"].extend(mlce.cpu().detach())

            if args.loss == "ce":
                loss = ce
            if args.loss == "mlce":
                loss = mlce
            if lcm_setting:
                loss = computed_lcm_loss

            avg_ce = ce.mean()
            avg_mlce = mlce.mean()
            avg_loss = loss.mean()

            total_ce += avg_ce.item() * bsz
            total_mlce += avg_mlce.item() * bsz
            total_loss += avg_loss.item() * bsz

            if writer:
                writer.add_scalar("%s/pdtb-loss" % (data_type), avg_loss.item(), step)
                writer.add_scalar("%s/pdtb-ce" % (data_type), avg_ce.item(), step)
                writer.add_scalar("%s/pdtb-mlce" % (data_type), avg_mlce.item(), step)
            if logger and batch_id == epoch_step - 1:
                logger.info(
                    "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}".format(
                        epoch, args.epochs, data_type, batch_id, epoch_step) + "\n" +
                    "\tpdtb-loss: {:10.4f}\tpdtb-ce: {:10.4f}\tpdtb-mlce: {:10.4f}".format(
                        avg_loss.item(), avg_ce.item(), avg_mlce.item()) + "\n" +
                    "\tpdtb-gold: {}".format(results["data"]["relation"][-1]) + "\n" +
                    "\tpdtb-pred: {}".format(results["prediction"]["prob"][-1]))

        mean_ce = total_ce / (total_cnt + 1e-6)
        mean_mlce = total_mlce / (total_cnt + 1e-6)
        mean_loss = total_loss / (total_cnt + 1e-6)

        pred = np.array(results["prediction"]["pred"])
        target = torch.cat(results["data"]["relation"], dim=0).view(total_cnt, -1).int().numpy()
        prefered_target = np.array(results["data"]["prefered_relation"])

        results["error"]["mean_ce"] = mean_ce
        results["error"]["mean_mlce"] = mean_mlce
        results["evaluation"]["accuracy"] = evaluate_accuracy(pred, target, prefered_target)
        results["evaluation"]["precision_recall_f1"] = evaluate_precision_recall_f1(pred, target, prefered_target)

        if writer:
            writer.add_scalar("%s/pdtb-loss-epoch" % (data_type), mean_loss, epoch)
            writer.add_scalar("%s/pdtb-ce-epoch" % (data_type), mean_ce, epoch)
            writer.add_scalar("%s/pdtb-mlce-epoch" % (data_type), mean_mlce, epoch)

        if logger:
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}".format(
                    epoch, args.epochs, data_type) + "\n" +
                "\tpdtb-loss-epoch: {:10.4f}\tpdtb-ce-epoch: {:10.4f}\tpdtb-mlce-epoch: {:10.4f}".format(
                    mean_loss, mean_ce, mean_mlce) + "\n" +
                "\tpdtb-accuray: {}".format(
                    pprint.pformat(results["evaluation"]["accuracy"]).replace("\n", "\n\t\t")) + "\n" +
                "\tpdtb-precision_recall_f1: {}".format(
                    pprint.pformat(results["evaluation"]["precision_recall_f1"]).replace("\n", "\n\t\t")))
    gc.collect()
    return mean_loss, results


def train_epoch(args, logger, writer, model, optimizer, data_type, data_loader, device, epoch, lcm_setting):
    model.train()
    epoch_step = len(data_loader)
    total_step = args.epochs * epoch_step
    total_cnt = 0
    total_ce = 0.0
    total_mlce = 0.0
    total_loss = 0.0

    results = {"data": {"id": list(), "relation": list(), "prefered_relation": list()},
               "prediction": {"prob": list(), "pred": list()},
               "error": {"ce": list(), "mlce": list(), "mean_ce": INF, "mean_mlce": INF},
               "evaluation": {"accuracy": dict(), "precision_recall_f1": dict()}}

    for batch_id, batch in enumerate(data_loader):
        step = epoch * epoch_step + batch_id
        _id, arg1, arg1_mask, arg2, arg2_mask, relation, prefered_relation = batch
        bsz = len(_id)
        total_cnt += bsz

        results["data"]["id"].extend(_id)
        results["data"]["relation"].extend(relation)
        results["data"]["prefered_relation"].extend(prefered_relation)

        arg1 = arg1.to(device)
        arg2 = arg2.to(device)
        if arg1_mask is not None:
            arg1_mask = arg1_mask.to(device)
        if arg2_mask is not None:
            arg2_mask = arg2_mask.to(device)
        relation = relation.to(device)
        prefered_relation = prefered_relation.to(device)

        if lcm_setting:
            X_Label = []
            X_Label = np.array([np.array(range(args.num_rels)) for i in range(len(arg1))])
            X_Label = torch.IntTensor(X_Label).to(device)
            output, label_sim_dict = model(arg1=arg1, arg2=arg2, arg1_mask=arg1_mask, arg2_mask=arg2_mask,
                                           lcm_setting=lcm_setting, X_Label=X_Label)
            computed_lcm_loss = lcm_loss(true_labels=prefered_relation, predicted_labels=output, label_similarity_dist=label_sim_dict,
                                         alpha=args.alpha)
            logp = F.log_softmax(output, dim=1)
            prob = logp.exp()

        if not lcm_setting:
            output = model(arg1=arg1, arg2=arg2, arg1_mask=arg1_mask, arg2_mask=arg2_mask, lcm_setting=lcm_setting)
            logp = F.log_softmax(output, dim=1)
            prob = logp.exp()

        results["prediction"]["prob"].extend(prob.cpu().detach())
        results["prediction"]["pred"].extend(prob.cpu().argmax(dim=1).detach())

        ce = F.nll_loss(logp, prefered_relation, reduction="none")
        mlce = F.multilabel_soft_margin_loss(output, relation, reduction="none")

        results["error"]["ce"].extend(ce.cpu().detach())
        results["error"]["mlce"].extend(mlce.cpu().detach())

        # probably a bad programming practice, might have to change later

        if args.loss == "ce":
            loss = ce
        if args.loss == "mlce":
            loss = mlce
        if lcm_setting:
            loss = computed_lcm_loss

        # else:
        #    raise NotImplementedError("Error: loss=%s is not supported now." % (args.loss))

        avg_ce = ce.mean()
        avg_mlce = mlce.mean()
        avg_loss = loss.mean()

        total_ce += avg_ce.item() * bsz
        total_mlce += avg_mlce.item() * bsz
        total_loss += avg_loss.item() * bsz

        avg_loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if writer:
            writer.add_scalar("%s/pdtb-loss" % (data_type), avg_loss.item(), step)
            writer.add_scalar("%s/pdtb-ce" % (data_type), avg_ce.item(), step)
            writer.add_scalar("%s/pdtb-mlce" % (data_type), avg_mlce.item(), step)
        if logger and (batch_id % args.print_every == 0 or batch_id == epoch_step - 1):
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}".format(
                    epoch, args.epochs, data_type, batch_id, epoch_step) + "\n" +
                "\tpdtb-loss: {:10.4f}\tpdtb-ce: {:10.4f}\tpdtb-mlce: {:10.4f}".format(
                    avg_loss.item(), avg_ce.item(), avg_mlce.item()) + "\n" +
                "\tpdtb-gold: {}".format(results["data"]["relation"][-1]) + "\n" +
                "\tpdtb-pred: {}".format(results["prediction"]["prob"][-1]))

    mean_ce = total_ce / (total_cnt + 1e-6)
    mean_mlce = total_mlce / (total_cnt + 1e-6)
    mean_loss = total_loss / (total_cnt + 1e-6)

    pred = np.array(results["prediction"]["pred"])
    target = torch.cat(results["data"]["relation"], dim=0).view(total_cnt, -1).int().numpy()
    prefered_relation = np.array(results["data"]["prefered_relation"])

    results["error"]["mean_ce"] = mean_ce
    results["error"]["mean_mlce"] = mean_mlce
    results["evaluation"]["accuracy"] = evaluate_accuracy(pred, target, prefered_relation)
    results["evaluation"]["precision_recall_f1"] = evaluate_precision_recall_f1(pred, target, prefered_relation)

    if writer:
        writer.add_scalar("%s/pdtb-loss-epoch" % (data_type), mean_loss, epoch)
        writer.add_scalar("%s/pdtb-ce-epoch" % (data_type), mean_ce, epoch)
        writer.add_scalar("%s/pdtb-mlce-epoch" % (data_type), mean_mlce, epoch)

    if logger:
        logger.info(
            "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}".format(
                epoch, args.epochs, data_type) + "\n" +
            "\tpdtb-loss-epoch: {:10.4f}\tpdtb-ce-epoch: {:10.4f}\tpdtb-mlce-epoch: {:10.4f}".format(
                mean_loss, mean_ce, mean_mlce) + "\n" +
            "\tpdtb-accuray: {}".format(
                pprint.pformat(results["evaluation"]["accuracy"]).replace("\n", "\n\t\t")) + "\n" +
            "\tpdtb-precision_recall_f1: {}".format(
                pprint.pformat(results["evaluation"]["precision_recall_f1"]).replace("\n", "\n\t\t")))

    gc.collect()
    return mean_loss, results


def train(args, logger, writer, lcm_setting):
    # set device
    if args.gpu_ids is None:
        device = torch.device("cpu")
    else:
        if isinstance(args.gpu_ids, int):
            args.gpu_ids = [args.gpu_ids]
        device = torch.device("cuda:%d" % args.gpu_ids[0])
        torch.cuda.set_device(device)

    if args.pretrained_model_path:
        # load pretrained model
        config = load_config(os.path.join(args.pretrained_model_path, "BMGFModel.config"))
        for by in ["accf1", "f1", "accuracy", "loss"]:
            best_epochs = get_best_epochs(os.path.join(args.pretrained_model_path, "BMGFModel.log"), by=by)
            if len(best_epochs) > 0:
                break
        logger.info("retrieve the best epochs for BMGFModel: %s" % (best_epochs))
        if len(best_epochs) > 0:
            model = BMGFModel(**(config._asdict()))
            if "test" in best_epochs:
                model.load_state_dict(torch.load(
                    os.path.join(args.pretrained_model_path, "epoch%d.pt" % (best_epochs["test"])),
                    map_location=device))
            elif "valid" in best_epochs:
                model.load_state_dict(torch.load(
                    os.path.join(args.pretrained_model_path, "epoch%d.pt" % (best_epochs["valid"])),
                    map_location=device))
            else:
                model.load_state_dict(torch.load(
                    os.path.join(args.pretrained_model_path, "epoch%d.pt" % (best_epochs["train"])),
                    map_location=device))
            if config.dropout != args.dropout:
                change_dropout_rate(model, args.dropout)
        else:
            raise ValueError("Error: cannot load BMGFModel from %s." % (args.pretrained_model_path))
    else:
        # build model
        model = BMGFModel(**vars(args))
    model.set_finetune(args.finetune)
    if args.gpu_ids and len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu_ids)
    model = model.to(device)
    logger.info(model)
    logger.info("num of trainable parameters: %d" % (
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # load data
    datasets = OrderedDict({
        "train": Dataset().load_pt(args.train_dataset_path),
        "valid": Dataset().load_pt(args.valid_dataset_path),
        "test": Dataset().load_pt(args.test_dataset_path)})
    if args.explicit_dataset_path != "":
        explicit_dataset = Dataset().load_pt(args.explicit_dataset_path)
        datasets["train"].data.extend(explicit_dataset.data)
        del explicit_dataset
    for data_type, dataset in datasets.items():
        for x in dataset:
            x["arg1_len"] = len(x["arg1"])
            x["arg2_len"] = len(x["arg2"])

    logger.info("train:valid:test = %d:%d:%d" % (len(datasets["train"]), len(datasets["valid"]), len(datasets["test"])))

    if args.num_rels == 14:
        rel_map = Dataset.rel_map_14
    elif args.num_rels == 12:
        rel_map = Dataset.rel_map_12
    elif args.num_rels == 11:
        rel_map = Dataset.rel_map_11
    elif args.num_rels == 4:
        rel_map = Dataset.rel_map_4
    elif args.num_rels == 3:
        rel_map = Dataset.rel_map_3
    else:
        raise NotImplementedError("Error: num_rels=%d is not supported now." % (args.num_rels))
    if args.encoder == "roberta":
        pad_id = 1
    else:
        pad_id = 0
    data_loaders = OrderedDict()
    batchify = partial(Dataset.batchify,
                       rel_map=rel_map, min_arg=args.min_arg, max_arg=args.max_arg, pad_id=pad_id)
    for data_type in datasets:
        sampler = Sampler(datasets[data_type],
                          group_by=["arg1_len", "arg2_len"], batch_size=args.batch_size,
                          shuffle=data_type == "train", drop_last=False)
        data_loaders[data_type] = data.DataLoader(datasets[data_type],
                                                  batch_sampler=sampler,
                                                  collate_fn=batchify,
                                                  pin_memory=data_type == "train")

    # optimizer and losses
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    best_losses = {dataset: INF for dataset in datasets}
    best_loss_epochs = {dataset: -1 for dataset in datasets}
    best_accs = {dataset: _INF for dataset in datasets}
    best_acc_epochs = {dataset: -1 for dataset in datasets}
    best_f1s = {dataset: _INF for dataset in datasets}
    best_f1_epochs = {dataset: -1 for dataset in datasets}
    best_accf1s = {dataset: _INF for dataset in datasets}
    best_accf1_epochs = {dataset: -1 for dataset in datasets}

    for epoch in range(args.epochs):
        if epoch >= args.lcm_stop and lcm_setting:
            lcm_setting = False
            logger.info("*******************switching to normal mode************************")

        for data_type, data_loader in data_loaders.items():
            if data_type == "train":
                mean_loss, results = train_epoch(args, logger, writer,
                                                 model, optimizer, data_type, data_loader, device, epoch,
                                                 lcm_setting=lcm_setting)
            else:
                mean_loss, results = eval_epoch(args, logger, writer,
                                                model, data_type, data_loader, device, epoch, lcm_setting=lcm_setting)
                save_results(results, os.path.join(args.save_model_dir, "%s_results%d.json" % (data_type, epoch)))

            if mean_loss <= best_losses[data_type]:
                best_losses[data_type] = mean_loss
                best_loss_epochs[data_type] = epoch
                logger.info("data_type: {:<5s}\tbest pdtb-loss: {:.4f} (epoch: {:0>3d})".format(
                    data_type, best_losses[data_type], best_loss_epochs[data_type]))
                if args.save_best == "loss":
                    if args.gpu_ids and len(args.gpu_ids) > 1:
                        torch.save(model.module.state_dict(),
                                   os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                   _use_new_zipfile_serialization=False)
                    else:
                        torch.save(model.state_dict(),
                                   os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                   _use_new_zipfile_serialization=False)
            if results["evaluation"]["accuracy"]["overall"] >= best_accs[data_type]:
                best_accs[data_type] = results["evaluation"]["accuracy"]["overall"]
                best_acc_epochs[data_type] = epoch
                logger.info("data_type: {:<5s}\tbest pdtb-accuracy: {:.4f} (epoch: {:0>3d})".format(
                    data_type, best_accs[data_type], best_acc_epochs[data_type]))
                if args.save_best == "acc":
                    if args.gpu_ids and len(args.gpu_ids) > 1:
                        torch.save(model.module.state_dict(),
                                   os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                   _use_new_zipfile_serialization=False)
                    else:
                        torch.save(model.state_dict(),
                                   os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                   _use_new_zipfile_serialization=False)
            if results["evaluation"]["precision_recall_f1"]["overall"][-1] >= best_f1s[data_type]:
                best_f1s[data_type] = results["evaluation"]["precision_recall_f1"]["overall"][-1]
                best_f1_epochs[data_type] = epoch
                logger.info("data_type: {:<5s}\tbest pdtb-f1: {:.4f} (epoch: {:0>3d})".format(
                    data_type, best_f1s[data_type], best_f1_epochs[data_type]))
                if args.save_best == "f1":
                    if args.gpu_ids and len(args.gpu_ids) > 1:
                        torch.save(model.module.state_dict(),
                                   os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                   _use_new_zipfile_serialization=False)
                    else:
                        torch.save(model.state_dict(),
                                   os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                   _use_new_zipfile_serialization=False)
            if results["evaluation"]["accuracy"]["overall"] + results["evaluation"]["precision_recall_f1"]["overall"][
                -1] >= best_accf1s[data_type]:
                best_accf1s[data_type] = results["evaluation"]["accuracy"]["overall"] + \
                                         results["evaluation"]["precision_recall_f1"]["overall"][-1]
                best_accf1_epochs[data_type] = epoch
                logger.info("data_type: {:<5s}\tbest pdtb-accf1: {:.4f} (epoch: {:0>3d})".format(
                    data_type, best_accf1s[data_type], best_accf1_epochs[data_type]))
                if args.save_best == "accf1":
                    if args.gpu_ids and len(args.gpu_ids) > 1:
                        torch.save(model.module.state_dict(),
                                   os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                   _use_new_zipfile_serialization=False)
                    else:
                        torch.save(model.state_dict(),
                                   os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                   _use_new_zipfile_serialization=False)
    for data_type in data_loaders:
        logger.info("data_type: {:<5s}\tbest pdtb-loss: {:.4f} (epoch: {:0>3d})".format(
            data_type, best_losses[data_type], best_loss_epochs[data_type]))
        logger.info("data_type: {:<5s}\tbest pdtb-accuracy: {:.4f} (epoch: {:0>3d})".format(
            data_type, best_accs[data_type], best_acc_epochs[data_type]))
        logger.info("data_type: {:<5s}\tbest pdtb-f1: {:.4f} (epoch: {:0>3d})".format(
            data_type, best_f1s[data_type], best_f1_epochs[data_type]))
        logger.info("data_type: {:<5s}\tbest pdtb-accf1: {:.4f} (epoch: {:0>3d})".format(
            data_type, best_accf1s[data_type], best_accf1_epochs[data_type]))

    return best_accs["test"], best_f1s["test"], best_accf1s["test"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="numer of processors")

    # data config
    parser.add_argument("--explicit_dataset_path", type=str, default="",
                        help="explicit Dataset path")
    parser.add_argument("--train_dataset_path", type=str,
                        help="training Dataset path")
    parser.add_argument("--valid_dataset_path", type=str,
                        help="validation Dataset path")
    parser.add_argument("--test_dataset_path", type=str,
                        help="test Dataset path")
    parser.add_argument("--pretrained_model_path", type=str, default="",
                        help="model path of pretrained BMGFModel")
    parser.add_argument("--save_model_dir", type=str,
                        help="model dir to save models")
    parser.add_argument("--num_rels", type=int, default=4,
                        choices=[3, 4, 11, 12, 14],
                        help="how many relations are computed")
    parser.add_argument("--min_arg", type=int, default=3,
                        help="the minimum length of arguments")
    parser.add_argument("--max_arg", type=int, default=512,
                        help="the maximum length of arguments")

    # training config
    parser.add_argument("--gpu_ids", type=str2list, default=None,
                        help="gpu ids")
    parser.add_argument("--epochs", type=int, default=150,
                        help="epochs of training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size of training")
    parser.add_argument("--print_every", type=int, default=300,
                        help="printing log every K batchs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=2.0,
                        help="max grad norm for gradient clipping")
    parser.add_argument("--save_best", type=str, default="f1",
                        choices=["loss", "acc", "f1", "accf1"],
                        help="the criteria to save best models")
    parser.add_argument("--loss", type=str, default="ce",
                        choices=["ce", "mlce"],
                        help="loss function")

    # BMGFModel config
    parser.add_argument("--encoder", type=str, default="roberta",
                        choices=["lstm", "bert", "roberta"],
                        help="the encoder")
    parser.add_argument("--finetune", type=str, default="type",
                        choices=["none", "type", "last", "full"],
                        help="how to finetune the encoder")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="hidden dimension")
    parser.add_argument("--num_lstm_layers", type=int, default=1,
                        help="number of lstm layers")
    parser.add_argument("--num_perspectives", type=int, default=16,
                        help="number of perspectives for bimpm")
    parser.add_argument("--num_filters", type=int, default=64,
                        help="number of filters for convolutional layers")
    parser.add_argument("--activation", type=str, default="leaky_relu",
                        choices=["relu", "tanh", "softmax", "sigmoid", "leaky_relu", "prelu", "gelu"],
                        help="activation function type")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout for neural networks")

    args = parser.parse_args()

    ts = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    args.save_model_dir = os.path.join(args.save_model_dir, ts)
    os.makedirs(args.save_model_dir, exist_ok=True)

    # save config
    save_config(args, os.path.join(args.save_model_dir, "BMGFModel.config"))

    # build logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(os.path.join(args.save_model_dir, "BMGFModel.log"), 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    # build writer
    writer = SummaryWriter(args.save_model_dir)

    ################################################################### cross-check changes below this############################################

    accuracies_test = []
    f_scores_test = []
    acc_f1_test_scores = []

    args.lcm_stop = 100
    args.alpha = 4
    logger.info(args)

    runs = range(0, 5)
    for run_ids in runs:
        torch.manual_seed(run_ids)
        np.random.seed(run_ids)
        logger.info('alpha equals=%s', args.alpha)
        accuracy, f1, acc_f1 = train(args, logger, writer, lcm_setting=True)
        accuracies_test.append(accuracy)
        f_scores_test.append(f1)
        acc_f1_test_scores.append(acc_f1)
        logger.info(
            '###############################################################################space############################################################')
        logger.info(
            '###############################################################################space############################################################')

    logger.info('avg accuracy : %s', (str(np.mean(accuracies_test))))
    logger.info('avg f scores : %s', (str(np.mean(f_scores_test))))
    logger.info('avg acc_f1 : %s', (str(np.mean(acc_f1_test_scores))))

    logger.info("list of all values")

    logger.info('accuracy  %s', accuracies_test)
    logger.info("f1  %s", f_scores_test)
    logger.info("accuracy f1  %s", acc_f1_test_scores)
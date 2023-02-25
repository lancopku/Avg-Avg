import os
import torch
import argparse
from loguru import logger
from datetime import datetime
from torch.nn import DataParallel
from transformers import AdamW

from lib.data_loader import  get_data_loader
from lib.models.networks import get_model, get_tokenizer
from lib.training.common import train_common, test_acc
from lib.training.RecAdam import RecAdam
from lib.exp import get_num_labels, seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3',
                        type=str, required=False, help='GPU ids')
    parser.add_argument('--model', default='roberta-base',help='pretrained model type')
    parser.add_argument('--pretrained_model', default=None,
                        type=str, required=False, help='the path of the model to load')
    parser.add_argument('--dataset', default='sst-2', help='training dataset')
    parser.add_argument('--epochs', default=5, type=int,
                        required=False, help='number of training epochs')
    parser.add_argument('--batch_size', default=16, type=int,
                        required=False, help='training batch size')
    parser.add_argument('--lr', default=2e-5, type=float,
                        required=False, help='learning rate')
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--shift_reg", default=0, type=float)
    parser.add_argument('--log_step', default=100, type=int,required=False)
    parser.add_argument('--max_grad_norm', default=1.0,
                        type=float, required=False)
    parser.add_argument('--output_dir', default='saved_model/',
                        type=str, required=False, help='save directory')
    parser.add_argument('--save_every_epoch', action='store_true',
                        help='save checkpoint every epoch')
    parser.add_argument('--output_name', default='model.pt',
                        type=str, required=False, help='model save name')
    parser.add_argument('--log_file', type=str, default='./log/default.log')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimizer', type=str,default='adam', choices=['adam', 'recadam'])
    parser.add_argument("--recadam_anneal_fun", type=str, default='sigmoid', 
        choices=["sigmoid", "linear", 'constant'],
        help="the type of annealing function in RecAdam. Default sigmoid")
    parser.add_argument("--recadam_anneal_k", type=float, default=0.2,
        help="k for the annealing function in RecAdam.")
    parser.add_argument("--recadam_anneal_t0", type=int, default=1000,
        help="t0 for the annealing function in RecAdam.")
    parser.add_argument("--recadam_anneal_w", type=float, default=1.0,
        help="Weight for the annealing function in RecAdam. Default 1.0.")
    parser.add_argument("--recadam_pretrain_cof", type=float, default=5000.0,
        help="Coefficient of the quadratic penalty in RecAdam. Default 5000.0.")
    parser.add_argument("--loss_type", default='ce',choices=['ce', 'scl', 'margin'])
    parser.add_argument("--scl_reg", default=2.0, type=float)
    parser.add_argument("--eval_metric", default='acc',type=str, choices=['acc', 'f1'])
    parser.add_argument("--save_steps", default=-1, type=int)
    args = parser.parse_args()
    seed_everything(args.seed)
    # args setting
    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

    output_dir = args.output_dir
    output_name = args.output_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_labels = get_num_labels(args.dataset)
    args.num_labels = num_labels

    # model loading
    model = get_model(args)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    logger.info("{} model loaded".format(args.model))
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))
        logger.info("model loaded from {}".format(args.pretrained_model))

    if torch.cuda.device_count() > 1:
        logger.info("Let's use " + str(len(args.device.split(','))) + " GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
    tokenizer = get_tokenizer(args.model)
    logger.info("{} tokenizer loaded".format(args.model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # data loading
    train_loader = get_data_loader(args.dataset, 'train', tokenizer, args.batch_size, shuffle=True)
    val_loader = get_data_loader(
        args.dataset, 'dev', tokenizer, args.batch_size)
    logger.info("dataset {} loaded".format(args.dataset))
    logger.info("num_labels: {}".format(num_labels))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.train()
    model.to(device)
    pretrained_model = get_model(args)
    pretrained_model.to(device)
    if torch.cuda.device_count() > 1:
        pretrained_model = DataParallel(pretrained_model, device_ids=[int(i) for i in args.device.split(',')])
    pretrained_model.eval()
    no_decay = ["bias", "LayerNorm.weight"]
    if args.optimizer == 'adam':
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    elif args.optimizer == 'recadam':
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and 'bert' in n.lower()],
                "weight_decay": args.weight_decay,
                "anneal_w": args.recadam_anneal_w,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                    not any(nd in p_n for nd in no_decay) and 'bert' in p_n.lower()]
            },
            {
                "params": [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and 'bert' not in n.lower()],
                "weight_decay": args.weight_decay,
                "anneal_w": 0.0,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                    not any(nd in p_n for nd in no_decay) and 'bert' not in p_n.lower()]
            },
            {
                "params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and 'bert' in n.lower()],
                "weight_decay": 0.0,
                "anneal_w": args.recadam_anneal_w,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                    any(nd in p_n for nd in no_decay) and 'bert' in p_n.lower()]
            },
            {
                "params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and 'bert' not in n.lower()],
                "weight_decay": 0.0,
                "anneal_w": 0.0,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                    any(nd in p_n for nd in no_decay) and 'bert' not in p_n.lower()]
            }
        ]
        optimizer = RecAdam(optimizer_grouped_parameters, lr=args.lr,
            anneal_fun=args.recadam_anneal_fun, anneal_k=args.recadam_anneal_k,
            anneal_t0=args.recadam_anneal_t0, pretrain_cof=args.recadam_pretrain_cof)
    logger.info('starting training')

    best_acc = 0
    step_counter = 0
    for epoch in range(args.epochs):
        model.train()
        pretrained_model.eval()
        logger.info("epoch {} start".format(epoch))
        start_time = datetime.now()

        step_counter = train_common(model, optimizer, train_loader,
            epoch, log_steps=args.log_step, pre_model=pretrained_model, shift_reg=args.shift_reg,
            loss_type=args.loss_type, scl_reg=args.scl_reg,
            save_steps=args.save_steps, save_dir=output_dir, step_counter=step_counter)

        acc = test_acc(model, val_loader, args.eval_metric)
        logger.info("epoch {} validation {}: {:.4f} ".format(epoch, args.eval_metric, acc))
        if acc > best_acc:
            best_acc = acc
            logger.info("best validation {} improved to {:.4f}".format(
                args.eval_metric, best_acc))
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(),'{}/{}'.format(output_dir, output_name))
            logger.info("model saved to {}/{}".format(output_dir, output_name))
        if args.save_every_epoch:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(),'{}/epoch{}_{}'.format(output_dir, epoch, output_name))
            logger.info("model saved to {}/epoch{}_{}".format(output_dir, epoch, output_name))

        end_time = datetime.now()
        logger.info('time for one epoch: {}'.format(end_time - start_time))

    logger.info("training finished")
    test_loader = get_data_loader(args.dataset, 'test', tokenizer, args.batch_size)
    model = get_model(args)
    model.to(device)
    model.load_state_dict(torch.load('{}/{}'.format(output_dir, output_name)))
    logger.info("best model loaded")
    logger.info("test {}: {:.4f}".format(args.eval_metric,test_acc(model, test_loader, args.eval_metric)))

if __name__ == '__main__':
    main()

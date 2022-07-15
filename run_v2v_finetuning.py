import argparse
import os
import pickle
import pprint
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import wandb
from timm.models import create_model
from torch.backends import cudnn
from torch.nn.parallel.data_parallel import data_parallel
from tqdm import tqdm

import modeling_finetune
import utils
from datasets import V2VListwiseDataset, V2VPairwiseDataset

cudnn.benchmark = True

def evaluate(args, model, loader):
    if args.mode == "eval":
        all_preds_ours = []
        all_labels = []
        all_paths = []

    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()
    with torch.no_grad():
        running_loss = 0
        count = 0
        correct = 0
        for batch in tqdm(loader, disable=args.disable_tqdm):
            bx = batch[0].cuda()
            count += len(batch[0]) // 2

            logits = model(bx)
            diffs = (logits[range(1, logits.shape[0], 2)] - logits[range(0,logits.shape[0], 2)]).squeeze(dim=1)
            loss = criterion(diffs, torch.ones(diffs.shape[0]).cuda())
            
            for idx, diff in enumerate(diffs.cpu()):
                # logit1-logit0 should always be >0 since video1 is always preferred over video2
                if diff > 0:
                    correct += 1
            running_loss += loss.cpu().data.numpy()
        loss = running_loss / count
    return loss, correct / count

def evaluate_listwise(args, model, loader):
    all_preds= []
    all_listlengths = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, disable=args.disable_tqdm):
            bx = batch[0].cuda()
            logits = model(bx)
            preds = logits.cpu().data.numpy()
            all_preds.append(preds)

            list_lengths = batch[1]
            for l in list_lengths:
                all_listlengths.append(l)
    
    all_preds = np.concatenate(all_preds).squeeze()
    assert sum(all_listlengths) == len(all_preds), (sum(all_listlengths), len(all_preds))

    index = 0
    correct = 0
    for length in all_listlengths:
        prev = -float('inf')
        curr_correct = True
        # Correct predictions must be in sorted order (small to large)
        for i in range(length):
            if all_preds[index] < prev:
                curr_correct = False
            prev = all_preds[index]
            index += 1
        if curr_correct:
            correct += 1
    total = len(all_listlengths)
    print("Total: {}, Correct: {}".format(total, correct))
    print("Accuracy: {}".format(correct / total))
    return correct / total
    
def train(args, model, train_data, test_data, listwise_data, use_wandb=True, accumulation_steps=1):
    if use_wandb:
        wandb.init(project='WellbeingVideo', entity='junshern')
        wandb.config.update(args) # add all argparse args as config variables
    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_pairwise_batch, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_pairwise_batch, pin_memory=True, drop_last=True)
    list_loader = torch.utils.data.DataLoader(
            listwise_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_listwise_batch, pin_memory=True, drop_last=True)
    num_epochs = args.num_epochs
    results_path = Path(args.results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs)

    loss_ema = np.inf

    criterion = torch.nn.BCEWithLogitsLoss()
    max_test_acc = 0
    step = 0
    for epoch in range(num_epochs):
        print(f"EPOCH {epoch} ----------------------------------------------------------------------------")
        epoch_start_time = time.time()
        
        # Evaluate listwise
        print("Running list eval...")
        listwise_start_t = time.time()
        model.train(False)
        accuracy = evaluate_listwise(args, model, list_loader)
        if use_wandb:
            wandb.log({"listwise/accuracy": accuracy, "step": step, "epoch": epoch})
        print("Listwise accuracy: {:.3f}".format(accuracy))
        print(f"Listwise time taken: {time.time() - listwise_start_t}")
        model.train(True)

        # Evaluate pairwise
        print("Running test eval...")
        test_start_t = time.time()
        model.train(False)
        loss, accuracy = evaluate(args, model, test_loader)
        if accuracy > max_test_acc:
            max_test_acc = accuracy
            torch.save(model.state_dict(), results_path / f"{args.results_prefix}_epoch{epoch}.pt")
            prev_path = results_path / f"{args.results_prefix}_epoch{epoch - 1}.pt"
            if prev_path.exists():
                prev_path.unlink() # Remove file
        if use_wandb:
            wandb.log({"test/loss": loss, "test/accuracy": accuracy, "step": step, "epoch": epoch})
        print('\nTest Loss: {:.3f}, Accuracy: {:.3f}'.format(loss, accuracy))
        print(f"Test time taken: {time.time() - test_start_t}")
        model.train(True)


        printout_start_time = time.time()
        correct = 0
        count = 0
        print(f"Training...")
        for i, batch in enumerate(tqdm(train_loader, disable=args.disable_tqdm)):
            bx = batch[0].cuda()
            count += len(batch[0]) // 2

            logits = model(bx)
            diffs = (logits[range(1, logits.shape[0], 2)] - logits[range(0,logits.shape[0], 2)]).squeeze(dim=1)
            loss = criterion(diffs, torch.ones(diffs.shape[0]).cuda())

            model.zero_grad()
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()

            # logit1-logit0 should always be >0 since video1 is always preferred over video2
            for diff in diffs.cpu():
                if diff > 0:
                    correct += 1

            if step % 50 == 0:
                acc = correct / count
                print(f'\nEpoch: {epoch}, Batch: {i}\nLoss: {loss:.3f} Accuracy: {acc:.3f} \tTime: {time.time() - printout_start_time}')
                if use_wandb:
                    wandb.log({"train/loss": loss, "train/accuracy": acc, "step": step, "epoch": step / len(train_loader)})
                printout_start_time = time.time()
            step += 1

        print('Time taken (s):', time.time() - epoch_start_time)

    # evaluate
    model.train(False)
    loss, accuracy = evaluate(args, model, test_loader)
    if use_wandb:
        wandb.log({"Test Loss": loss, "Test Accuracy": accuracy})
    print('\nTest Loss: {:.3f}'.format(loss))
    model.train(True)

    torch.save(model.state_dict(), results_path / f"{args.results_prefix}_final.pt")

def load_wellbeing_datasets(args):
    train_dataset = V2VPairwiseDataset(args.data_dir, split="train", video_width=args.input_size, video_height=args.input_size, num_frames=args.num_frames)
    test_dataset = V2VPairwiseDataset(args.data_dir, split="test", video_width=args.input_size, video_height=args.input_size, num_frames=args.num_frames)
    listwise_dataset = V2VListwiseDataset(args.data_dir, video_width=args.input_size, video_height=args.input_size, num_frames=args.num_frames)
    # train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
    # test_dataset = torch.utils.data.Subset(test_dataset, range(100))
    # listwise_dataset = torch.utils.data.Subset(listwise_dataset, range(100))
    return train_dataset, test_dataset, listwise_dataset

def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def collate_listwise_batch(batch):
    videos_concatenated, list_lengths = [], []
    for (videos_list, metadata_list) in batch:
        for video in videos_list:
            videos_concatenated.append(video)
        list_lengths.append(len(videos_list))
    stacked_videos = torch.from_numpy(np.stack(videos_concatenated)) # B T H W C
    # Normalize image colors
    stacked_videos = tensor_normalize(
        stacked_videos, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    # Convert from B T H W C -> B C T H W
    stacked_videos = stacked_videos.permute(0, 4, 1, 2, 3)
    assert stacked_videos.shape == (len(videos_concatenated), video.shape[3], video.shape[0], video.shape[1], video.shape[2]), \
        (stacked_videos.shape, (len(videos_concatenated), video.shape[3], video.shape[0], video.shape[1], video.shape[2]))
    return stacked_videos, list_lengths

def collate_pairwise_batch(batch):
    videos_alternating, metadata_1, metadata_2 = [], [], []
    for (video_1, video_2, md_1, md_2) in batch:
        videos_alternating.append(video_1)
        videos_alternating.append(video_2)
        metadata_1.append(md_1)
        metadata_2.append(md_2)
    stacked_videos = torch.from_numpy(np.stack(videos_alternating)) # B T H W C
    # Normalize image colors
    stacked_videos = tensor_normalize(
        stacked_videos, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    # Convert from B T H W C -> B C T H W
    stacked_videos = stacked_videos.permute(0, 4, 1, 2, 3)
    assert stacked_videos.shape == (len(batch) * 2, video_1.shape[3], video_1.shape[0], video_1.shape[1], video_1.shape[2]), \
        (stacked_videos.shape, (len(batch) * 2, video_1.shape[3], video_1.shape[0], video_1.shape[1], video_1.shape[2]))
    return stacked_videos, metadata_1, metadata_2

def load_videomae_model(args):
    device = torch.device('cuda')

    # Create model
    print(args.model)
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=2, # Doesn't matter, we will replace the output layer
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=args.init_scale,
    )
    # Replace classification head with a single regression output
    final_layer_num_inputs = model.get_classifier().in_features
    model.head = torch.nn.Sequential(
        torch.nn.Linear(in_features=final_layer_num_inputs, out_features=1, bias=False)
    )

    # Load weights
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        print("Load ckpt from %s" % args.checkpoint)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model))
    print('number of params:', n_parameters)

    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
    return model


def main(args):
    torch.cuda.set_device("cuda:0")
    if args.mode == 'train':
        train_data, test_data, listwise_data = load_wellbeing_datasets(args)
        model = load_videomae_model(args)
        train(args, model, train_data, test_data, listwise_data, use_wandb=args.wandb)
    elif args.mode == 'eval':
        train_data, test_data, listwise_data = load_wellbeing_datasets(args)
        model = load_videomae_model(args)

        test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, collate_fn=collate_pairwise_batch, pin_memory=True, drop_last=False)
        evaluate(args, model, test_loader)

        list_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, collate_fn=collate_listwise_batch, pin_memory=True, drop_last=False)
        evaluate_listwise(args, model, list_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training script params')

    ## SCRIPT MODE
    parser.add_argument('--mode', type=str, default='train')

    ## ARGUMENTS SPECIFYING PATHS
    parser.add_argument("-d", "--data_dir", required=True, help="Dataset directory.")
    parser.add_argument('--results_path', type=str, required=True, help="Path to folder where result will be output")
    parser.add_argument('--suffix', type=str, default='_64_256_crop224_wellbeing.pkl', help="Suffix describing how videos were processed for metadata")
    parser.add_argument('--results_prefix', type=str, default='wellbeing', help="Prefix describing relevant parameters for trained model")
    parser.add_argument('--wandb', type=int, default=0, help="Whether to use wandb (0 for False, 1 for True)")
    parser.add_argument('--disable_tqdm', default=0, type=int)
    ## ARGUMENTS FOR LOADING DATA
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for model training")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of Workers for dataloaders")
    parser.add_argument('--num_frames', type=int, default= 16)
    ## ARGUMENTS FOR MODEL TRAINING
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs to train models")
    parser.add_argument('--lr', type=float, default=0.01, help="Optimizer learning rate")
    parser.add_argument('--weight_decay', type=int, default=0, help="Optimizer weight decay")
    parser.add_argument('--momentum', type=float, default=0.9, help="Optimizer momentum")
    parser.add_argument('--num_gpus', type=int, default=1,help="Number of GPUS to use to train/eval model (used by DataParallel)")
    ## ARGUMENTS FOR EVALUATING PRETRAINED MODELS
    parser.add_argument('--eval_model_path', type=str, default=None, help="Path to load model for eval")
    parser.add_argument('--eval_results_path', type=str, default=None,help="Path to output results from eval")

    ## VideoMAE model params
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', type=int, default=224, help="Size in pixels of model input (input_size x input_size)")
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT', help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    ## Finetuning params
    parser.add_argument('--checkpoint', type=str, default=None, help='finetune from checkpoint')
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    args = parser.parse_args()
    main(args)

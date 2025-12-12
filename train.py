import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SoccerDataset
from models import load_model
from models.utils import (
    calc_class_acc,
    calc_real_loss,
    calc_speed,
    calc_trace_dist,
    l1_regularizer,
    num_trainable_params,
)


# Helper functions
def printlog(line):
    print(line)
    with open(save_path + "/log.txt", "a") as file:
        file.write(line + "\n")


def loss_str(losses: dict):
    ret = ""
    for key, value in losses.items():
        ret += f" {key}: {np.mean(value):.4f} |"
    # if len(losses) > 1:
    #     ret += " total_loss: {:.4f} |".format(sum(losses.values()))
    return ret[:-2]


def hyperparams_str(epoch, hp):
    ret = f"\nEpoch {epoch:d}"
    if hp["pretrain"]:
        ret += " (pretrain)"
    return ret


# For one epoch
def run_epoch(model: nn.DataParallel, optimizer: torch.optim.Adam, train=False, print_batch=50):
    # torch.autograd.set_detect_anomaly(True)
    model.train() if train else model.eval()

    loader = train_loader if train else valid_loader
    n_batches = len(loader)

    if model.module.model_type == "classifier":
        loss_dict = {"ce_loss": [], "accuracy": []}

    elif model.module.model_type == "regressor":
        loss_dict = {"mse_loss": [], "pos_error": []}

    elif model.module.model_type == "generator":
        loss_dict = {"kld_loss": [], "recon_loss": [], "pos_error": []}

    elif model.module.model_type == "macro_classifier":
        loss_dict = {"macro_ce_loss": [], "micro_ce_loss": [], "macro_acc": [], "micro_acc": []}

    elif model.module.model_type == "macro_regressor":
        loss_dict = {"ce_loss": [], "mse_loss": [], "accuracy": [], "pos_error": []}

    if model.module.params.get("rloss_weight") > 0:
        loss_dict["real_loss"] = []

    if train and model.module.params.get("l1_weight") > 0:
        loss_dict["l1_loss"] = []

    for batch_idx, data in enumerate(loader):
        if model.module.model_type == "classifier":
            input = data[0].to(default_device)
            target = data[1].to(default_device)

            if train:
                out = model(input).transpose(1, 2)
            else:
                with torch.no_grad():
                    out = model(input).transpose(1, 2)

            loss = nn.CrossEntropyLoss()(out, target)
            loss_dict["ce_loss"] += [loss.item()]
            loss_dict["accuracy"] += [calc_class_acc(out, target)]

        elif model.module.model_type == "regressor":
            input = data[0].to(default_device)
            target = data[1].to(default_device)

            if train:
                out = model(input)
            else:
                with torch.no_grad():
                    out = model(input)

            if "speed_loss" in model.module.params and model.module.params["speed_loss"]:
                out = calc_speed(out)

            loss = nn.MSELoss()(out, target)
            loss_dict["mse_loss"] += [loss.item()]

            n_features = model.module.params["n_features"]
            rloss_weight = model.module.params.get("rloss_weight")

            if rloss_weight > 0:
                real_loss = calc_real_loss(out[:, :, 0:2], input, n_features)
                loss += real_loss * rloss_weight
                loss_dict["real_loss"] += [real_loss.item()]

            if model.module.target_type == "gk":
                team1_pos_error = calc_trace_dist(out[:, :, 0:2], target[:, :, 0:2])
                team2_pos_error = calc_trace_dist(out[:, :, 2:4], target[:, :, 2:4])
                loss_dict["pos_error"] += [(team1_pos_error + team2_pos_error) / 2]
            else:
                loss_dict["pos_error"] += [calc_trace_dist(out[:, :, 0:2], target[:, :, 0:2])]

        elif model.module.model_type == "generator":
            input = data[0].to(default_device)
            target = data[1].to(default_device)

            kld_weight = model.module.params["kld_weight"]
            if train:
                loss_tensor = model(input, target).mean(0)
                loss = loss_tensor[0] * kld_weight + loss_tensor[1]  # kld_loss + recon_loss
            else:
                with torch.no_grad():
                    loss_tensor = model(input, target).mean(0)

            loss_dict["kld_loss"] += [loss_tensor[0].item() * kld_weight]
            loss_dict["recon_loss"] += [loss_tensor[1].item()]
            loss_dict["pos_error"] += [loss_tensor[2].item()]

        elif model.module.model_type.startswith("macro"):
            input = data[0].to(default_device)
            macro_target = data[1].to(default_device)
            micro_target = data[2].to(default_device)

            # Mask the target trajectories for the model to leverage
            if model.module.model_type == "player_ball" and "masking" in model.module.params:
                if train and np.random.choice([True, False], p=[0.5, 0.5]):
                    masking_prob = 1
                else:
                    masking_prob = model.module.params["masking"]
                random_numbers = torch.FloatTensor(input.size(1), input.size(0), 1).uniform_()
                random_mask = (random_numbers > masking_prob).to(default_device)

                if train:
                    out = model(input, macro_target, micro_target, random_mask)
                else:
                    with torch.no_grad():
                        out = model(input, macro_target, micro_target, random_mask)

            else:
                if train:
                    out = model(input)
                else:
                    with torch.no_grad():
                        out = model(input)

            micro_dim = model.module.micro_dim  # 4 if target_type == "gk" else 2
            macro_out = out[:, :, :-micro_dim].transpose(1, 2)
            macro_weight = model.module.params["macro_weight"]
            macro_loss = nn.CrossEntropyLoss()(macro_out, macro_target) * macro_weight

            if model.module.model_type == "macro_classifier":
                micro_out = out[:, :, -micro_dim:].transpose(1, 2)
                micro_loss = nn.CrossEntropyLoss()(micro_out, micro_target)
                loss = macro_loss + micro_loss

                loss_dict["macro_ce_loss"] += [macro_loss.item()]
                loss_dict["micro_ce_loss"] += [micro_loss.item()]
                loss_dict["macro_acc"] += [calc_class_acc(macro_out, macro_target)]
                loss_dict["micro_acc"] += [calc_class_acc(micro_out, micro_target)]

            else:  # model.module.model_type == "macro_regressor"
                micro_out = out[:, :, -micro_dim:]
                if "speed_loss" in model.module.params and model.module.params["speed_loss"]:
                    micro_out = calc_speed(micro_out)
                micro_loss = nn.MSELoss()(micro_out, micro_target)

                n_features = model.module.params["n_features"]
                real_loss = calc_real_loss(micro_out[:, :, 0:2], input, n_features)

                loss = macro_loss + micro_loss
                loss_dict["ce_loss"] += [macro_loss.item()]
                loss_dict["mse_loss"] += [micro_loss.item()]

                rloss_weight = model.module.params.get("rloss_weight")
                if rloss_weight > 0:
                    loss += real_loss * rloss_weight
                    loss_dict["real_loss"] += [real_loss.item()]

                loss_dict["accuracy"] += [calc_class_acc(macro_out, macro_target)]
                if model.module.target_type == "gk":
                    team1_pos_error = calc_trace_dist(micro_out[:, :, 0:2], micro_target[:, :, 0:2])
                    team2_pos_error = calc_trace_dist(micro_out[:, :, 2:4], micro_target[:, :, 2:4])
                    loss_dict["pos_error"] += [(team1_pos_error + team2_pos_error) / 2]
                else:
                    loss_dict["pos_error"] += [calc_trace_dist(micro_out[:, :, 0:2], micro_target[:, :, 0:2])]

        if train and args.l1_weight > 0:
            l1_loss = l1_regularizer(model.module)
            loss += args.l1_weight * l1_loss
            loss_dict["l1_loss"] += [(args.l1_weight * l1_loss).item()]

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.module.parameters(), args.clip)
            optimizer.step()

        if train and batch_idx % print_batch == 0:
            print(f"[{batch_idx:>{len(str(n_batches))}d}/{n_batches}]  {loss_str(loss_dict)}")

    for key, value in loss_dict.items():
        loss_dict[key] = np.mean(value)  # /= len(loader.dataset)

    return loss_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--model", type=str, required=False, default="player_ball")
    parser.add_argument("--macro_type", type=str, required=False, default="player_poss", help="Type of macro-intents")
    parser.add_argument("--target_type", type=str, required=False, default="ball", choices=[None, "gk", "ball"])

    parser.add_argument("--n_features", type=int, required=False, default=2, help="Num features")
    parser.add_argument("--window_seconds", type=float, required=False, default=10.0, help="Length of a window")
    parser.add_argument("--window_stride", type=int, required=False, default=5, help="Step size between windows")
    parser.add_argument("--flip_pitch", action="store_true", default=False, help="Augment data by flipping the pitch")

    parser.add_argument("--macro_weight", type=float, required=False, default=20, help="Weight for the macro loss")
    parser.add_argument("--rloss_weight", type=float, required=False, default=0, help="Weight for the reality loss")
    parser.add_argument("--l1_weight", type=float, required=False, default=0, help="Weight for the L1 loss")
    parser.add_argument("--kld_weight", type=float, required=False, default=1, help="Weight for the KLD loss in VRNN")
    parser.add_argument("--speed_loss", action="store_true", default=False, help="Include speed loss in MSE")
    parser.add_argument("--masking", type=float, required=False, default=1, help="Masking proportion of the target")
    parser.add_argument("--prev_out_aware", action="store_true", default=False, help="Input previous outputs to RNN")
    parser.add_argument("--bidirectional", action="store_true", default=False, help="Use bidirectional RNNs")

    parser.add_argument("--n_epochs", type=int, required=False, default=200, help="Num epochs")
    parser.add_argument("--batch_size", type=int, required=False, default=32, help="Batch size")
    parser.add_argument("--start_lr", type=float, required=False, default=0.0001, help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, required=False, default=0.0001, help="Minimum learning rate")
    parser.add_argument("--clip", type=int, required=False, default=10, help="Gradient clipping")

    parser.add_argument("--print_batch", type=int, required=False, default=50, help="Periodically print performance")
    parser.add_argument("--save_epoch", type=int, required=False, default=10, help="periodically save model")
    parser.add_argument("--pretrain_time", type=int, required=False, default=0, help="Num epochs to train macro policy")
    parser.add_argument("--seed", type=int, required=False, default=128, help="PyTorch random seed")
    parser.add_argument("--cont", action="store_true", default=False, help="Continue training previous best model")
    parser.add_argument("--best_total_loss", type=float, required=False, default=0, help="Best total loss")
    parser.add_argument("--best_pos_error", type=float, required=False, default=0, help="Best position error")

    args, _ = parser.parse_known_args()
    args_dict = vars(args)

    # Set device and manual seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        default_device = "cuda:0"
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    else:
        default_device = "cpu"

    # Load model
    model = load_model(args.model, args_dict, parser).to(default_device)
    model = nn.DataParallel(model)

    # Update params with model parameters
    args_dict = model.module.params
    args_dict["total_params"] = num_trainable_params(model)

    # Create save path and saving parameters
    save_path = f"saved/{args.trial:03d}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "/model")
    with open(f"{save_path}/args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    # Continue a previous experiment or start a new one
    if args.cont:
        state_dict = torch.load(f"{save_path}/model/state_dict_best_pe.pt", weights_only=False)
        model.module.load_state_dict(state_dict)

    data_dir = "data/sportec/tracking_processed"
    data_paths = [f"{data_dir}/{f}" for f in os.listdir(data_dir)]
    data_paths.sort()

    train_paths = data_paths[:5]
    valid_paths = data_paths[5:6]

    print("Generating datasets...")
    dataset_args = {
        "macro_type": args.macro_type,
        "target_type": args.target_type,
        "n_features": args.n_features,
        "window_seconds": args.window_seconds,
        "window_stride": args.window_stride,
        "target_speed": args.speed_loss,
        "flip_pitch": args.flip_pitch,
    }
    nw = len(model.device_ids) * 4
    train_dataset = SoccerDataset(train_paths, **dataset_args)
    test_dataset = SoccerDataset(valid_paths, **dataset_args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    valid_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)

    # Train loop
    best_total_loss = args.best_total_loss
    best_pos_error = args.best_pos_error
    epochs_since_best = 0
    lr = max(args.start_lr, args.min_lr)

    for e in range(args.n_epochs):
        epoch = e + 1
        hyperparams = {"pretrain": epoch <= args.pretrain_time}

        # Set a custom learning rate schedule
        if epochs_since_best == 3 and lr > args.min_lr:
            # Load previous best model
            path = f"{save_path}/model/state_dict_best.pt"
            if epoch <= args.pretrain_time:
                path = f"{save_path}/model/state_dict_best_pretrain.pt"
            state_dict = torch.load(path, weights_only=False)

            # Decrease learning rate
            lr = max(lr * 0.5, args.min_lr)
            printlog(f"########## lr {lr} ##########")
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # Remove parameters with requires_grad=False (https://github.com/pytorch/pytorch/issues/679)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)

        printlog(hyperparams_str(epoch, hyperparams))
        start_time = time.time()

        train_losses = run_epoch(model, optimizer, train=True, print_batch=args.print_batch)
        printlog("Train:\t" + loss_str(train_losses))

        valid_losses = run_epoch(model, optimizer, train=False)
        printlog("Test:\t" + loss_str(valid_losses))

        epoch_time = time.time() - start_time
        printlog(f"Time:\t {epoch_time:.2f}s")

        valid_total_loss = sum([value for key, value in valid_losses.items() if key.endswith("loss")])

        # Best model on test set
        if best_total_loss == 0 or valid_total_loss < best_total_loss:
            best_total_loss = valid_total_loss
            epochs_since_best = 0

            if epoch <= args.pretrain_time:
                path = f"{save_path}/model/state_dict_best_pretrain.pt"
            else:
                path = f"{save_path}/model/state_dict_best.pt"

            torch.save(model.module.state_dict(), path)
            printlog("######## Best Total Loss ########")

        if "pos_error" in valid_losses and (best_pos_error == 0 or valid_losses["pos_error"] < best_pos_error):
            best_pos_error = valid_losses["pos_error"]
            epochs_since_best = 0
            path = f"{save_path}/model/state_dict_best_pe.pt"
            torch.save(model.module.state_dict(), path)
            printlog("######## Best Pos Error #########")

        # Periodically save model
        if epoch % args.save_epoch == 0:
            path = f"{save_path}/model/state_dict_{epoch}.pt"
            torch.save(model.module.state_dict(), path)
            printlog("########## Saved Model ##########")

        # End of pretrain stage
        if epoch == args.pretrain_time:
            printlog("######### End Pretrain ##########")
            best_total_loss = 0
            epochs_since_best = 0
            lr = max(args.start_lr, args.min_lr)

            state_dict = torch.load(f"{save_path}/model/state_dict_best_pretrain.pt", weights_only=False)
            model.module.load_state_dict(state_dict)
            valid_losses = run_epoch(model, optimizer, train=False)
            printlog("Test:\t" + loss_str(valid_losses))

    printlog(f"Best Test Loss: {best_total_loss:.4f}")


import argparse
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from data_utils import load_raw_data, PromptIterableDataset, collator
from tqdm import tqdm
from torch.utils.data import DataLoader
import bmtrain as bmt
from functools import partial
import time
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def get_model_tokenizer(args):
    print("loading model...")
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
    if args.start_step != 0:
        print(f"loading model from step {args.start_step}")
        bmt.load(model, os.path.join(args.save_dir, f"{args.model}/step_{args.start_step}/checkpoint.pt"))

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    if args.start_step != 0:
        tokenizer = LlamaTokenizer.from_pretrained(os.path.join(args.save_dir, f"{args.model}/step_{args.start_step}"))
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    print("loaded")
    
    model = bmt.BMTrainModelWrapper(model)
    return model, tokenizer

def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(
        model.parameters(), weight_decay=args.weight_decay
    )
    if args.start_step != 0:
        file_name = os.path.join(os.path.join(args.save_dir, f"{args.model}/step_{args.start_step}"), "optim.rank-{}.opt".format(bmt.rank()))
        print(file_name)
        if os.path.exists(file_name):
            print("start to load grad ckpt {}".format(file_name))
            states = torch.load(file_name)
            optimizer.load_state_dict(states)
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    if args.lr_decay_style == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_ratio * args.train_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )
    elif args.lr_decay_style == "cosine":
        print("use cosine")
        lr_scheduler = bmt.lr_scheduler.Cosine(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_ratio * args.train_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )

    elif args.lr_decay_style == "noam":
        print("use noam")
        lr_scheduler = bmt.lr_scheduler.Noam(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_ratio * args.train_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )
    else:
        raise NotImplementedError
    if args.start_step != 0:
        print(f"loading scheduler from step {args.start_step}")
        lr_scheduler.load_state_dict(torch.load(os.path.join(args.save_dir, f"{args.model}/step_{args.start_step}/scheduler.pt")))
    return lr_scheduler


def setup_model_and_optimizer(args):
    model, tokenizer = get_model_tokenizer(args)
    bmt.synchronize()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    return tokenizer, model, optimizer, lr_scheduler



def train(args):

    bmt.init_distributed(
        seed=args.seed,
        zero_level=3,
    )

    if args.tensorboard is not None and bmt.rank() == 0:
        from torch.utils.tensorboard import SummaryWriter
        import distutils.version  # noqa: F401

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)

    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    optim_manager = bmt.optim.OptimManager(loss_scale=2**10)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    bmt.synchronize()
    original_dataset = load_raw_data(args.data_dir, max_sample=args.max_sample, random_state=0)

    bmt.print_rank("Model memory")
    bmt.print_rank(torch.cuda.memory_summary())

    avg_time_recorder = bmt.utils.AverageRecorder()
    avg_loss_recorder = bmt.utils.AverageRecorder()
    train_start_time = time.time()
    global_step = 0

    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    
    for epoch in range(args.epochs):
        indices = torch.randperm(len(original_dataset))
        dataset = [original_dataset[i] for i in indices]

        data_per_gpu = len(dataset) // bmt.world_size()
        dataset = dataset[bmt.rank() * data_per_gpu : (bmt.rank() + 1) * data_per_gpu]

        dataset = PromptIterableDataset(dataset, tokenizer = tokenizer, max_seq_length = args.max_seq_length, teacher_forcing=True, truncate_method="tail")
        dataloader = DataLoader(dataset, batch_size=args.batch_size_per_device, collate_fn=partial(collator, tokenizer))

        if global_step >= args.train_iters:
            break
        progress_bar = tqdm(range(len(dataloader)), disable=not bmt.rank()==0, desc=f"epoch {epoch}")

        for step, inputs in enumerate(dataloader):
            if global_step < args.start_step:
                global_step += 1
                continue

            st = time.time()

            with bmt.inspect.inspect_tensor() as inspector:
                for k in inputs:
                    inputs[k] = inputs[k].cuda()
                labels = inputs.pop("labels")
                logits = model(**inputs).logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, len(tokenizer))
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_func(shift_logits, shift_labels)
            
                global_loss = bmt.sum_loss(loss).item()

                optim_manager.backward(loss)


                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                    optim_manager.clip_grad_norm(optimizer.param_groups, max_norm=args.clip_grad)

                    optim_manager.step()
                    optim_manager.zero_grad()

            
            global_step += 1
            progress_bar.update(1)

            # record time and loss
            iteration_time = time.time() - st

            avg_time_recorder.record(iteration_time)
            avg_loss_recorder.record(global_loss)

            # print time and loss
            if global_step % args.logging_step == 0:
                bmt.print_rank(
                    "| Iter: {:6d} | loss: {:.4f} average_loss: {:.4f} | lr: {:.4e} | time: {:.4f} seconds | total_time_passed: {:.4f} minutes".format(
                        global_step,
                        global_loss,
                        avg_loss_recorder.value,
                        lr_scheduler.current_lr,
                        avg_time_recorder.value,
                        (time.time() - train_start_time) / 60
                    )
                )

                if args.tensorboard and bmt.rank() == 0:
                    writer.add_scalar("Loss/train", global_loss, global_step)
                    writer.add_scalar("average_Loss/train", avg_loss_recorder.value, global_step)
                    writer.add_scalar("lr/train", lr_scheduler.current_lr, global_step)


            # save model
            if global_step % args.save_step == 0:
                save_dir = os.path.join(args.save_dir, f"{args.model}/step_{global_step}")
                os.makedirs(save_dir, exist_ok=True)

                bmt.save(model, os.path.join(save_dir, "checkpoint.pt"))
                print("saving optimizer state", os.path.join(save_dir, "optim.rank-%d.opt" % bmt.rank()))
                torch.save(optimizer.state_dict(), os.path.join(save_dir, "optim.rank-%d.opt" % bmt.rank()))

                if bmt.rank() == 0:
                    torch.save(lr_scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
                    tokenizer.save_pretrained(save_dir)
                bmt.print_rank(f"model saved at {save_dir}")
            
            if global_step == args.train_iters:
                break

    if bmt.rank() == 0:
        bmt.save(model, os.path.join(args.save_dir, f"{args.model}/final.pt"))
        tokenizer.save_pretrained(os.path.join(args.save_dir, f"{args.model}"))
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model", type=str, default='ultralm-13b', help="a short name for this training")
    parser.add_argument("--model_name_or_path", default='/llama-13b', help="pretrained huggingface llama path")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--batch_size_per_device", default=2, type=int)
    parser.add_argument("--logging_step", default=100, type=int)
    parser.add_argument("--save_step", default=50000, type=int)
    parser.add_argument("--data_dir", default="../data/processed", type=str)

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--with_eval", action="store_true")

    parser.add_argument("--clip-grad", type=float, default=1.0, help="gradient clipping")
    # Learning rate.
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay rate")
    parser.add_argument("--loss-scale", type=float, default=65536, help="loss scale")

    parser.add_argument("--train-iters", type=int, default=2000000)

    parser.add_argument("--save_dir", type=str, default="/data/models")

    parser.add_argument("--max_sample", type=int, default=None, help="maximum training sample number")
    parser.add_argument("--debug", default=False, action="store_true")




    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--lr-decay-style",
        type=str,
        default="cosine",
        choices=["constant", "linear", "cosine", "exponential", "noam"],
        help="learning rate decay function",
    )
    parser.add_argument("--lr-decay-iters", type=int, default=None, help="lr decay steps")
    parser.add_argument(
        "--start-step", type=int, default=0, help="step to start or continue training"
    )
    parser.add_argument("--tensorboard", type=str, default=None, help="lr decay steps")


    args = parser.parse_args()

    train(args)

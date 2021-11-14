import logging
import time
import torch

from tools.tools import AverageMeter, accuracy, is_main_process


def train(
        train_loader,
        model,
        criterion,
        optimizer,
        epoch,
        print_freq,
        writer,
        args=None,):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    if args.distribute:
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda")

    for i, (input, target) in enumerate(train_loader):


        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True, device=device)

        if args.half:
            with torch.cuda.amp.autocast():
                output = model(input.cuda(device))
                loss = criterion(output, target)
        else:
            output = model(input.cuda(device))
            loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top5.update(prec5.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        if i % print_freq == 0 and is_main_process():
            logging.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.8f}\t'
                          'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(epoch,
                                                                            int(i),
                                                                            int(len(train_loader)),
                                                                            batch_time=batch_time,
                                                                            loss=losses,
                                                                            top1=top1,
                                                                            top5=top5,
                                                                            lr=optimizer.param_groups[-1]['lr'])))
    if args.distribute:
        losses.synchronize_between_processes()
        top1.synchronize_between_processes()
        top5.synchronize_between_processes()
    if is_main_process():
        writer.add_scalar('Train/loss', losses.avg, epoch)
        writer.add_scalar('Train/top1', top1.avg, epoch)
        writer.add_scalar('Train/lr', optimizer.param_groups[-1]['lr'], epoch)
        logging.info(
            ('Epoch {epoch} Training Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
             .format(epoch=epoch, top1=top1, top5=top5, loss=losses)))



def validate(val_loader, model, criterion, print_freq, epoch, writer, args=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    if args.distribute:
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda")
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True, device=device)

            # compute output
            output = model(input)
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            top5.update(prec5.item(), input.size(0))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 and is_main_process():
                logging.info(
                    ('Test: [{0}/{1}]\t'
                     'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         i,
                         len(val_loader),
                         batch_time=batch_time,
                         top1=top1,
                         top5=top5)))

    if args.distribute:
        losses.synchronize_between_processes()
        top1.synchronize_between_processes()
        top5.synchronize_between_processes()
    if is_main_process():
        writer.add_scalar('Test/loss', losses.avg, epoch)
        writer.add_scalar('Test/top1', top1.avg, epoch)
        logging.info(
            ('Epoch {epoch} Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
             .format(epoch=epoch, top1=top1, top5=top5, loss=losses)))

    return (top1.avg + top5.avg) / 2

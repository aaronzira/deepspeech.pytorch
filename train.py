import argparse
import errno
import json
import os
import time

import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

from data.data_loader import AudioDataLoader, SpectrogramDataset
from decoder import ArgMaxDecoder
from model import DeepSpeech, supported_rnns
from spell import correction

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--train_sample_manifest', metavar='DIR',
                    help='path to subset of train files manifest csv', default='data/train_sample_manifest.csv')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for prediction')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden_size', default=400, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=4, type=int, help='Number of RNN layers')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning_anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint_per_batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--final_model_path', default='models/deepspeech_final.pth.tar',
                    help='Location to save final model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--rnn_type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise_dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise_prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise_min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise_max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log_dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log_params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.set_defaults(cuda=False, silent=False, checkpoint=False, visdom=False, augment=False, tensorboard=False, log_params=False)

def to_np(x):
    return x.data.cpu().numpy()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    args = parser.parse_args()
    save_folder = args.save_folder

    loss_results, cer_results, wer_results = torch.Tensor(args.epochs),torch.Tensor(args.epochs),torch.Tensor(args.epochs)
    lm_cer_results, lm_wer_results = torch.Tensor(args.epochs),torch.Tensor(args.epochs)
    train_time_results, train_sample_cer_results, train_sample_wer_results = \
            torch.Tensor(args.epochs),torch.Tensor(args.epochs),torch.Tensor(args.epochs)
    val_loss_results, train_sample_lm_cer_results, train_sample_lm_wer_results = \
            torch.Tensor(args.epochs),torch.Tensor(args.epochs),torch.Tensor(args.epochs)
    if args.visdom:
        from visdom import Visdom
        viz = Visdom()

        opts = [dict(title='Loss', ylabel='Loss', xlabel='Epoch'),
                dict(title='Val WER', ylabel='WER', xlabel='Epoch'),
                dict(title='Val CER', ylabel='CER', xlabel='Epoch'),
                dict(title='Training Time', ylabel='Hours', xlabel='Epoch'),
                dict(title='Train (subset) WER', ylabel='WER', xlabel='Epoch'),
                dict(title='Train (subset) CER', ylabel='CER', xlabel='Epoch')]

        viz_windows = [None, None, None, None, None, None]
        loss_results, cer_results, wer_results, \
            train_time_results, train_sample_cer_results, train_sample_wer_results = \
                torch.Tensor(args.epochs),torch.Tensor(args.epochs),torch.Tensor(args.epochs), \
                torch.Tensor(args.epochs),torch.Tensor(args.epochs),torch.Tensor(args.epochs)
        epochs = torch.arange(1, args.epochs + 1)

    if args.tensorboard:
        from logger import TensorBoardLogger
        try:
            os.makedirs(args.log_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory already exists.')
                for file in os.listdir(args.log_dir):
                    file_path = os.path.join(args.log_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        raise
            else:
                raise
        logger = TensorBoardLogger(args.log_dir)

    try:
        os.makedirs(save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise
    criterion = CTCLoss()

    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=args.augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, augment=False)
    train_sample_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_sample_manifest, labels=labels,
                                      normalize=True, augment=False)
    train_loader = AudioDataLoader(train_dataset, batch_size=args.batch_size,
                                   num_workers=args.num_workers)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size/2,
                                  num_workers=args.num_workers)
    train_sample_loader = AudioDataLoader(train_sample_dataset, batch_size=args.batch_size/2,
                                  num_workers=args.num_workers)

    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
    model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                       nb_layers=args.hidden_layers,
                       labels=labels,
                       rnn_type=supported_rnns[rnn_type],
                       audio_conf=audio_conf,
                       bidirectional=True)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum, nesterov=True)
    decoder = ArgMaxDecoder(labels)

    if args.continue_from:
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from)
        model.load_state_dict(package['state_dict'])
        optimizer.load_state_dict(package['optim_dict'])
        start_epoch = int(package.get('epoch', None) or 1) - 1  # Python index start at 0 for training
        start_iter = package.get('iteration', None)
        if start_iter is None:
            start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
            start_iter = 0
        else:
            start_iter += 1
        avg_loss = int(package.get('avg_loss', 0))
        if args.visdom and \
                        package['loss_results'] is not None and start_epoch > 0:  # Add previous scores to visdom graph
            epoch = start_epoch
            loss_results, cer_results, wer_results, \
                training_time_results, train_sample_wer_results, train_sample_cer_results = \
                    package['loss_results'],package['cer_results'], \
                    package['wer_results'],package['training_time_results'], \
                    package['train_sample_wer_results'], package['train_sample_cer_results']
            x_axis = epochs[0:epoch]
            y_axis = [loss_results[0:epoch], wer_results[0:epoch], \
                        cer_results[0:epoch], training_time_results[0:epoch], \
                        train_sample_wer_results[0:epoch], train_sample_cer_results[0:epoch]]
            for x in range(len(viz_windows)):
                viz_windows[x] = viz.line(
                    X=x_axis,
                    Y=y_axis[x],
                    opts=opts[x],
                )
        if args.tensorboard and package['loss_results'] is not None and start_epoch > 0:  # Add previous scores to tensorboard logs
            epoch = start_epoch
            loss_results, cer_results, wer_results = package['loss_results'], package['cer_results'], package['wer_results']
            lm_cer_results, lm_wer_results = package['lm_cer_results'], package['lm_wer_results']
            val_loss_results, train_sample_cer_results, train_sample_wer_results = \
                    package['val_loss_results'], package['train_sample_cer_results'], package['train_sample_wer_results']
            train_time_results, train_sample_lm_cer_results, train_sample_lm_wer_results = \
                    package['train_time_results'], package['train_sample_lm_cer_results'], package['train_sample_lm_wer_results']

            for i in range(len(loss_results)):
                info = {
                    'Avg Train Loss': loss_results[i],
                    'Avg Val WER': wer_results[i],
                    'Avg Val CER': cer_results[i],
                    'Avg Val Loss': val_loss_results[i],
                    'Avg LM-Corrected Val WER': lm_wer_results[i],
                    'Avg LM-Corrected Val CER': lm_cer_results[i],
                    'Avg Train Time': train_time_results[i],
                    'Avg Train WER': train_sample_wer_results[i],
                    'Avg Train CER': train_sample_cer_results[i],
                    'Avg LM-Corrected Train WER': train_sample_lm_wer_results[i],
                    'Avg LM-Corrected Train CER': train_sample_lm_cer_results[i]
                }
                for tag, val in info.items():
                    logger.scalar_summary(tag, val, i+1)
    else:
        avg_loss = 0
        start_epoch = 0
        start_iter = 0
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    print(model)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    inf = float("inf")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        start = time.time()
        end = time.time()
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_loader):
                break
            inputs, targets, input_percentages, target_sizes = data
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = Variable(inputs, requires_grad=False)
            target_sizes = Variable(target_sizes, requires_grad=False)
            targets = Variable(targets, requires_grad=False)

            if args.cuda:
                inputs = inputs.cuda()

            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH

            seq_length = out.size(0)
            sizes = Variable(input_percentages.mul_(int(seq_length)).int(), requires_grad=False)

            loss = criterion(out, targets, sizes, target_sizes)
            loss = loss / inputs.size(0)  # average the loss by minibatch

            loss_sum = loss.data.sum()
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.data[0]

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
            # SGD step
            optimizer.step()

            if args.cuda:
                torch.cuda.synchronize()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            train_time = (start-end)/3600.
            if not args.silent:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (i + 1), len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0:
                file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth.tar' % (save_folder, epoch + 1, i + 1)
                print("Saving checkpoint model to %s" % file_path)
                torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                loss_results=loss_results,
                                                train_time_results=train_time_results,
                                                wer_results=wer_results, cer_results=cer_results,
                                                lm_wer_results=lm_wer_results, lm_cer_results=lm_cer_results,
                                                train_sample_wer_results=train_sample_wer_results,
                                                train_sample_cer_results=train_sample_cer_results,
                                                train_sample_lm_wer_results=train_sample_lm_wer_results,
                                                train_sample_lm_cer_results=train_sample_lm_cer_results,
                                                avg_loss=avg_loss),
                           file_path)
            del loss
            del out
        avg_loss /= len(train_loader)

        print('Training Summary Epoch: [{0:02d}]\t'
              'Average Loss {loss:.3f}\t'.format(
            epoch + 1, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        val_loss = 0
        total_cer, total_wer = 0, 0
        total_lm_cer, total_lm_wer = 0, 0
        model.eval()

        for i, (data) in enumerate(test_loader):  # test
            inputs, targets, input_percentages, target_sizes = data

            inputs = Variable(inputs, volatile=True)

            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            if args.cuda:
                inputs = inputs.cuda()

            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH
            seq_length = out.size(0)
            sizes = Variable(input_percentages.mul_(int(seq_length)).int(), volatile=True)

            # val loss
            targets = Variable(targets, requires_grad=False)
            target_sizes = Variable(target_sizes, requires_grad=False)
            loss = criterion(out, targets, sizes, target_sizes)
            loss = loss / inputs.size(0)  # average the loss by minibatch
            loss_sum = loss.data.sum()
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.data[0]
            val_loss += loss_value

            decoded_output = decoder.decode(out.data, sizes)
            corrected_output = [correction(output).upper() for output in decoded_output]
            target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
            wer, cer = 0, 0
            lm_wer, lm_cer = 0, 0
            for x in range(len(target_strings)):
                wer += decoder.wer(decoded_output[x], target_strings[x]) / float(len(target_strings[x].split()))
                cer += decoder.cer(decoded_output[x], target_strings[x]) / float(len(target_strings[x]))
                lm_wer += decoder.wer(corrected_output[x], target_strings[x]) / float(len(target_strings[x].split()))
                lm_cer += decoder.cer(corrected_output[x], target_strings[x]) / float(len(target_strings[x]))
            total_cer += cer
            total_wer += wer
            total_lm_cer += lm_cer
            total_lm_wer += lm_wer

            if args.cuda:
                torch.cuda.synchronize()
            del loss
            del out
        wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)
        lm_wer = total_lm_wer / len(test_loader.dataset)
        lm_cer = total_lm_cer / len(test_loader.dataset)
        wer *= 100
        cer *= 100
        lm_wer *= 100
        lm_cer *= 100
        avg_val_loss = val_loss / len(test_loader.dataset)

        print('Validation Summary Epoch: [{0:02d}]\t\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
            epoch + 1, wer=wer, cer=cer))
        print('LM-Corrected Validation Summary:\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
            wer=lm_wer, cer=lm_cer))

        # train sample to monitor WER, CER
        train_sample_total_cer, train_sample_total_wer = 0, 0
        train_sample_lm_total_cer, train_sample_lm_total_wer = 0, 0
        for i, (data) in enumerate(train_sample_loader):
            inputs, targets, input_percentages, target_sizes = data

            inputs = Variable(inputs)

            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            if args.cuda:
                inputs = inputs.cuda()

            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH
            seq_length = out.size(0)
            sizes = Variable(input_percentages.mul_(int(seq_length)).int())

            decoded_output = decoder.decode(out.data, sizes)
            corrected_output = [correction(output).upper() for output in decoded_output]
            target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
            train_sample_wer, train_sample_cer = 0, 0
            train_sample_lm_wer, train_sample_lm_cer = 0, 0
            for x in range(len(target_strings)):
                train_sample_wer += decoder.wer(decoded_output[x], target_strings[x]) / float(len(target_strings[x].split()))
                train_sample_cer += decoder.cer(decoded_output[x], target_strings[x]) / float(len(target_strings[x]))
                train_sample_lm_wer += decoder.wer(corrected_output[x], target_strings[x]) / float(len(target_strings[x].split()))
                train_sample_lm_cer += decoder.cer(corrected_output[x], target_strings[x]) / float(len(target_strings[x]))
            train_sample_total_cer += train_sample_cer
            train_sample_total_wer += train_sample_wer
            train_sample_lm_total_cer += train_sample_lm_cer
            train_sample_lm_total_wer += train_sample_lm_wer

        train_sample_wer = train_sample_total_wer / len(train_sample_loader.dataset)
        train_sample_cer = train_sample_total_cer / len(train_sample_loader.dataset)
        train_sample_lm_wer = train_sample_lm_total_wer / len(train_sample_loader.dataset)
        train_sample_lm_cer = train_sample_lm_total_cer / len(train_sample_loader.dataset)
        train_sample_wer *= 100
        train_sample_cer *= 100
        train_sample_lm_wer *= 100
        train_sample_lm_cer *= 100

        loss_results[epoch] = avg_loss
        wer_results[epoch] = wer
        cer_results[epoch] = cer
        val_loss_results[epoch] = val_loss
        lm_wer_results[epoch] = lm_wer
        lm_cer_results[epoch] = lm_cer
        train_time_results[epoch] = train_time
        train_sample_wer_results[epoch] = train_sample_wer
        train_sample_cer_results[epoch] = train_sample_cer
        train_sample_lm_wer_results[epoch] = train_sample_lm_wer
        train_sample_lm_cer_results[epoch] = train_sample_lm_cer

        print('Train Sample Summary Epoch: [{0:02d}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
            epoch + 1, wer=train_sample_wer, cer=train_sample_cer))
        print('LM-Corrected Train Sample Summary:\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
            wer=train_sample_lm_wer, cer=train_sample_lm_cer))


        if args.visdom:
            #epoch += 1
            x_axis = epochs[0:epoch]
            y_axis = [loss_results[0:epoch], wer_results[0:epoch], \
                        cer_results[0:epoch], training_time_results[0:epoch], \
                        train_sample_wer_results[0:epoch], train_sample_cer_results[0:epoch]]
            for x in range(len(viz_windows)):
                if viz_windows[x] is None:
                    viz_windows[x] = viz.line(
                        X=x_axis,
                        Y=y_axis[x],
                        opts=opts[x],
                    )
                else:
                    viz.line(
                        X=x_axis,
                        Y=y_axis[x],
                        win=viz_windows[x],
                        update='replace',
                    )
        if args.tensorboard:
            info = {
                'Avg Train Loss': avg_loss,
                'Avg WER': wer,
                'Avg CER': cer,
                'Avg Val Loss': val_loss,
                'Avg LM-Corrected Val WER': lm_wer,
                'Avg LM-Corrected Val CER': lm_cer,
                'Avg Train Time': train_time,
                'Avg Train WER': train_sample_wer,
                'Avg Train CER': train_sample_cer,
                'Avg LM-Corrected Train WER': train_sample_lm_wer,
                'Avg LM-Corrected Train CER': train_sample_lm_cer
            }
            for tag, val in info.items():
                logger.scalar_summary(tag, val, epoch+1)
            if args.log_params:
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), epoch+1)
                    if value.grad is not None: # Condition inserted because batch_norm RNN_0 weights.grad and bias.grad are None. Check why
                        logger.histo_summary(tag+'/grad', to_np(value.grad), epoch+1)
        if args.checkpoint:
            file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch + 1)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                                train_time_results=train_time_results,
                                                val_loss_results=val_loss_results,
                                                wer_results=wer_results, cer_results=cer_results,
                                                lm_wer_results=lm_wer_results, lm_cer_results=lm_cer_results,
                                                train_sample_wer_results=train_sample_wer_results,
                                                train_sample_cer_results=train_sample_cer_results,
                                                train_sample_lm_wer_results=train_sample_lm_wer_results,
                                                train_sample_lm_cer_results=train_sample_lm_cer_results),
                       file_path)
        # anneal lr
        optim_state = optimizer.state_dict()
        optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args.learning_anneal
        optimizer.load_state_dict(optim_state)
        print('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

        avg_loss = 0

    torch.save(DeepSpeech.serialize(model, optimizer=optimizer), args.final_model_path)


if __name__ == '__main__':
    main()

import sys
import numpy as np
import torch
from model import DeepSpeech

path = sys.argv[1]
print("Loading checkpoint model %s" % path)
package = torch.load(path)
np.save('wer.npy', np.asarray([val for val in package['wer_results']]))
np.save('cer.npy', np.asarray([val for val in package['cer_results']]))
np.save('train_wer.npy', np.asarray([val for val in package['train_sample_wer_results']]))
np.save('train_cer.npy', np.asarray([val for val in package['train_sample_cer_results']]))
np.save('loss.npy', np.asarray([val for val in package['loss_results']]))
np.save('train_time.npy', np.asarray([val for val in package['training_time_results']]))


import Levenshtein as Lev

from decoder import ArgMaxDecoder
from model_original import DeepSpeech



def get_wer(string1,string2,labels):
    
    decoder = ArgMaxDecoder(labels)
    s1 = string1.split()
    s2 = string2.split()
    vocab = set(s1 + s2)

    word2index = dict(zip(vocab,range(len(vocab))))
    
    w1 = [unichr(word2index[w]) for w in s1]
    w2 = [unichr(word2index[w]) for w in s2]

    print('vocab len: {}'.format(len(vocab)))

    return Lev.distance(''.join(w1),''.join(w2))/float(len(w2))


if __name__ == '__main__':    
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument('--labels',default="_'abcdefghijklmnopqrstuvwxyz ")
    parser.add_argument('--prediction',default=None,help='File of single string of predicted transcript')
    parser.add_argument('--truth',default=None,help='File of single string of ground truth')
    parser.add_argument('--ctm',action='store_true',help='Prediction is a ctm rather than text file')
    args = parser.parse_args()

    with open(args.prediction) as f:
        predicted = f.read().strip()
        if args.ctm:
            predicted = eval(predicted)
            predicted = ' '.join([word['word'] for word in predicted])
    with open(args.truth) as f:
        truth = f.read()
        truth = re.sub('\d+:\d+:\d+\.\d+',' ',truth)
        truth = re.sub('S\d+:',' ',truth)
        truth = re.sub(',|_|"|\.|\?|-|\[|\]',' ',truth)
        truth = truth.lower().strip()
    
    print(get_wer(predicted,truth,args.labels))


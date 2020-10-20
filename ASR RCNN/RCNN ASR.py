
#
import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np

def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer

# defines the transformations done to the training set audio
train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(   sample_rate=16000, n_mels=128,
                                            n_fft=1024, hop_length=256  ),
    torchaudio.transforms.FrequencyMasking( freq_mask_param=30          ),
    torchaudio.transforms.TimeMasking(      time_mask_param=100         ))

# initializes dictionaries for converting to our labels
token2Idx = dict()
token2Idx['<BLANK>'] = 0
idx2Token = dict()

# loads our phenomes into the cache
def loadPhenomes(filePath):

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Param:
        filePath    - the directory of the file that contains all our phenomes
    Return:
        phenomeDict - the dictionary of our pheonmes indexed by the word the
                      phenomes apply to
    Notes:
    loads all our phenomes into a dictionary indexed by the word the phenomes
    apply to
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    # initilializes a dictionary that will store our phenomes
    phenomeDict = dict()

    # opens the file that stores the phenomes
    infile = open(filePath)

    # for each line in the phenome file
    for line in infile:

        # takes of the '\n' character
        line = line.strip()

        # converts the line to be all lowercase
        line = line.lower()

        # splits the line over spaces
        line = line.split()

        # the first item is the word, the rest are the phenomes
        phenomeDict[line[0]] = line[1:]

    # deallocates space for the file
    infile.close()

    # returns the dict we loaded in
    return phenomeDict

# defines a function to unpack and transform our data
def data_processing(data, phenomeDict, data_type="train"):

    # defines an empty list to store our spectrograms
    spectrograms = []
    # defines an empty list to store our labels
    labels = []

    # defines an empty list to store the lengths of our inputs
    input_lengths = []
    # defines an empty list to store the lengths of our labels
    label_lengths = []

    # unpacks our waveform and utterance (input and targets) for each datum
    for (waveform, _, utterance, _, _, _) in data:

        # if the data is classified as a training data set
        if data_type == 'train':
            # performs the training transformation on the data
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)

        # appends the transformed spectrogram
        spectrograms.append(spec)

        # converts to all lowercase and splits over spaces
        utterance = utterance.lower()
        wordList = utterance.split()

        targets = []
        lbl = []
        for word in wordList:
            if word not in phenomeDict:
                phenomeDict[word] = word
            targets.append(phenomeDict[word])

        for target in targets:
            for t in target:
                if t not in token2Idx:
                    token2Idx[t] = len(token2Idx)
                lbl.append(token2Idx[t])

        label = torch.Tensor(lbl)

        # performs the transformation on the labels
        # label = torch.Tensor(text_transform.text_to_int(utterance.lower()))

        # appends the transformed labels
        labels.append(label)

        # appends the lengths of our inputs and labels
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    for t in token2Idx:
        idx2Token[token2Idx[t]] = t

    # pads our inputs and targets so we can batch
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    # returns our inputs, targets, and the lengths of each
    return spectrograms, labels, input_lengths, label_lengths

# chooses the most likely target for a given prediction
def GreedyDecoder(output, labels, label_lengths, blank_label=0, collapse_repeated=True):

    # gets the highest predicted guess for all of them
    arg_maxes = torch.argmax(output, dim=2)
    # loops through each arg max
    for i in range(len(arg_maxes)):
        for j in range(len(arg_maxes[i])):
            # if we predict the blank label but it is less than our threshold for the guess
            # our threshold for the guess is 0.75
            if arg_maxes[i][j] == blank_label and output[i][j][blank_label] <= np.log(0.99):
                # we choose the 2nd most likely guess
                arg_maxes[i][j] = torch.argmax(output[i][j][1:])

    # initializes the decodes and targets as empty lists
    decodes = []
    targets = []

    # loops through each arg max
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append([ idx2Token[l] for l in labels[i][:label_lengths[i]].tolist()])
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append([idx2Token[d] for d in decode])
    return decodes, targets

class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 16, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 16
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(16, 16, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*16, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model, device, train_loader, criterion, optimizer, epoch, iter_meter):
    model.train()

    data_len = len(train_loader.dataset)
    cost = 0.0
    for batch_idx, _data in enumerate(train_loader):

        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        cost += loss.item()
        loss.backward()

        optimizer.step()
        # scheduler.step()
        iter_meter.step()


    print('Train Epoch: {}  Loss: {:.6f}'.format(epoch, cost / data_len))



def test(model, device, test_loader, criterion, epoch, iter_meter):
    print('evaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for _data in test_loader:
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() #/ len(test_loader)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            # for j in range(len(decoded_preds)):
            #     test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            #     test_wer.append(wer(decoded_targets[j], decoded_preds[j]))



    # avg_cer = sum(test_cer)/len(test_cer)
    # avg_wer = sum(test_wer)/len(test_wer)

    print('Test set: Average loss: {:.4f}'.format(test_loss))


def main(token2Idx, learning_rate=5e-4, batch_size=20, epochs=10,
        train_url="train-clean-100", test_url="test-clean"):

    hparams = {
        "n_cnn_layers": 1,
        "n_rnn_layers": 1,
        "rnn_dim": 128,
        "n_class": 80, # the length of our fully loaded token2Idx dictionary
        "n_feats": 128,
        "stride":2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    train_dataset = torchaudio.datasets.LIBRISPEECH("./", url=train_url, download=True)

    phenomeDict = loadPhenomes('./phenomes.txt')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, phenomeDict, 'train'),
                                **kwargs)

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)

    # print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=0).to(device)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
    #                                         steps_per_epoch=int(len(train_loader)),
    #                                         epochs=hparams['epochs'],
    #                                         anneal_strategy='linear')

    iter_meter = IterMeter()
    model.load_state_dict(torch.load('./newClassifier65'))

    # for i in train_loader:
    #     pass

    # test(model, device, train_loader, criterion, 6, iter_meter)
    for epoch in range(66, epochs + 1):
        train(model, device, train_loader, criterion, optimizer, epoch, iter_meter)
        # test(model, device, train_loader, criterion, epoch, iter_meter)
        torch.save(model.state_dict(), './newClassifier'+str(epoch))

learning_rate = 1e-4
batch_size = 1
epochs = 200
libri_train_set = "dev-clean"
libri_test_set = "test-clean"

main(token2Idx, learning_rate, batch_size, epochs, libri_test_set, libri_test_set)
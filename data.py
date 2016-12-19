from konlpy.tag import Twitter

pos_tagger = Twitter()

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def read_raw_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        print('loading data')
        data = [line.split('\t') for line in f.read().splitlines()]

        print('pos tagging to token')
        data = [(tokenize(row[1]), int(row[2])) for row in data[1:]]
    return data

def build_vocab(tokens):
    print('building vocabulary')
    vocab = dict()
    vocab['#UNKOWN'] = 0
    vocab['#PAD'] = 1
    for t in tokens:
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab

def get_token_id(token, vocab):
    if token in vocab:
        return vocab[token]
    else:
        0 # unkown

def build_input(data, vocab):

    def get_onehot(index, size):
        onehot = [0] * size
        onehot[index] = 1
        return onehot

    print('building input')
    result = []
    for d in data:
        sequence = [get_token_id(t, vocab) for t in d[0]]
        while len(sequence) > 0:
            seq_seg = sequence[:60]
            sequence = sequence[60:]

            padding = [1] *(60 - len(seq_seg))
            seq_seg = seq_seg + padding

            result.append((seq_seg, get_onehot(d[1], 2)))

    return result 

def save_data(filename, data):
    def make_csv_str(d):
        output = '%d' % d[0]
        for index in d[1:]:
            output = '%s,%d' % (output, index)
        return output

    with open(filename, 'w', encoding='utf-8') as f:
        for d in data:
            data_str = make_csv_str(d[0])
            label_str = make_csv_str(d[1])
            f.write (data_str + '\n')
            f.write (label_str + '\n')

def save_vocab(filename, vocab):
    with open(filename, 'w', encoding='utf-8') as f:
        for v in vocab:
            f.write('%s\t%d\n' % (v, vocab[v]))
            
def load_data(filename):
    result = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(int(len(lines)/2)):
            data = lines[i*2]
            label = lines[i*2 + 1]

            result.append(([int(s) for s in data.split(',')], [int(s) for s in label.split(',')]))
    return result

def load_vocab(filename):
    result = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            ls = line.split('\t')
            result[ls[0]] = int(ls[1])

    return result


if __name__ == '__main__':
    data = read_raw_data('ratings_train.txt')
    tokens = [t for d in data for t in d[0]]
    vocab = build_vocab(tokens)
    d = build_input(data, vocab)
    
    save_data('test_data.txt', d)
    save_vocab('test_vocab.txt', vocab)

    d2 = load_data('test_data.txt')
    vocab2 = load_vocab('test_vocab.txt')

    assert(len(d2) == len(d))
    for i in range(len(d)):
        assert(len(d2[i]) ==  len(d[i]))
        for j in range(len(d[i])):
            assert(d2[i][j] == d[i][j])

    for index in vocab:
        assert(vocab2[index] == vocab[index])
    
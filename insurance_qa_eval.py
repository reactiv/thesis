from __future__ import print_function

import os

import sys
import random
from time import strftime, gmtime, time

import pickle
import json

import thread
from scipy.stats import rankdata
import keras.optimizers

#random.seed(42)


def log(x):
    print(x)


class Evaluator:
    def __init__(self, conf, model, optimizer=None):
        try:
            #data_path = os.environ['INSURANCE_QA']
            data_path = 'limit'
        except KeyError:
            print("INSURANCE_QA is not set. Set it to your clone of https://github.com/codekansas/insurance_qa_python")
            sys.exit(1)
        if isinstance(conf, str):
            conf = json.load(open(conf, 'rb'))
        self.model = model(conf)
        self.path = data_path
        self.conf = conf
        self.params = conf['training']
        optimizer = self.params['optimizer'] if optimizer is None else optimizer
        self.model.compile(optimizer)
        self.answers = self.load('answers') # self.load('generated')
        self.generated_answers = self.load('generated_answers')
        self._vocab = None
        self._reverse_vocab = None
        self._eval_sets = None

    ##### Resources #####

    def load(self, name):
        return pickle.load(open(os.path.join(self.path, name), 'rb'))

    def vocab(self):
        if self._vocab is None:
            self._vocab = self.load('vocabulary')
        return self._vocab

    def reverse_vocab(self):
        if self._reverse_vocab is None:
            vocab = self.vocab()
            self._reverse_vocab = dict((v.lower(), k) for k, v in vocab.items())
        return self._reverse_vocab

    ##### Loading / saving #####

    def save_epoch(self, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        self.model.save_weights('models/weights_epoch_%d.h5' % epoch, overwrite=True)

    def load_epoch(self, epoch):
        assert os.path.exists('models/weights_epoch_%d.h5' % epoch), 'Weights at epoch %d not found' % epoch
        self.model.load_weights('models/weights_epoch_%d.h5' % epoch)

    ##### Converting / reverting #####

    def convert(self, words):
        rvocab = self.reverse_vocab()
        if type(words) == str:
            words = words.strip().lower().split(' ')
        return [rvocab.get(w, 0) for w in words]

    def revert(self, indices):
        vocab = self.vocab()
        return [vocab.get(i, 'X') for i in indices]

    ##### Padding #####

    def padq(self, data):
        return self.pad(data, self.conf.get('question_len', None))

    def pada(self, data):
        return self.pad(data, self.conf.get('answer_len', None))

    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    ##### Training #####

    def get_time(self):
        return strftime('%Y-%m-%d %H:%M:%S', gmtime())

    def train(self):
        batch_size = self.params['batch_size']
        nb_epoch = self.params['nb_epoch']
        validation_split = self.params['validation_split']

        training_set = self.load('train')
        # top_50 = self.load('top_50')

        questions = list()
        good_answers = list()
        indices = list()

        for j, q in enumerate(training_set):
            questions += [q['question']] * len(q['answers'])
            good_answers += [self.real_answers[i] for i in q['answers']]
            indices += [j] * len(q['answers'])
        log('Began training at %s on %d samples' % (self.get_time(), len(questions)))

        questions = self.padq(questions)
        good_answers = self.pada(good_answers)

        val_loss = {'loss': 1., 'epoch': 0}

        # def get_bad_samples(indices, top_50):
        #     return [self.answers[random.choice(top_50[i])] for i in indices]

        for i in range(1, nb_epoch+1):
            # sample from all answers to get bad answers
            # if i % 2 == 0:
            #     bad_answers = self.pada(random.sample(self.answers.values(), len(good_answers)))
            # else:
            #     bad_answers = self.pada(get_bad_samples(indices, top_50))
            bad_answers = self.pada(random.sample(self.real_answers.values(), len(good_answers)))

            print('Fitting epoch %d' % i, file=sys.stderr)
            hist = self.model.fit([questions, good_answers, bad_answers], nb_epoch=1, batch_size=batch_size,
                             validation_split=validation_split, verbose=1)

            if hist.history['val_loss'][0] < val_loss['loss']:
                val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
            log('%s -- Epoch %d ' % (self.get_time(), i) +
                'Loss = %.4f, Validation Loss = %.4f ' % (hist.history['loss'][0], hist.history['val_loss'][0]) +
                '(Best: Loss = %.4f, Epoch = %d)' % (val_loss['loss'], val_loss['epoch']))
            evaluator.get_score(verbose=False)
            self.save_epoch(i)

        return val_loss

    ##### Evaluation #####

    def prog_bar(self, so_far, total, n_bars=20):
        n_complete = int(so_far * n_bars / total)
        if n_complete >= n_bars - 1:
            print('\r[' + '=' * n_bars + ']', end='', file=sys.stderr)
        else:
            s = '\r[' + '=' * (n_complete - 1) + '>' + '.' * (n_bars - n_complete) + ']'
            print(s, end='', file=sys.stderr)

    def eval_sets(self):
        if self._eval_sets is None:
            self._eval_sets = dict([(s, self.load(s)) for s in ['dev']])
        return self._eval_sets

    def get_score(self, verbose=False):
        for name, data in self.eval_sets().items():
            print('----- %s -----' % name)

            random.shuffle(data)

            if 'n_eval' in self.params:
                data = data[:self.params['n_eval']]

            c_1, c_2 = 0, 0
            counts = [0,0,0,0]
            res_dict = {}
            for i, d in enumerate(data):
                # self.prog_bar(i, len(data))

                indices = d['bad'] + d['good']
                answers = self.pada([self.answers[i] for i in indices])
                question = self.padq([d['question']] * len(indices))

                sims = self.model.predict([question, answers])
                #print(sims)
                n_good = len(d['good'])
                max_r = np.argmax(sims)
                max_n = np.argmax(sims[:n_good])
                counts[max_r] += 1
                r = rankdata(sims, method='max')

                if verbose:
                    min_r = np.argmin(sims)
                    amin_r = self.answers[indices[min_r]]
                    amax_r = self.answers[indices[max_r]]
                    amax_n = self.answers[indices[max_n]]

                    print(' '.join(self.revert(d['question'])))
                    print('Predicted: ({}) '.format(sims[max_r]) + ' '.join(self.revert(amax_r)))
                    print('Expected: ({}) Rank = {} '.format(sims[max_n], r[max_n]) + ' '.join(self.revert(amax_n)))
                    print('Worst: ({})'.format(sims[min_r]) + ' '.join(self.revert(amin_r)))

                res_dict[i] = max_r == 3
                c_1 += 1 if max_r == 3 else 0
                c_2 += 1 / float(r[max_r] - r[max_n] + 1)
            
            top1 = c_1 / float(len(data))
            mrr = c_2 / float(len(data))
            print(counts)
            print(len(data))
            if name == 'dev':
                test_perf = top1
                test_mrr = mrr
                res = res_dict

            del data
            print('Top-1 Precision: %f' % top1)
            print('MRR: %f' % mrr)
        return test_perf, test_mrr, res


    def get_epoch(self, real, gen, ratio, size):
        q_list = []
        a_list = []
        sample = np.random.rand(size) > ratio
        gen_ind = np.random.permutation(len(gen))

        print(gen_ind[0])
        print(sample[0])
        for i in range(size):
            if sample[i]:
                q = np.random.choice(real)
                ans = self.answers
                # print('beep')
            else:
                q = gen[gen_ind[i]]
                ans = self.generated_answers
                # print('FRIP')

            q_list += [q['question']] * len(q['answers'])
            a_list += [ans[i] for i in q['answers']]

        return self.padq(q_list), self.pada(a_list)


    def train_gen(self, gen_ratio):
        print(gen_ratio)
        save_every = self.params.get('save_every', None)
        batch_size = self.params.get('batch_size', 128)
        nb_epoch = self.params.get('nb_epoch', 10)
        split = self.params.get('validation_split', 0)

        # gen_ratio = self.params.get('gen_ratio', 0.9)
        # print(gen_ratio)
        epoch_size = self.params.get('epoch_size', 1000)
        res_list = []

        training_set = self.load('train')
        generated_set = self.load('gen')
        validation_set = self.load('validation')
        valid_q_list = []
        valid_a_list = []
        for q in validation_set:
            valid_q_list += [q['question']] * len(q['answers'])
            valid_a_list += [self.answers[i] for i in q['answers']]
        valid_qs = self.padq(valid_q_list)
        valid_as = self.pada(valid_a_list)


        val_loss = {'loss': 1., 'epoch': 0}

        for i in range(1, nb_epoch):
            questions, good_answers = self.get_epoch(training_set, generated_set, gen_ratio, epoch_size)
            # sample from all answers to get bad answers
            bad_answers = self.pada(random.sample(self.answers.values() + self.generated_answers.values(), len(good_answers)))
            valid_bad_answers = self.pada(random.sample(self.answers.values(), len(valid_qs)))
            print(len(questions))
            print('Epoch %d :: ' % i, end='')
            
            hist = self.model.fit([questions, good_answers, bad_answers], nb_epoch=1, batch_size=batch_size,
                             validation_data=([valid_qs, valid_as, valid_bad_answers],np.zeros(shape=(valid_qs.shape[0],))), verbose=False)

            if hist.history['val_loss'][0] < val_loss['loss']:
                val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
            print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))
            self.save_epoch(i)
            top1, mrr, _ = evaluator.get_score(verbose=False)
            res_list.append((hist.history['loss'], hist.history['val_loss'], top1, mrr))

        return val_loss, res_list


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'serve':
        from flask import Flask
        app = Flask(__name__)
        port = 5000
        lines = list()
        def log(x):
            lines.append(x)

        @app.route('/')
        def home():
            return ('<html><body><h1>Training Log</h1>' +
                    ''.join(['<code>{}</code><br/>'.format(line) for line in lines]) +
                    '</body></html>')

        def start_server():
            app.run(debug=False, use_evalex=False, port=port)

        thread.start_new_thread(start_server, tuple())
        print('Serving to port %d' % port, file=sys.stderr)

    import numpy as np

    conf = {
        'n_words': 6442,
        'question_len': 50,
        'answer_len': 50,
        'margin': 0.05,
        'gen_ratio': 1,
        'epoch_size': 1000,
        #'initial_embed_weights': 'word2vec_100_dim.embeddings',

        'training': {
            'batch_size': 1,
            'nb_epoch': 75,
            'validation_split': 0.1,
        },

        'similarity': {
            'mode': 'gesd',
            'gamma': 1,
            'c': 1,
            'd': 2,
            'dropout': 0.1,
        }
    }

    from keras_models import EmbeddingModel, ConvolutionModel


    for drop in [0, 0.5]:
        for opt in ['adam', 'sgd']:
            for model in [EmbeddingModel]:
                res = {}
                for i in range(0,5):

                    conf['similarity']['dropout'] = drop
                    optimiser = keras.optimizers.get(opt)
                    optimiser.clipnorm = 0.1
                    evaluator = Evaluator(conf, model=model, optimizer=optimiser)

                    # train the model
                    evaluator.get_score(verbose=False)
                    best_loss, hist = evaluator.train_gen(i/10.0)

                    # evaluate mrr for a particular epoch
                    evaluator.load_epoch(best_loss['epoch'])
                    res_tup = evaluator.get_score(verbose=False)
                    res[i] = hist

                    pickle.dump(res, open('{}-{}-{}.p'.format(drop, opt, str(model)).replace('.','p'), 'wb'))


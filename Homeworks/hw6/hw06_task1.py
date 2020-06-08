#!/usr/bin/env python

# text_classification_with_gru.py

"""
This script is an attempt at solving the sentiment classification problem
with an RNN that uses a GRU to get around the problem of vanishing gradients
that are common to neural networks with feedback.
"""
import sys
import os
import torch
import numpy
import random
sys.path.insert(1, r'/content/drive/My Drive/Colab Notebooks/ece695_Deep_Learning/DLStudio-1.1.3/DLStudio')
from DLStudio import *


seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmarks = False
os.environ['PYTHONHASHSEED'] = str(seed)


# watch -d -n 0.5 nvidia-smi


class TextClassification(nn.Module):
    """
    The purpose of this inner class is to be able to use the DLStudio module for simple 
    experiments in text classification.  Consider, for example, the problem of automatic 
    classification of variable-length user feedback: you want to create a neural network
    that can label an uploaded product review of arbitrary length as positive or negative.  
    One way to solve this problem is with a recurrent neural network in which you use a 
    hidden state for characterizing a variable-length product review with a fixed-length 
    state vector.  This inner class allows you to carry out such experiments.
    """

    def __init__(self, dl_studio, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
        super(TextClassification, self).__init__()
        self.dl_studio = dl_studio
        self.dataserver_train = dataserver_train
        self.dataserver_test = dataserver_test

    class SentimentAnalysisDataset(torch.utils.data.Dataset):
        """
        The sentiment analysis datasets that I have made available were extracted from
        an archive of user feedback comments as made available by Amazon for the year
        2007.  The original archive contains user feedback on 25 product categories. 
        For each product category, there are two files named 'positive.reviews' and
        'negative.reviews', with each file containing 1000 reviews. I believe that
        characterizing the reviews as 'positive' or 'negative' was carried out by 
        human annotators. Regardless, the reviews in these two files can be used to 
        train a neural network whose purpose would be to automatically characterize
        a product as being positive or negative. 

        I have extracted the following datasets extracted from the Amazon archive:

                sentiment_dataset_train_200.tar.gz        vocab_size = 43,285
                sentiment_dataset_test_200.tar.gz  

                sentiment_dataset_train_40.tar.gz         vocab_size = 17,001
                sentiment_dataset_test_40.tar.gz    

                sentiment_dataset_train_3.tar.gz          vocab_size = 3,402
                sentiment_dataset_test_3.tar.gz    

        The integer in the name of each dataset is the number of reviews collected 
        from the 'positive.reviews' and the 'negative.reviews' files for each product
        category.  Therefore, the dataset with 200 in its name has a total of 400 
        reviews for each product category.

        As to why I am presenting these three different datasets, note that, as shown
        above, the size of the vocabulary depends on the number of reviews selected
        and the size of the vocabulary has a strong bearing on how long it takes to 
        train an algorithm for text classification. For one simple reason for that: 
        the size of the one-hot representation for the words equals the size of the 
        vocabulary.  Therefore, the one-hot representation for the words for the 
        dataset with 200 in its name will be a one-axis tensor of size 43,285.

        For a purely feedforward network, it is not a big deal for the input tensors
        to be size Nx43285 where N is the number of words in a review.  And even for
        RNNs with simple feedback, that does not slow things down.  However, when 
        using GRUs, it's an entirely different matter if you are tying to run your
        experiments on, say, a laptop with a Quadro GPU.  Hence the reason for providing
        the datasets with 200 and 40 reviews.  The dataset with just 3 reviews is for
        debugging your code.
        """

        def __init__(self, dl_studio, train_or_test, dataset_file):
            super(TextClassification.SentimentAnalysisDataset, self).__init__()
            self.train_or_test = train_or_test
            root_dir = dl_studio.dataroot
            f = gzip.open(root_dir + dataset_file, 'rb')
            dataset = f.read()
            if train_or_test is 'train':
                if sys.version_info[0] == 3:
                    self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset, encoding='latin1')
                else:
                    self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset)
                self.categories = sorted(list(self.positive_reviews_train.keys()))
                self.category_sizes_train_pos = {category: len(self.positive_reviews_train[category]) for category in self.categories}
                self.category_sizes_train_neg = {category: len(self.negative_reviews_train[category]) for category in self.categories}
                self.indexed_dataset_train = []
                for category in self.positive_reviews_train:
                    for review in self.positive_reviews_train[category]:
                        self.indexed_dataset_train.append([review, category, 1])
                for category in self.negative_reviews_train:
                    for review in self.negative_reviews_train[category]:
                        self.indexed_dataset_train.append([review, category, 0])
                random.shuffle(self.indexed_dataset_train)
            elif train_or_test is 'test':
                if sys.version_info[0] == 3:
                    self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset, encoding='latin1')
                else:
                    self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset)
                self.vocab = sorted(self.vocab)
                self.categories = sorted(list(self.positive_reviews_test.keys()))
                self.category_sizes_test_pos = {category: len(self.positive_reviews_test[category]) for category in self.categories}
                self.category_sizes_test_neg = {category: len(self.negative_reviews_test[category]) for category in self.categories}
                self.indexed_dataset_test = []
                for category in self.positive_reviews_test:
                    for review in self.positive_reviews_test[category]:
                        self.indexed_dataset_test.append([review, category, 1])
                for category in self.negative_reviews_test:
                    for review in self.negative_reviews_test[category]:
                        self.indexed_dataset_test.append([review, category, 0])
                random.shuffle(self.indexed_dataset_test)

        def get_vocab_size(self):
            return len(self.vocab)

        def one_hotvec_for_word(self, word):
            word_index = self.vocab.index(word)
            return word_index
            # hotvec = torch.zeros(1, len(self.vocab))
            # hotvec[0, word_index] = 1
            # return hotvec

        def review_to_tensor(self, review):
            review_tensor = torch.zeros(len(review), 1)  # changed len(self.vocab) to 1
            for i, word in enumerate(review):
                review_tensor[i] = self.one_hotvec_for_word(word)  # changed review_tensor[i,:] to [i]
            return review_tensor

        def sentiment_to_tensor(self, sentiment):
            """
            Sentiment is ordinarily just a binary valued thing.  It is 0 for negative
            sentiment and 1 for positive sentiment.  We need to pack this value in a
            two-element tensor.
            """
            sentiment_tensor = torch.zeros(2)
            if sentiment is 1:
                sentiment_tensor[1] = 1
            elif sentiment is 0:
                sentiment_tensor[0] = 1
            sentiment_tensor = sentiment_tensor.type(torch.long)
            return sentiment_tensor

        def __len__(self):
            if self.train_or_test is 'train':
                return len(self.indexed_dataset_train)
            elif self.train_or_test is 'test':
                return len(self.indexed_dataset_test)

        def __getitem__(self, idx):
            sample = self.indexed_dataset_train[idx] if self.train_or_test is 'train' else self.indexed_dataset_test[idx]
            review = sample[0]
            review_category = sample[1]
            review_sentiment = sample[2]
            review_sentiment = self.sentiment_to_tensor(review_sentiment)
            review_tensor = self.review_to_tensor(review)
            category_index = self.categories.index(review_category)
            sample = {'review': review_tensor,
                      'category': category_index,  # should be converted to tensor, but not yet used
                      'sentiment': review_sentiment}
            return sample

    def load_SentimentAnalysisDataset(self, dataserver_train, dataserver_test):
        self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                                                            batch_size=self.dl_studio.batch_size, shuffle=True, num_workers=1)
        self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                                                           batch_size=self.dl_studio.batch_size, shuffle=False, num_workers=1)

    class TEXTnet(nn.Module):
        """
        TEXTnet stands for "Text Classification Network".
        This network is meant for semantic classification of variable length sentiment 
        data.  Based on my limited testing, the performance of this network is rather
        poor because it has no protection against vanishing gradients when used in an
        RNN.
        Location: Inner class TextClassification
        """

        def __init__(self, input_size, hidden_size, output_size):
            super(TextClassification.TEXTnet, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.combined_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
            self.combined_to_middle = nn.Linear(input_size + hidden_size, 100)
            self.middle_to_out = nn.Linear(100, output_size)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout(p=0.1)

        def forward(self, input, hidden):
            combined = torch.cat((input, hidden), 1)
            hidden = self.combined_to_hidden(combined)
            out = self.combined_to_middle(combined)
            out = torch.nn.functional.relu(out)
            out = self.dropout(out)
            out = self.middle_to_out(out)
            out = self.logsoftmax(out)
            return out, hidden

    class TEXTnetOrder2(nn.Module):
        """
        This is a variant of the TEXTnet because it uses a "second-order" model for
        the text.  By second-order I mean that, at each instant, it also uses the
        value of the hidden state at the previous instant.  The previous value of 
        the hidden state is stored away in a cell named "self.cell".  Note that what
        is stored there is fed into a linear layer and then subject to the tanh 
        activation.

        Although it is tempting to think of this a poor man's implementation of what
        is achieved by the famous GRU and LSTM mechanisms, but nothing could be farther
        from the truth.  Based on my experiments, the network shown below is no 
        better at learning than the vanilla TEXTnet.

        Location: Inner class TextClassification
        """

        def __init__(self, input_size, hidden_size, output_size, dls):
            super(TextClassification.TEXTnetOrder2, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.combined_to_hidden = nn.Linear(input_size + 2*hidden_size, hidden_size)
            self.combined_to_middle = nn.Linear(input_size + 2*hidden_size, 100)
            self.middle_to_out = nn.Linear(100, output_size)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout(p=0.1)
            # for the cell
            self.cell = torch.zeros(1, hidden_size).to(dls.device)
            self.linear_for_cell = nn.Linear(hidden_size, hidden_size)

        def forward(self, input, hidden):
            combined = torch.cat((input, hidden, self.cell), 1)
            hidden = self.combined_to_hidden(combined)
            out = self.combined_to_middle(combined)
            out = torch.nn.functional.relu(out)
            out = self.dropout(out)
            out = self.middle_to_out(out)
            out = self.logsoftmax(out)
            hidden_clone = hidden.clone()
            self.cell = torch.tanh(self.linear_for_cell(hidden_clone))
            return out, hidden

    class GRUnet(nn.Module):
        """
        Source: https://blog.floydhub.com/gru-with-pytorch/
        with the only modification that the final output of forward() is now
        routed through LogSoftmax activation. 
        """

        def __init__(self, input_size, hidden_size, output_size, n_layers, drop_prob=0.2):
            super(TextClassification.GRUnet, self).__init__()
            self.hidden_size = hidden_size
            self.n_layers = n_layers
            self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=drop_prob)
            self.fc = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
            self.logsoftmax = nn.LogSoftmax(dim=1)

        def forward(self, x, h):
            out, h = self.gru(x, h)
            out = self.fc(self.relu(out[:, -1]))
            out = self.logsoftmax(out)
            return out, h

        def init_hidden(self, batch_size):
            weight = next(self.parameters()).data
            hidden = weight.new(self.n_layers, batch_size, self.hidden_size).zero_()
            return hidden

    def save_model(self, model):
        "Save the trained model to a disk file"
        torch.save(model.state_dict(), self.dl_studio.path_saved_model)

    def run_code_for_training_for_text_classification_no_gru(self, net, hidden_size):
        filename_for_out = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
        FILE = open(filename_for_out, 'w')
        net = copy.deepcopy(net)
        net = net.to(self.dl_studio.device)
        # Note that the TEXTnet and TEXTnetOrder2 both produce LogSoftmax output:
        criterion = nn.NLLLoss()
#            criterion = nn.MSELoss()
#            criterion = nn.CrossEntropyLoss()
        accum_times = []
        optimizer = optim.SGD(net.parameters(),
                              lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
        start_time = time.clock()
        for epoch in range(self.dl_studio.epochs):
            print("")
            running_loss = 0.0
            for i, data in enumerate(self.train_dataloader):
                hidden = torch.zeros(1, hidden_size)
                hidden = hidden.to(self.dl_studio.device)
                review_tensor, category, sentiment = data['review'], data['category'], data['sentiment']
                review_tensor = review_tensor.to(self.dl_studio.device)
                sentiment = sentiment.to(self.dl_studio.device)
                optimizer.zero_grad()
                input = torch.zeros(1, review_tensor.shape[2])
                input = input.to(self.dl_studio.device)
                for k in range(review_tensor.shape[0]):
                    input[0, :] = review_tensor[0, k]
                    output, hidden = net(input, hidden)
                loss = criterion(output, torch.argmax(sentiment, 1))
                running_loss += loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()
                if i % 100 == 99:
                    avg_loss = running_loss / float(100)
                    current_time = time.clock()
                    time_elapsed = current_time-start_time
                    print("[epoch:%d  iter:%4d  elapsed_time: %4d secs]     loss: %.3f" % (epoch+1, i+1, time_elapsed, avg_loss))
                    accum_times.append(current_time-start_time)
                    FILE.write("%.3f\n" % avg_loss)
                    FILE.flush()
                    running_loss = 0.0
        print("\nFinished Training\n")
        self.save_model(net)

    def run_code_for_training_for_text_classification_with_gru(self, net, hidden_size):
        filename_for_out = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
        FILE = open(filename_for_out, 'w')
        net = copy.deepcopy(net)
        net = net.to(self.dl_studio.device)
        # Note that the GREnet now produces the LogSoftmax output:
        criterion = nn.NLLLoss()
#            criterion = nn.MSELoss()
#            criterion = nn.CrossEntropyLoss()
        accum_times = []
        optimizer = optim.SGD(net.parameters(),
                              lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
        for epoch in range(self.dl_studio.epochs):
            print("")
            running_loss = 0.0
            start_time = time.clock()
            for i, data in enumerate(self.train_dataloader):
                review_tensor, category, sentiment = data['review'], data['category'], data['sentiment']
                review_tensor = review_tensor.to(self.dl_studio.device)
                sentiment = sentiment.to(self.dl_studio.device)
                # The following type conversion needed for MSELoss:
                ##sentiment = sentiment.float()
                optimizer.zero_grad()
                hidden = net.init_hidden(1).to(self.dl_studio.device)
                for k in range(1):  # MODIFIED FROM: review_tensor.shape[1]):
                    output, hidden = net(review_tensor, hidden)  # MODIFIED FROM: review_tensor[0,k]
                # If using NLLLoss, CrossEntropyLoss
                loss = criterion(output, torch.argmax(sentiment, 1))
                # If using MSELoss:
                ## loss = criterion(output, sentiment)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                if i % 100 == 99:
                    avg_loss = running_loss / float(100)
                    current_time = time.clock()
                    time_elapsed = current_time-start_time
                    print("[epoch:%d  iter:%4d  elapsed_time:%4d secs]     loss: %.3f" % (epoch+1, i+1, time_elapsed, avg_loss))
                    accum_times.append(current_time-start_time)
                    FILE.write("%.3f\n" % avg_loss)
                    FILE.flush()
                    running_loss = 0.0
        print("Total Training Time: {}".format(str(sum(accum_times))))
        print("\nFinished Training\n")
        self.save_model(net)

    def run_code_for_testing_text_classification_no_gru(self, net, hidden_size):
        net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
        classification_accuracy = 0.0
        negative_total = 0
        positive_total = 0
        confusion_matrix = torch.zeros(2, 2)
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                review_tensor, category, sentiment = data['review'], data['category'], data['sentiment']
                input = torch.zeros(1, review_tensor.shape[2])
                hidden = torch.zeros(1, hidden_size)
                for k in range(review_tensor.shape[1]):
                    input[0, :] = review_tensor[0, k]
                    output, hidden = net(input, hidden)
                predicted_idx = torch.argmax(output).item()
                gt_idx = torch.argmax(sentiment).item()
                if i % 100 == 99:
                    print("   [i=%4d]    predicted_label=%d       gt_label=%d" % (i+1, predicted_idx, gt_idx))
                if predicted_idx == gt_idx:
                    classification_accuracy += 1
                if gt_idx is 0:
                    negative_total += 1
                elif gt_idx is 1:
                    positive_total += 1
                confusion_matrix[gt_idx, predicted_idx] += 1
        out_percent = np.zeros((2, 2), dtype='float')
        print("\n\nNumber of positive reviews tested: %d" % positive_total)
        print("\n\nNumber of negative reviews tested: %d" % negative_total)
        print("\n\nDisplaying the confusion matrix:\n")
        out_str = "                      "
        out_str += "%18s    %18s" % ('predicted negative', 'predicted positive')
        print(out_str + "\n")
        for i, label in enumerate(['true negative', 'true positive']):
            out_percent[0, 0] = "%.3f" % (100 * confusion_matrix[0, 0] / float(negative_total))
            out_percent[0, 1] = "%.3f" % (100 * confusion_matrix[0, 1] / float(negative_total))
            out_percent[1, 0] = "%.3f" % (100 * confusion_matrix[1, 0] / float(positive_total))
            out_percent[1, 1] = "%.3f" % (100 * confusion_matrix[1, 1] / float(positive_total))
            out_str = "%12s:  " % label
            for j in range(2):
                out_str += "%18s" % out_percent[i, j]
            print(out_str)

    def run_code_for_testing_text_classification_with_gru(self, net, hidden_size):
        net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
        classification_accuracy = 0.0
        negative_total = 0
        positive_total = 0
        confusion_matrix = torch.zeros(2, 2)
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                review_tensor, category, sentiment = data['review'], data['category'], data['sentiment']
                hidden = net.init_hidden(1)
                for k in range(1):
                    output, hidden = net(review_tensor, hidden)
                predicted_idx = torch.argmax(output).item()
                gt_idx = torch.argmax(sentiment).item()
                if i % 100 == 99:
                    print("   [i=%d]    predicted_label=%d       gt_label=%d\n\n" % (i+1, predicted_idx, gt_idx))
                if predicted_idx == gt_idx:
                    classification_accuracy += 1
                if gt_idx is 0:
                    negative_total += 1
                elif gt_idx is 1:
                    positive_total += 1
                confusion_matrix[gt_idx, predicted_idx] += 1
        out_percent = np.zeros((2, 2), dtype='float')
        print("\n\nNumber of positive reviews tested: %d" % positive_total)
        print("\n\nNumber of negative reviews tested: %d" % negative_total)
        print("\n\nDisplaying the confusion matrix:\n")
        out_str = "                      "
        out_str += "%18s    %18s" % ('predicted negative', 'predicted positive')
        print(out_str + "\n")
        for i, label in enumerate(['true negative', 'true positive']):
            out_percent[0, 0] = "%.3f" % (100 * confusion_matrix[0, 0] / float(negative_total))
            out_percent[0, 1] = "%.3f" % (100 * confusion_matrix[0, 1] / float(negative_total))
            out_percent[1, 0] = "%.3f" % (100 * confusion_matrix[1, 0] / float(positive_total))
            out_percent[1, 1] = "%.3f" % (100 * confusion_matrix[1, 1] / float(positive_total))
            out_str = "%12s:  " % label
            for j in range(2):
                out_str += "%18s" % out_percent[i, j]
            print(out_str)


def plot_loss(self):
    plt.figure()
    plt.plot(self.LOSS)
    plt.show()


dls = DLStudio(
    dataroot="/content/drive/My Drive/Colab Notebooks/ece695_Deep_Learning/DLStudio-1.1.3/Examples/data/",
    path_saved_model="./saved_model",
    momentum=0.9,
    #                  learning_rate =  0.004,
    learning_rate=1e-6,
    epochs=1,
    batch_size=1,
    classes=('negative', 'positive'),
    debug_train=1,
    debug_test=1,
    use_gpu=True,
)


text_cl = TextClassification(dl_studio=dls)
dataserver_train = TextClassification.SentimentAnalysisDataset(
    train_or_test='train',
    dl_studio=dls,
    #                                dataset_file = "sentiment_dataset_train_3.tar.gz",
    #                                dataset_file = "sentiment_dataset_train_200.tar.gz",
    dataset_file="sentiment_dataset_train_40.tar.gz",
)
dataserver_test = TextClassification.SentimentAnalysisDataset(
    train_or_test='test',
    dl_studio=dls,
    #                                dataset_file = "sentiment_dataset_test_3.tar.gz",
    #                                dataset_file = "sentiment_dataset_test_200.tar.gz",
    dataset_file="sentiment_dataset_test_40.tar.gz",
)
text_cl.dataserver_train = dataserver_train
text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)

vocab_size = dataserver_train.get_vocab_size()
input_size = 1  # since we are using an integer for input word instead of len(vocab) one-hot vector
hidden_size = 512
output_size = 2                            # for positive and negative sentiments
n_layers = 2

model = text_cl.GRUnet(input_size, hidden_size, output_size, n_layers)

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

num_layers = len(list(model.parameters()))

print("\n\nThe number of layers in the model: %d" % num_layers)
print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
print("\nThe size of the vocabulary (which is also the size of the one-hot vecs for words): %d\n\n" % vocab_size)

# TRAINING:
print("\nStarting training --- BE VERY PATIENT, PLEASE!  The first report will be at 100th iteration. May take around 5 minutes.\n")
text_cl.run_code_for_training_for_text_classification_with_gru(model, hidden_size)

# TESTING:
# import pymsgbox
# response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
# if response == "OK":
text_cl.run_code_for_testing_text_classification_with_gru(model, hidden_size)

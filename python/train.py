from cnn import CNN
import torch
import random
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt

class TextTrain():
    def __init__(self, config, train_X, train_Y, test_X, test_Y):
        self.config = config
        self.train_X = torch.LongTensor(train_X)
        self.train_Y = torch.LongTensor(train_Y)
        self.test_X = torch.LongTensor(test_X)
        self.test_Y = torch.LongTensor(test_Y)
        self.setup_seed(1234)
        self.__cnn_build()

    @staticmethod
    def setup_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def evaluate(criterion, o, y):
        loss = criterion(o, y)
        accuracy = float((y == torch.argmax(o, 1)).sum()) / y.shape[0]
        return loss, accuracy

    @staticmethod
    def print_report_table(o, y, labels):
        p, r, f1, s = metrics.precision_recall_fscore_support(y.cpu(), o.cpu())
        tot_p = np.average(p, weights=s)
        tot_r = np.average(r, weights=s)
        tot_f1 = np.average(f1, weights=s)
        tot_s = np.sum(s)
        res1 = pd.DataFrame({
            'Label': labels,
            'Precision': p,
            'Recall': r,
            'F1': f1,
            'Support': s
        })
        res2 = pd.DataFrame({
            'Label': ['总体'],
            'Precision': [tot_p],
            'Recall': [tot_r],
            'F1': [tot_f1],
            'Support': [tot_s]
        })
        res2.index = [2]
        res = pd.concat([res1, res2])
        print('\n Report Table:')
        print(res)

    @staticmethod
    def print_confusion_matrix(o, y, labels):
        df = pd.DataFrame(metrics.confusion_matrix(y.cpu().detach().numpy(), o.cpu().detach().numpy()), 
                          columns=labels,
                          index=labels)
        print('\n Confusion Matrix:')
        print(df)

    @staticmethod
    def plot_roc_curve(o, y):
        fpr, tpr, threshold = metrics.roc_curve(y.cpu(), o.cpu())
        roc_auc = metrics.auc(fpr,tpr)

        plt.figure(figsize=(10,10))
        plt.plot(fpr, tpr, color='darkorange',
                lw=2, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FP')
        plt.ylabel('TP')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()
    
    def __cnn_build(self):
        torch.no_grad()
        torch.cuda.empty_cache()
        self.device = torch.device('cuda:0')
        self.cnn = CNN(embedding_dim=self.config['embedding_dim'], 
                       vocab_size=self.config['vocab_size'],
                       num_filters=self.config['num_filters'], 
                       kernel_size=self.config['kernel_size'], 
                       hidden_dim=self.config['hidden_dim'], 
                       dropout_keep_prob=self.config['dropout_keep_prob'], 
                       num_classes=self.config['num_classes']).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.cnn.parameters(), lr=self.config['learning_rate'], momentum=0.9)

    def cnn_train(self):
        max_accuracy = 0
        for i in range(self.config['num_iteration']):
            selected_index = random.sample(list(range(len(self.train_X))), k=self.config['batch_size'])
            batch_X = self.train_X[selected_index].to(self.device)
            batch_Y = self.train_Y[selected_index].to(self.device)
            outputs = self.cnn(batch_X)
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, batch_Y)
            loss.backward()
            self.optimizer.step()
            step = i + 1
            if step % self.config['print_per_batch'] == 0 or step == 1:
                loss, accuracy = self.evaluate(self.criterion, outputs, batch_Y)
                print('step:%d loss:%.4f accuracy:%.4f' % (step, loss, accuracy))
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    torch.save(self.cnn.state_dict(),'../model/best_cnn.pkl')

    def cnn_test(self):
        self.cnn.load_state_dict(torch.load('../model/best_cnn.pkl'))
        outputs = self.cnn(self.test_X.to(self.device))
        test_Y = self.test_Y.to(self.device)
        loss, accuracy = self.evaluate(self.criterion, outputs, test_Y)
        print('test loss:%.4f accuracy:%.4f' % (loss, accuracy))
        outputs = torch.argmax(outputs, 1)
        self.print_report_table(outputs, test_Y, self.config['labels'])
        self.print_confusion_matrix(outputs, test_Y, self.config['labels'])
        self.plot_roc_curve(outputs, test_Y)




import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold
from custom_model import MultiHeadClassifier, ConvModel
from torch.utils import data
from torch import nn, optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sampling(X, Y, Number):
    """
    随机选取样本中的Number样本
    :param X: 总数据集
    :param Y: 总数据标签
    :param Number: 随机选择的样本数
    :return: 采样数据
    """
    AN = len(X)
    index = [i for i in range(AN)]
    newIndex = random.sample(index, Number)
    NewX = pd.DataFrame(X, index=newIndex)
    NewY = pd.DataFrame(Y, index=newIndex)
    return NewX, NewY

def getMatric(y_test, ptest):
    fpr1, tpr1, thresholds = metrics.roc_curve(y_test, ptest)
    roc_auc = metrics.auc(fpr1, tpr1)
    precision1, recall1, _ = metrics.precision_recall_curve(y_test, ptest)
    aupr = metrics.auc(recall1, precision1)
    accuracy = metrics.accuracy_score(y_test, ptest)
    precision = metrics.precision_score(y_test, ptest, zero_division=0)
    recall = metrics.recall_score(y_test, ptest, zero_division=0)
    F1 = metrics.f1_score(y_test, ptest)
    if np.isnan(roc_auc): roc_auc = 0
    if np.isnan(aupr): aupr = 0
    return roc_auc, aupr, accuracy, precision, recall, F1

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=True)

def train_model(model_type, net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_features = train_features.clone().detach().requires_grad_(True).to(torch.float32).to(device)
    train_labels = train_labels.clone().detach().requires_grad_(True).to(torch.float32).to(device)
    train_iter = load_array((train_features, train_labels), batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    for epoch in range(num_epochs):
        correct = 0
        running_loss = 0
        total = 0
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            outputs = net(X)
            act = nn.Sigmoid()
            outputs = act(outputs)
            loss = loss_fn(outputs, y.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step()
        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}, loss: {running_loss / len(train_iter):.3f}, accuracy: {accuracy:.3f}%, model: {model_type}')
    return net

def train_ensmble(train_features, train_labels, test_features, test_labels, return_individual_metrics=False):
    train_features = torch.tensor(train_features, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)
    test_features = torch.tensor(test_features, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.float32).to(device)
    X_base, X_meta, y_base, y_meta = train_test_split(train_features, train_labels, test_size=0.3, random_state=42)

    # Train SVM model
    print("SVM*" * 20, device)
    svm_model = SVC(gamma=0.1, C=2.5, kernel='poly', tol=0.001, cache_size=100, probability=True)
    svm_model.fit(X_base.cpu().detach().numpy(), y_base.cpu().detach().numpy())
    X_meta_svm = svm_model.predict_proba(X_meta.cpu().detach().numpy())

    # Train MultiHeadClassifier
    print("MHSA*" * 20, device)
    mhsa = MultiHeadClassifier(input_dim=400, num_classes=2, num_heads=8, hidden_size=256).to(device)
    num_epochs, lr, weight_decay, batch_size = 30, 0.001, 1e-5, 64
    mhsa = train_model('MHSA', mhsa, X_base, y_base, X_meta, y_meta, num_epochs, lr, weight_decay, batch_size)
    X_meta_mhsa = mhsa(X_meta)

    # Train ConvModel
    print("CNN*" * 20, device)
    cnn = ConvModel().to(device)
    num_epochs, lr, weight_decay, batch_size = 30, 0.001, 1e-5, 64
    cnn = train_model('CNN', cnn, X_base, y_base, X_meta, y_meta, num_epochs, lr, weight_decay, batch_size)
    X_meta_cnn = cnn(X_meta)

    if return_individual_metrics:
            svm_metrics = getMatric(svm_model.predict(X_meta.cpu().detach().numpy()), y_meta.cpu().detach().numpy())
            mhsa_metrics = getMatric(torch.argmax(X_meta_mhsa, dim=1).cpu().detach().numpy(), y_meta.cpu().detach().numpy())
            cnn_metrics = getMatric(torch.argmax(X_meta_cnn, dim=1).cpu().detach().numpy(), y_meta.cpu().detach().numpy())

    # Combine meta-features
    X_meta_tensor = torch.from_numpy(np.concatenate((X_meta_mhsa.cpu().detach().numpy(), X_meta_svm, X_meta_cnn.cpu().detach().numpy()),axis=1)).float().to(device)
    y_meta_tensor = torch.from_numpy(y_meta.cpu().detach().numpy()).long().to(device)

    # Define and train MetaClassifier
    class MetaClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(6, 12)
            self.fc2 = nn.Linear(12, 8)
            self.fc3 = nn.Linear(8, 2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = F.softmax(self.fc3(x), dim=1)
            return x
    meta_classifier = MetaClassifier().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(meta_classifier.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Train the meta-classifier model
    for epoch in range(1000):
        # Forward pass
        outputs = meta_classifier(X_meta_tensor)
        loss = criterion(outputs, y_meta_tensor)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{1000}], Loss: {loss.item():.4f}')

    # Predict the test set using the meta-classifier
    with torch.no_grad():
        X_test_mhsa = mhsa(test_features)
        X_test_svm = svm_model.predict_proba(test_features.cpu().detach().numpy())
        X_test_cnn = cnn(test_features)
        X_test_meta_tensor = torch.from_numpy(np.concatenate((X_test_mhsa.cpu().detach().numpy(), X_test_svm,X_test_cnn.cpu().detach().numpy()), axis=1)).float().to(device)
        output = meta_classifier(X_test_meta_tensor)
        y_pred = torch.argmax(output, dim=1)
        print('The probabilities are:', output)
    accuracy = (y_pred == test_labels).sum().item() / len(test_labels)
    print(f'The accuracy of the Stacking model is {accuracy:.2f}')
    if return_individual_metrics:
        return svm_metrics, mhsa_metrics, cnn_metrics, getMatric(y_pred.cpu().detach().numpy(),
                                                                 test_labels.cpu().detach().numpy())
    else:
        return getMatric(y_pred.cpu().detach().numpy(), test_labels.cpu().detach().numpy())

if __name__ == '__main__':
    data1 = pd.read_csv('data.csv')
    print("*" * 50, data1.shape)
    corr_matrix = data1.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    data1.drop(to_drop, axis=1, inplace=True)
    X = data1.loc[:, data1.columns != "label"]
    Y = data1["label"]
    X, Y = sampling(X, Y, X.shape[0])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    times = 20

    SresultsForOyt20, dis_SresultsForOyt20 = [], []
    MresultsForOyt20, dis_MresultsForOyt20 = [], []
    CresultsForOyt20, dis_CresultsForOyt20 = [], []
    resultsForOyt20, dis_resultsForOyt20 = [], []

    # Cycle 20 times
    for i in range(times):
        print("*" * 50, "第", i + 1, "次", "*" * 50)
        SfinalResults = []
        MfinalResults = []
        CfinalResults = []
        finalResults = []
        kf = KFold(n_splits=5, shuffle=False)
        for train_index, test_index in kf.split(X, Y):
            x_train = X.iloc[train_index]
            y_train = Y.iloc[train_index]
            x_test = X.iloc[test_index]
            y_test = Y.iloc[test_index]
            x_train = np.array(x_train)
            x_test = np.array(x_test)
            y_train = np.array(y_train).ravel()
            y_test = np.array(y_test).ravel()
            svm_metrics, mhsa_metrics, cnn_metrics, ensemble_metrics = train_ensmble(x_train, y_train, x_test, y_test, return_individual_metrics=True)
            SfinalResults.append(svm_metrics)
            MfinalResults.append(mhsa_metrics)
            CfinalResults.append(cnn_metrics)
            finalResults.append(ensemble_metrics)

        print("-------------------一次五折交叉验证的最终结果-------------------")
        SfinalResults = np.array(SfinalResults)
        MfinalResults = np.array(MfinalResults)
        CfinalResults = np.array(CfinalResults)
        finalResults = np.array(finalResults)

        SVMRes = (SfinalResults[0] + SfinalResults[1] + SfinalResults[2] + SfinalResults[3] + SfinalResults[4]) / 5.0
        dis_SVMRes = np.std(np.array([SfinalResults[0], SfinalResults[1], SfinalResults[2], SfinalResults[3], SfinalResults[4]]), axis=0)
        MHSARes = (MfinalResults[0] + MfinalResults[1] + MfinalResults[2] + MfinalResults[3] + MfinalResults[4]) / 5.0
        dis_MHSARes = np.std(np.array([MfinalResults[0], MfinalResults[1], MfinalResults[2], MfinalResults[3], MfinalResults[4]]), axis=0)
        CNNRes = (CfinalResults[0] + CfinalResults[1] + CfinalResults[2] + CfinalResults[3] + CfinalResults[4]) / 5.0
        dis_CNNRes = np.std(np.array([CfinalResults[0], CfinalResults[1], CfinalResults[2], CfinalResults[3], CfinalResults[4]]), axis=0)
        TFRes = (finalResults[0] + finalResults[1] + finalResults[2] + finalResults[3] + finalResults[4]) / 5.0
        dis_TFRes = np.std(np.array([finalResults[0], finalResults[1], finalResults[2], finalResults[3], finalResults[4]]), axis=0)

        print("SVM Res auc: %.4f" % SVMRes[0], "aupr: %.4f" % SVMRes[1], "accuracy: %.4f" % SVMRes[2],
                "precision: %.4f" % SVMRes[3], "recall: %.4f" % SVMRes[4], "F1: %.4f" % SVMRes[5])
        print("SVM Res auc 误差 auc: %.4f" % dis_SVMRes[0], "aupr: %.4f" % dis_SVMRes[1], "accuracy: %.4f" % dis_SVMRes[2],
                        "precision: %.4f" % dis_SVMRes[3], "recall: %.4f" % dis_SVMRes[4], "F1: %.4f" % dis_SVMRes[5])
        print("MHSA Res auc: %.4f" % MHSARes[0], "aupr: %.4f" % MHSARes[1], "accuracy: %.4f" % MHSARes[2],
                 "precision: %.4f" % MHSARes[3], "recall: %.4f" % MHSARes[4], "F1: %.4f" % MHSARes[5])
        print("MHSA Res auc 误差 auc: %.4f" % dis_MHSARes[0], "aupr: %.4f" % dis_MHSARes[1], "accuracy: %.4f" % dis_MHSARes[2],
                         "precision: %.4f" % dis_MHSARes[3], "recall: %.4f" % dis_MHSARes[4], "F1: %.4f" % dis_MHSARes[5])
        print("CNN Res auc: %.4f" % CNNRes[0], "aupr: %.4f" % CNNRes[1], "accuracy: %.4f" % CNNRes[2],
                "precision: %.4f" % CNNRes[3], "recall: %.4f" % CNNRes[4], "F1: %.4f" % CNNRes[5])
        print("CNN Res auc 误差 auc: %.4f" % dis_CNNRes[0], "aupr: %.4f" % dis_CNNRes[1], "accuracy: %.4f" % dis_CNNRes[2],
                        "precision: %.4f" % dis_CNNRes[3], "recall: %.4f" % dis_CNNRes[4], "F1: %.4f" % dis_CNNRes[5])
        print("Stacking Res auc: %.4f" % TFRes[0], "aupr: %.4f" % TFRes[1], "accuracy: %.4f" % TFRes[2],
                     "precision: %.4f" % TFRes[3], "recall: %.4f" % TFRes[4], "F1: %.4f" % TFRes[5])
        print("Stacking Res auc 误差 auc: %.4f" % dis_TFRes[0], "aupr: %.4f" % dis_TFRes[1], "accuracy: %.4f" % dis_TFRes[2],
                             "precision: %.4f" % dis_TFRes[3], "recall: %.4f" % dis_TFRes[4], "F1: %.4f" % dis_TFRes[5])

        SresultsForOyt20.append([SVMRes])
        dis_SresultsForOyt20.append([dis_SVMRes])
        MresultsForOyt20.append([MHSARes])
        dis_MresultsForOyt20.append([dis_MHSARes])
        CresultsForOyt20.append([CNNRes])
        dis_CresultsForOyt20.append([dis_CNNRes])
        resultsForOyt20.append([TFRes])
        dis_resultsForOyt20.append([dis_TFRes])

    print("-------------------二十次五折交叉验证后的结果-------------------")
    SVMRes20 = []
    dis_SVMRes20 = []
    MHSARes20 = []
    dis_MHSARes20 = []
    CNNRes20 = []
    dis_CNNRes20 = []
    TFRes20 = []
    dis_TFRes20 = []

    for i in range(times):
        SVMRes20.append(SresultsForOyt20[i][0])
        dis_SVMRes20.append(dis_SresultsForOyt20[i][0])
        MHSARes20.append(MresultsForOyt20[i][0])
        dis_MHSARes20.append(dis_MresultsForOyt20[i][0])
        CNNRes20.append(CresultsForOyt20[i][0])
        dis_CNNRes20.append(dis_CresultsForOyt20[i][0])
        TFRes20.append(resultsForOyt20[i][0])
        dis_TFRes20.append(dis_resultsForOyt20[i][0])

    SVMRes = np.average(SVMRes20, axis=0)
    dis_SVMRes = np.std(dis_SVMRes20, axis=0)
    MHSARes = np.average(MHSARes20, axis=0)
    dis_MHSARes = np.std(dis_MHSARes20, axis=0)
    CNNRes = np.average(CNNRes20, axis=0)
    dis_CNNRes = np.std(dis_CNNRes20, axis=0)
    TFRes = np.average(TFRes20, axis=0)
    dis_TFRes = np.std(dis_TFRes20, axis=0)

    print("final SVM Final Res auc: %.4f" % SVMRes[0], "aupr: %.4f" % SVMRes[1], "accuracy: %.4f" % SVMRes[2],
                        "precision: %.4f" % SVMRes[3], "recall: %.4f" % SVMRes[4], "F1: %.4f" % SVMRes[5])
    print("final SVM Final Res 误差 auc: %.4f" % dis_SVMRes[0], "aupr: %.4f" % dis_SVMRes[1], "accuracy: %.4f" % dis_SVMRes[2],
                            "precision: %.4f" % dis_SVMRes[3], "recall: %.4f" % dis_SVMRes[4], "F1: %.4f" % dis_SVMRes[5])
    print("final MHSA Final Res auc: %.4f" % MHSARes[0], "aupr: %.4f" % MHSARes[1], "accuracy: %.4f" % MHSARes[2],
                         "precision: %.4f" % MHSARes[3], "recall: %.4f" % MHSARes[4], "F1: %.4f" % MHSARes[5])
    print("final MHSA Final Res 误差 auc: %.4f" % dis_MHSARes[0], "aupr: %.4f" % dis_MHSARes[1], "accuracy: %.4f" % dis_MHSARes[2],
                             "precision: %.4f" % dis_MHSARes[3], "recall: %.4f" % dis_MHSARes[4], "F1: %.4f" % dis_MHSARes[5])
    print("final CNN Final Res auc: %.4f" % CNNRes[0], "aupr: %.4f" % CNNRes[1], "accuracy: %.4f" % CNNRes[2],
                        "precision: %.4f" % CNNRes[3], "recall: %.4f" % CNNRes[4], "F1: %.4f" % CNNRes[5])
    print("final CNN Final Res 误差 auc: %.4f" % dis_CNNRes[0], "aupr: %.4f" % dis_CNNRes[1], "accuracy: %.4f" % dis_CNNRes[2],
                            "precision: %.4f" % dis_CNNRes[3], "recall: %.4f" % dis_CNNRes[4], "F1: %.4f" % dis_CNNRes[5])
    print("final Stacking Final Res auc: %.4f" % TFRes[0], "aupr: %.4f" % TFRes[1], "accuracy: %.4f" % TFRes[2],
                             "precision: %.4f" % TFRes[3], "recall: %.4f" % TFRes[4], "F1: %.4f" % TFRes[5])
    print("final Stacking Final Res 误差 auc: %.4f" % dis_TFRes[0], "aupr: %.4f" % dis_TFRes[1], "accuracy: %.4f" % dis_TFRes[2],
                                 "precision: %.4f" % dis_TFRes[3], "recall: %.4f" % dis_TFRes[4], "F1: %.4f" % dis_TFRes[5])
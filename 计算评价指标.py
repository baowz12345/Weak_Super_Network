import numpy as np
from sklearn import metrics

# 读取txt文件中的预测值和标签
with open(r'D:\workship\anewlife\wscnnlstmchange\真实值与预测值\bilstm+cnn\score_1fold.txt', 'r') as f:
    lines = f.readlines()
    pred_scores = np.array([float(line.split()[0]) for line in lines])
    true_labels = np.array([float(line.split()[1]) for line in lines])

# 计算AUC
auc = metrics.roc_auc_score(true_labels, pred_scores)

# 计算f1-score
threshold = 0.5 # 阈值，二分类问题一般为0.5
pred_labels = (pred_scores >= threshold).astype(int)
f1_score = metrics.f1_score(true_labels, pred_labels)

# 计算MCC
mcc = metrics.matthews_corrcoef(true_labels, pred_labels)

# 计算Precision和recall
precision = metrics.precision_score(true_labels, pred_labels)
recall = metrics.recall_score(true_labels, pred_labels)
# 计算accuracy
accuracy = metrics.accuracy_score(true_labels, pred_labels)

# 输出结果
print("AUC: {:.4f}".format(auc))
print("f1-score: {:.4f}".format(f1_score))
print("MCC: {:.4f}".format(mcc))
print("Precision: {:.4f}".format(precision))
print("recall: {:.4f}".format(recall))
print("accuracy: {:.4f}".format(accuracy))
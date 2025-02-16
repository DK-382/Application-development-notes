import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 生成随机数据作为示例
X = np.random.rand(10000, 50)  # 10000个样本，每个样本50个特征
y = np.random.randint(2, size=10000)  # 10000个样本的二分类标签

# 划分训练集和测试集，测试集占20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost的基本参数设置
params = {
    'objective': 'binary:logistic',  # 二分类问题，使用逻辑回归目标函数
    'eta': 0.1,  # 学习率，控制每棵树的贡献
    'max_depth': 6,  # 树的最大深度，控制模型复杂度，防止过拟合
    'subsample': 0.8,  # 训练每棵树时使用的数据样本比例，增加随机性，防止过拟合
    'colsample_bytree': 0.8,  # 训练每棵树时使用的特征比例，增加随机性，防止过拟合
    'min_child_weight': 1,  # 叶子节点的最小权重和，控制模型复杂度，防止过拟合
    'eval_metric': 'logloss'  # 评估指标，对于二分类问题使用对数损失
}

# 准备XGBoost的DMatrix数据格式
dtrain = xgb.DMatrix(X_train, label=y_train)  # 训练数据集
dtest = xgb.DMatrix(X_test, label=y_test)  # 测试数据集

# 设置早停法的参数
evals = [(dtrain, 'train'), (dtest, 'test')]
num_boost_round = 1000  # 最大迭代次数
early_stopping_rounds = 50  # 如果连续50轮迭代没有提升，则停止训练

# 初始化evals_result字典来存储评估结果
evals_result = {}

# 训练模型，同时监控训练集和测试集的性能
bst = xgb.train(params, dtrain, num_boost_round, evals=evals,
                early_stopping_rounds=early_stopping_rounds, verbose_eval=True,
                evals_result=evals_result)

# 预测测试集的结果
y_pred = bst.predict(dtest)

# 将预测概率转换为二分类标签，阈值为0.5
y_pred_class = (y_pred > 0.5).astype(int)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Test Accuracy: {accuracy:.4f}")

# 输出模型在训练和测试过程中的性能
print("\n模型在训练和测试过程中的性能：")
for key in evals_result:
    print(f"{key}:")
    for metric, values in evals_result[key].items():
        print(f"  {metric}: {values}")
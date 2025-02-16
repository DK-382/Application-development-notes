#include <xgboost/xgboost.h>
#include <iostream>
#include <vector>
#include <random>

int main() {
    // 设置随机种子以确保结果可重复
    unsigned seed = 42;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // 生成随机数据作为示例
    std::vector<float> X;
    std::vector<float> y;
    for (int i = 0; i < 10000; ++i) {
        std::vector<float> row(50);
        for (int j = 0; j < 50; ++j) {
            row[j] = dis(gen);
        }
        X.insert(X.end(), row.begin(), row.end());
        y.push_back(gen() > 0.5 ? 0 : 1);
    }

    // 划分训练集和测试集
    int train_size = 8000;
    std::vector<float> X_train(X.begin(), X.begin() + train_size * 50);
    std::vector<float> X_test(X.begin() + train_size * 50, X.end());
    std::vector<float> y_train(y.begin(), y.begin() + train_size);
    std::vector<float> y_test(y.begin() + train_size, y.end());

    // 创建DMatrix
    xgboost::DMatrix dtrain(&X_train[0], train_size, 50);
    xgboost::DMatrix dtest(&X_test[0], 2000, 50);

    // 设置参数
    std::vector<std::pair<std::string, std::string> > params = {
        {"objective", "binary:logistic"},
        {"eta", "0.1"},
        {"max_depth", "6"},
        {"subsample", "0.8"},
        {"colsample_bytree", "0.8"},
        {"min_child_weight", "1"},
        {"eval_metric", "logloss"}
    };

    // 训练模型
    xgboost::bst_ulong num_round = 100;
    xgboost::train(params, dtrain, num_round);

    // 预测
    std::vector<float> pred;
    xgboost::predict(dtrain, &pred);

    // 计算准确率
    int correct = 0;
    for (size_t i = 0; i < pred.size(); ++i) {
        if (pred[i] > 0.5f && y_test[i] == 1) {
            ++correct;
        } else if (pred[i] <= 0.5f && y_test[i] == 0) {
            ++correct;
        }
    }
    float accuracy = static_cast<float>(correct) / pred.size();
    std::cout << "Test Accuracy: " << accuracy << std::endl;

    return 0;
}
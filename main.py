import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# 读取数据集
def load_dataset():
    pos_reviews = []
    neg_reviews = []
    pos_dir = 'train/pos/'
    neg_dir = 'train/neg/'
    for file in os.listdir(pos_dir):
        with open(pos_dir + file, 'r', encoding='utf-8') as f:
            pos_reviews.append(f.read())
    for file in os.listdir(neg_dir):
        with open(neg_dir + file, 'r', encoding='utf-8') as f:
            neg_reviews.append(f.read())
    reviews = pos_reviews + neg_reviews
    labels = ['positive'] * len(pos_reviews) + ['negative'] * len(neg_reviews)
    return reviews, labels


# 特征提取
def extract_features(reviews):
    vectorizer = CountVectorizer()  # 使用词袋模型进行特征提取
    X = vectorizer.fit_transform(reviews)
    return X


# 训练决策树模型
def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf, X_val, y_val


# 测试模型
def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# 主函数
def main():
    # 加载数据集
    reviews, labels = load_dataset()
    # 提取特征
    X = extract_features(reviews)
    # 训练模型
    model, X_test, y_test = train_model(X, labels)
    # 测试模型
    accuracy = test_model(model, X_test, y_test)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    main()

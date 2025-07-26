import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import word_tokenize
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
def load_data(path):
    return pd.read_csv(path)
# Kiểm tra thông tin và trực quan hóa dữ liệu thô
def check_data(df, target_col, feature_col):
    print("Thông tin datasets:")
    print(df.info())

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    disease_counts = df[target_col].value_counts()
    bars = ax.barh(range(len(disease_counts)), disease_counts.values, color='skyblue', edgecolor='navy', alpha=0.7)
    ax.set_yticks(range(len(disease_counts)))
    ax.set_yticklabels(disease_counts.index, fontsize=10)
    ax.set_xlabel('Số lượng mẫu', fontsize=12)
    ax.set_title('Phân bố dữ liệu theo nhóm bệnh', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 10, bar.get_y() + bar.get_height() / 2,
                f'{int(width)}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()
# Encode nhãn nhóm bệnh
def encode_labels(data, column_name, encoded_label_colname, encoder):
    data[encoded_label_colname] = encoder.fit_transform(data[column_name])
    return data
# Lấy Vietnamese stopwords
def get_vietnamese_stopwords():
    try:
        response = requests.get("https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt")
        return set(response.text.splitlines())
    except Exception as e:
        print(f"Error name: {e}")
        return set()
# Tiền xử lý văn bản
def preprocess_text(text, stopwords):
    if isinstance(text, str):
        # Loại bỏ kí tự đặc biệt
        text = re.sub(r'[^\w\s]', '', text)
        # Chuyển text về chữ thường
        text = text.lower()
        # Tách từ bằng underthesea
        text_tokenized = word_tokenize(text, format="text")
        # Loại bỏ stopwords
        words = text_tokenized.split()
        filtered_words = [word for word in words if word not in stopwords]
        return " ".join(filtered_words)
    return ""
# Làm sạch dataset
def process_dataset(data, label_col, feature_col, processed_feature_col, stopwords):
    # Loại bỏ trùng lắp
    data.drop_duplicates(inplace=True)
    # Loại bỏ giá trị bị thiếu
    data = data.dropna(subset=[feature_col, label_col])
    # Xử lý text trong cột feature
    data[processed_feature_col] = data[feature_col].apply(lambda x: preprocess_text(x, stopwords))
    return data
# Tách dữ liệu thành 3 tập train/validation/test (80% - 10% - 10%)
def split_data(X, y, test_size=0.2, val_size=0.5, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test
# Huấn luyện mô hình
def train_model(X_train, y_train, X_val, y_val):
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    model = LogisticRegression(random_state=42, class_weight=class_weight_dict, C=10.0, max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    # # Đánh giá trên tập validation
    # y_val_pred = model.predict(X_val)
    # val_accuracy = accuracy_score(y_val, y_val_pred)
    # val_f1 = f1_score(y_val, y_val_pred, average='macro')
    # print(f"Hiệu suất trên tập validation của model:")
    # print(f"Độ chính xác: {val_accuracy:.4f}")
    # print(f"F1 Score (Macro): {val_f1:.4f}")
    return model
# Vẽ ma trận nhầm lẫn và đánh giá model trên tập test
def evaluate_model(model, X_test, y_test, label_encoder):
    y_test_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Ma trận nhầm lẫn', fontsize=16, fontweight='bold')
    plt.xlabel('Dự đoán', fontsize=12)
    plt.ylabel('Thực tế', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    print(f"Hiệu suất trên tập test của model:")
    print(f"Độ chính xác: {test_accuracy:.4f}")
    print(f"F1 Score (Macro): {test_f1:.4f}")
    return test_accuracy, test_f1
# Dự đoán nhóm bệnh từ mô tả
def predict_disease_group(input_text, model, vectorizer, label_encoder, stopwords):
    processed_text = preprocess_text(input_text, stopwords)
    text_vector = vectorizer.transform([processed_text])
    predicted_label_index = model.predict(text_vector)[0]
    predicted_group = label_encoder.inverse_transform([predicted_label_index])[0]
    probabilities = model.predict_proba(text_vector)[0]
    confidence = max(probabilities)
    return predicted_group, confidence, probabilities
# Thử nghiệm
def testing(model, vectorizer, label_encoder, stopwords):
    print("\n" + "="*60)
    while True:
        user_input = input("\nNhập mô tả triệu chứng (hoặc 'quit' để thoát): ")
        if user_input.lower() in ['quit', 'exit', 'thoát']:
            break
        predicted_group, confidence, probabilities = predict_disease_group(user_input, model, vectorizer, label_encoder, stopwords)
        top_indices = probabilities.argsort()[-3:][::-1]
        print("Top 3 nhóm bệnh bạn có khả năng mắc cao nhất:")
        for i, idx in enumerate(top_indices, 1):
            group_name = label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx]
            print(f"{i}. {group_name}: \nXác suất: {prob:.2%}")
        print("Lưu ý: Đây chỉ là chuẩn đoán tham khảo, bạn nên tham khảo ý kiến bác sĩ để có chẩn đoán chính xác.")
    print("\nCảm ơn bạn đã sử dụng hệ thống dự đoán nhóm bệnh!")

if __name__ == "__main__":
    group_col = 'DiseaseCategory'
    feature_col = 'Question'
    PROCESSED_feature_col = f'{feature_col}_processed'
    ENCODED_group_col = f'{group_col}_encoded'

    df = load_data("hf://datasets/joon985/ViMedical_Disease_Category/ViMedicalDiseaseCategory.csv")
    check_data(df, group_col, feature_col)

    vietnamese_stopwords = get_vietnamese_stopwords()
    label_encoder = LabelEncoder()
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    df = encode_labels(df, group_col, ENCODED_group_col, label_encoder)
    df = process_dataset(df, group_col, feature_col, PROCESSED_feature_col, vietnamese_stopwords)
    X_vec = vectorizer.fit_transform(df[PROCESSED_feature_col])
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_vec, df[ENCODED_group_col])

    model = train_model(X_train, y_train, X_val, y_val)
    test_accuracy, test_f1 = evaluate_model(model, X_test, y_test, label_encoder)
    testing(model, vectorizer, label_encoder, vietnamese_stopwords)

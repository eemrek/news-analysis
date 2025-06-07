# Haber Başlığı ve Açıklamasına Göre Çok Sınıflı Metin Sınıflandırma

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix ,roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Veri kümesini yükle
file_path = "./ag_news.csv"
df = pd.read_csv(file_path)

# Sınıf etiketlerini 0-3 aralığına çek
class_labels = sorted(df['Class Index'].unique())
label_to_index = {label: idx for idx, label in enumerate(class_labels)}
df['label'] = df['Class Index'].map(label_to_index)

# Metinleri oluştur: başlık + açıklama
df['text'] = df['Title'].fillna('') + " " + df['Description'].fillna('')

# Tokenizasyon ve sıralama
MAX_VOCAB = 10000
MAX_LEN = 200
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# Etiketleri one-hot encode et
labels = to_categorical(df['label'])

# Eğitim ve test ayırımı
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, stratify=labels, random_state=42)

# Model
model = Sequential([
    Embedding(MAX_VOCAB, 128, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_proba = model.predict(X_test)

# Eğitim
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Değerlendirme
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")



# Confusion Matrix ve Sınıflandırma Raporu
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true_classes, y_pred_classes))

conf_mat = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Eğitim eğrileri
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.show()

#ROC egrisi
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 3  # Sınıf sayısı

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
class_names = ['World', 'Sports', 'Business']

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

# Görselde referans için diagonal çizgi (rastgele sınıflandırıcı)
plt.plot([0, 1], [0, 1], 'k--', lw=1.5)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Çok Sınıflı ROC Eğrisi (One-vs-Rest)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()
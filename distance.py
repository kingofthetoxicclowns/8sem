from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from tkinter.filedialog import asksaveasfilename, askopenfilename
import tkinter as tk

def str_to_trigram(s):
    s = s.rstrip()
    s = s.lower()
    s = ''.join([letter for letter in s if letter.isalpha()])
    s = ["".join(j) for j in zip(*[s[i:] for i in range(3)])]
    s = ' '.join(s)
    return s

root = tk.Tk()
root.withdraw()
file_first = askopenfilename(title="Выберите первый файл", filetypes=(("text file", "*.txt"),))
with open(file_first, encoding='utf-8') as f:
    text1 = f.readlines()
file_second = askopenfilename(title="Выберите второй файл", filetypes=(("text file", "*.txt"),))
with open(file_second, encoding='utf-8') as f:
    text2 = f.readlines()

tfidf_vectorizer = TfidfVectorizer(analyzer='word')
vectors = tfidf_vectorizer.fit_transform([str_to_trigram(i) for i in text1] +
                                         [str_to_trigram(i) for i in text2])
result = []
print('Обработка данных')
for indx, s in enumerate(text1):
    cosines = cosine_similarity(vectors[indx, :], vectors[len(text1):, :])
    result.append(f'{text1[indx].rstrip()} {text2[cosines.argmax()].rstrip()} {cosines.max()}')
file_save = asksaveasfilename(title="Запись результата по косинусному расстоянию в файл", filetypes=(("Text files", "*.txt"),))
if file_save:
    with open(file_save + '.txt', 'w', encoding='utf-8') as f:
        for line in result:
            f.write(f"{line}\n")

print('Обработка данных')
for indx, s in enumerate(text1):
    distances = euclidean_distances(vectors[indx, :], vectors[len(text1):, :])
    result.append(f'{text1[indx].rstrip()} {text2[distances.argmin()].rstrip()} {distances.min()}')
file_save = asksaveasfilename(title="Запись результата по евклидовому расстоянию в файл", filetypes=(("Text files", "*.txt"),))
if file_save:
    with open(file_save + '.txt', 'w', encoding='utf-8') as f:
        for line in result:
            f.write(f"{line}\n")
root.destroy()

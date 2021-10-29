#ISSUES TO BE SOLVED
#1 Show one sentiment and one emotion value as a result
#2 Show one sentiment and more than one emotiion values as a result
#3 Change the emotion algorithm logic, cause accuracy is so low
#4 Label neutral for all_final
#5 Label neutral and emotions for 1-8000
#6 Label neutral and emotions for 8001-end
#7 Double check labelled sentiments and emotions for all_lexicon
#8 Add edge cases
#9 Add entity recognition
#10 Note: There is 40k+ fully unlabelled data in all_final of Vafa in case it is needed. 

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

import pickle
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

vectorizer = CountVectorizer() 

#BOW1-1-8000
news1 = pd.read_csv("1-8000.csv", low_memory=False)

#BOW
#news1 = pd.read_csv("all_final.csv", low_memory=False)

#LD
#news1 = pd.read_csv("all_lexicon.csv", low_memory=False)

neg1 = news1[news1.negative == 'neg']
pos1 = news1[news1.positive == 'pos']
neu1 = news1[news1.neutral == 'neu']
ang1 = news1[news1.anger == 'ang']
ant1 = news1[news1.anticipation == 'ant']
dis1 = news1[news1.disgust == 'dis']
fea1 = news1[news1.fear == 'fea']
joy1 = news1[news1.joy == 'joy']
sad1 = news1[news1.sadness == 'sad']
sur1 = news1[news1.surprise == 'sur']
tru1 = news1[news1.trust == 'tru']

negative = list(neg1.content) 
positive = list(pos1.content)
neutral = list(neu1.content)
anger = list(ang1.content) 
anticipation = list(ant1.content)
disgust = list(dis1.content) 
fear = list(fea1.content)
joy = list(joy1.content) 
sadness = list(sad1.content)
surprise = list(sur1.content) 
trust = list(tru1.content) 

mylist1 = list(dict.fromkeys(negative))
mylist2 = list(dict.fromkeys(positive))
mylist3 = list(dict.fromkeys(neutral))
mylist4 = list(dict.fromkeys(anger))
mylist5 = list(dict.fromkeys(anticipation))
mylist6 = list(dict.fromkeys(disgust))
mylist7 = list(dict.fromkeys(fear))
mylist8 = list(dict.fromkeys(joy))
mylist9 = list(dict.fromkeys(sadness))
mylist10 = list(dict.fromkeys(surprise))
mylist11 = list(dict.fromkeys(trust))

neglabels = ['negative' for l in range(len(mylist1))]
poslabels = ['positive' for l in range(len(mylist2))]
neulabels = ['neutral' for l in range(len(mylist3))]
anglabels = ['anger' for l in range(len(mylist4))]
antlabels = ['anticipation' for l in range(len(mylist5))]
dislabels = ['disgust' for l in range(len(mylist6))]
fealabels = ['fear' for l in range(len(mylist7))]
joylabels = ['joy' for l in range(len(mylist8))]
sadlabels = ['sadness' for l in range(len(mylist9))]
surlabels = ['surprise' for l in range(len(mylist10))]
trulabels = ['trust' for l in range(len(mylist11))]

#only sentiments without neutral (Accuracy: 80-90 for BOW, 55 for LD)
#news = mylist1 + mylist2
#labels = neglabels + poslabels

#only emotions (Accuracy: 0-10 for BOW, 0-1 for LD)
news = mylist4 + mylist5 + mylist6 + mylist7 + mylist8 + mylist9 + mylist10 + mylist11
labels = anglabels + antlabels + dislabels + fealabels + joylabels + sadlabels + surlabels  + trulabels

news=np.array(news)
labels=np.array(labels)

train_corpus, test_corpus, train_labels, test_labels = train_test_split(news, labels,test_size=0.1)

labels.reshape(-1,1)

vectorized_train = vectorizer.fit_transform(train_corpus)
vectorized_test = vectorizer.transform(test_corpus)

vectorized_test.shape


mlp=MLPClassifier()

mlp.fit(vectorized_train,train_labels)

with open('model.pickle', 'wb') as f:
    pickle.dump(mlp, f)
    
with open('vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model.pickle', 'rb') as f:
    mlp = pickle.load(f)
    
with open('vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)
    
predictions=mlp.predict(vectorized_test)
l = sorted(set(labels))
print('Accuracy')
accuracy = np.sum(predictions == test_labels) / len(test_labels)
print(accuracy)
print('Confusion Matrix and Classification Report')
##print(metrics.classification_report(test_labels, predictions, target_names=l))
##print(metrics.confusion_matrix(test_labels, predictions, labels=l))


class SentimentApp(tk.Frame):
    APP_TITLE = "Sentiment App"
    ICON_FILE = "question.ico"
    PACK_FILL_EXPAND = {"expand": True, "fill": tk.BOTH}
    PLACE_MIDDLE = {"anchor": tk.N, "relx": 0.5}
    TRANSLATE = {"positive": "pozitiv", "negative": "neqativ","neutral": "neytral", "anger": "əsəbi", "anticipation": "təxminetmə", "disgust": "iyrənc", "fear": "qorxu", "joy": "sevincli", "sadness": "qəmgin", "surprise": "təəcüblənmiş", "trust": "günənli"}
    TEXTFONT = ("Arial", 12)
    APP_BACKGROUND = "#6D6F90"
    HEADER_TEXT = "Aşağıdakı qutuya mətn girin və onun pozitiv və ya neqativ olduğunu öyrənin!"

    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.pack(**SentimentApp.PACK_FILL_EXPAND)
        self.configure_master()
        self.configure_styling()
        self.create_widgets()

    def configure_master(self):
        self.master.title(SentimentApp.APP_TITLE)
        self.master.iconbitmap(SentimentApp.ICON_FILE)
        self.master.geometry("600x600")
        self.master.resizable(False, False)

    def configure_styling(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TButton", font=SentimentApp.TEXTFONT)
        self.style.configure("TLabel", font=SentimentApp.TEXTFONT, background=SentimentApp.APP_BACKGROUND)

    def create_widgets(self):
        self.mainframe = tk.LabelFrame(self, bg=SentimentApp.APP_BACKGROUND)
        self.mainframe.place(relx=0, rely=0, relheight=1, relwidth=1)

        self.header = ttk.Label(self.mainframe, text=SentimentApp.HEADER_TEXT)
        self.header.place(**SentimentApp.PLACE_MIDDLE, rely=0.2)

        self.sc_y = ttk.Scrollbar(self.mainframe, orient=tk.VERTICAL)
        self.text_input = tk.Text(self.mainframe, width=52, height=10, yscrollcommand=self.sc_y.set, font=SentimentApp.TEXTFONT)
        self.sc_y.config(command=self.text_input.yview)
        self.sc_y.place(relx=0.8984, rely=0.40016, relheight=0.308)
        self.text_input.place(**SentimentApp.PLACE_MIDDLE, rely=0.4)

        self.submit_button = self.label_button = ttk.Button(self.mainframe, takefocus=0, text="Submit",
                                                            command=self.determine_text_sentiment)
        self.submit_button.place(**SentimentApp.PLACE_MIDDLE, rely=0.8)

    def determine_text_sentiment(self):
        user_input = self.text_input.get("1.0", tk.END)
        with open('model.pickle', 'rb') as f:
            mlp = pickle.load(f)
        result = mlp.predict(vectorizer.transform([user_input]))
        messagebox.showinfo("Nəticə", f"Sizin xəbərinizin tipi: {SentimentApp.TRANSLATE[result[0]]}")


def main():
    root = tk.Tk()
    app = SentimentApp(root)
    app.mainloop()


if __name__ == "__main__":
    main()




#ISSUES TO BE SOLVED

    #BOW
        #Data Size: 3000
        #Sent_Accuracy(wit neutral): 70-78
        #Sent_Accuracy(without neutral): 72-100
        #Emot_Accuracy: 0-24
        
    #LD
        #Data Size: 14000
        #Sent_Accuracy: 50-55
        #Emot_Accuracy: 0-1


#1 DONE Show one sentiment and one emotion value as a result
#2 DONE Show all sentiment and emotion values and percentages as a result

    #BOW
        #Data Size: 3000
        #Sent_Accuracy(wit neutral): 70-78
        #Sent_Accuracy(without neutral): 72-100
        #Emot_Accuracy: 31-35 !!! 
        
    #LD
        #Data Size: 14000
        #Sent_Accuracy: 50-55
        #Emot_Accuracy: 0-1

#3 Add another dictionary to the LD  
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


data = pd.read_csv("all_final_3800.csv", low_memory=False) #BOW
#data = pd.read_csv("all_lexicon.csv", low_memory=False) #LD

pos = data[data.positive == 'pos']
neg = data[data.negative == 'neg']
neu = data[data.neutral == 'neu']
ang = data[data.anger == 'ang']
ant = data[data.anticipation == 'ant']
dis = data[data.disgust == 'dis']
fea = data[data.fear == 'fea']
joy = data[data.joy == 'joy']
sad = data[data.sadness == 'sad']
sur = data[data.surprise == 'sur']
tru = data[data.trust == 'tru']

positive = list(pos.content)
negative = list(neg.content) 
neutral = list(neu.content)
anger = list(ang.content) 
anticipation = list(ant.content)
disgust = list(dis.content) 
fear = list(fea.content)
joy = list(joy.content) 
sadness = list(sad.content)
surprise = list(sur.content) 
trust = list(tru.content) 

mylist1 = list(dict.fromkeys(positive))
mylist2 = list(dict.fromkeys(negative))
mylist3 = list(dict.fromkeys(neutral))
mylist4 = list(dict.fromkeys(anger))
mylist5 = list(dict.fromkeys(anticipation))
mylist6 = list(dict.fromkeys(disgust))
mylist7 = list(dict.fromkeys(fear))
mylist8 = list(dict.fromkeys(joy))
mylist9 = list(dict.fromkeys(sadness))
mylist10 = list(dict.fromkeys(surprise))
mylist11 = list(dict.fromkeys(trust))

poslabels = ['positive' for l in range(len(mylist1))]
neglabels = ['negative' for l in range(len(mylist2))]
neulabels = ['neutral' for l in range(len(mylist3))]
anglabels = ['anger' for l in range(len(mylist4))]
antlabels = ['anticipation' for l in range(len(mylist5))]
dislabels = ['disgust' for l in range(len(mylist6))]
fealabels = ['fear' for l in range(len(mylist7))]
joylabels = ['joy' for l in range(len(mylist8))]
sadlabels = ['sadness' for l in range(len(mylist9))]
surlabels = ['surprise' for l in range(len(mylist10))]
trulabels = ['trust' for l in range(len(mylist11))]

vectorizer_sent = CountVectorizer()
vectorizer_emot = CountVectorizer() 

news_sent = mylist1 + mylist2 
labels_sent = neglabels + poslabels 
news_emot = mylist4 + mylist5 + mylist6 + mylist7 + mylist8 + mylist9 + mylist10 + mylist11
labels_emot = anglabels + antlabels + dislabels + fealabels + joylabels + sadlabels + surlabels  + trulabels

news_sent=np.array(news_sent)
labels_sent=np.array(labels_sent)
news_emot=np.array(news_emot)
labels_emot=np.array(labels_emot)

train_corpus_sent, test_corpus_sent, train_labels_sent, test_labels_sent = train_test_split(news_sent, labels_sent,test_size=0.4)
train_corpus_emot, test_corpus_emot, train_labels_emot, test_labels_emot = train_test_split(news_emot, labels_emot,test_size=0.4)

labels_sent.reshape(-1,1)
labels_emot.reshape(-1,1)

vectorized_train_sent = vectorizer_sent.fit_transform(train_corpus_sent)
vectorized_test_sent = vectorizer_sent.transform(test_corpus_sent)
vectorized_test_sent.shape
vectorized_train_emot = vectorizer_emot.fit_transform(train_corpus_emot)
vectorized_test_emot = vectorizer_emot.transform(test_corpus_emot)
vectorized_test_emot.shape

mlp_sent=MLPClassifier()
mlp_sent.fit(vectorized_train_sent,train_labels_sent)
mlp_emot=MLPClassifier()
mlp_emot.fit(vectorized_train_emot,train_labels_emot)

with open('model.pickle', 'wb') as f:
    pickle.dump(mlp_sent, f)
    pickle.dump(mlp_emot, f)
    
with open('vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer_sent, f)
    pickle.dump(vectorizer_emot, f)

with open('model.pickle', 'rb') as f:
    mlp_sent = pickle.load(f)
    mlp_emot = pickle.load(f)
    
with open('vectorizer.pickle', 'rb') as f:
    vectorizer_sent = pickle.load(f)
    vectorizer_emot = pickle.load(f)
    
predictions_sent=mlp_sent.predict(vectorized_test_sent)
predictions_emot=mlp_emot.predict(vectorized_test_emot)

l_sent = sorted(set(labels_sent))
l_emot = sorted(set(labels_emot))

print('Accuracy Sentiment')
accuracy_sent = np.sum(predictions_sent == test_labels_sent) / len(test_labels_sent)
print(accuracy_sent)

print('Accuracy Emotion')
accuracy_emot = np.sum(predictions_emot == test_labels_emot) / len(test_labels_emot)
print(accuracy_emot)


class SentimentApp(tk.Frame):
    APP_TITLE = "Sentiment App"
    ICON_FILE = "question.ico"
    PACK_FILL_EXPAND = {"expand": True, "fill": tk.BOTH}
    PLACE_MIDDLE = {"anchor": tk.N, "relx": 0.5}
    TRANSLATE = {"positive": "pozitiv", "negative": "neqativ","neutral": "neytral", "anger": "əsəbi", "anticipation": "təxminetmə", "disgust": "iyrənc", "fear": "qorxu", "joy": "sevincli", "sadness": "qəmgin", "surprise": "təəcüblənmiş", "trust": "güvənli"}
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
            mlp_sent = pickle.load(f)
            mlp_emot = pickle.load(f)
        result_sent = mlp_sent.predict(vectorizer_sent.transform([user_input]))
        result_emot = mlp_emot.predict(vectorizer_emot.transform([user_input]))
        
        prob_sent = mlp_sent.predict_proba(vectorizer_sent.transform([user_input]))
        prob_emot = mlp_emot.predict_proba(vectorizer_emot.transform([user_input]))

        prob_pos = prob_sent[0][0]

        prob_neg = prob_sent[0][1]

        #prob_neu = prob_sent[0][2]
        prob_neu = 0

        prob_ang = prob_emot[0][0]
        prob_ant = prob_emot[0][1]
        prob_dis = prob_emot[0][2]
        prob_fea = prob_emot[0][3]
        prob_joy = prob_emot[0][4]
        prob_sad = prob_emot[0][5]
        prob_sur = prob_emot[0][6]
        prob_tru = prob_emot[0][7]

        messagebox.showinfo("Nəticə",
                            f"Sentiment dəyərləri\nPozitiv: {round(prob_pos, 2)}\nNeqativ: {round(prob_neg, 2)}\nNeytral: {round(prob_neu, 2)}\n\nEmosiya dəyərləri\nƏsəbi: {round(prob_ang, 2)}\nTəxminetmə: {round(prob_ant, 2)}\nİyrənc: {round(prob_dis, 2)}\nQorxunc: {round(prob_fea, 2)}\nSevincli: {round(prob_joy, 2)}\nQəmgin: {round(prob_sad, 2)}\nTəəccüblü: {round(prob_sur, 2)}\nGüvənverici: {round(prob_tru, 2)}")
    

def main():
    root = tk.Tk()
    app = SentimentApp(root)
    app.mainloop()


if __name__ == "__main__":
    main()

# import stanza # type: ignore
import numpy as np
import pandas as pd
import seaborn as sns
# import xgboost as xgb # type: ignore
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# from stanza.pipeline.core import DownloadMethod # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

class email_classifier:
    X = None
    y_test = None
    X_test = None
    X_train = None
    y_train = None
    classifier = None

    def __init__(self, filename):
        # Use Case 1: Load and Preprocess Data
        # df = pd.read_csv("AppGallery.csv")
        self.df = pd.read_csv(filename)

    def proprocess_dataset(self):
        # Preprocess the dataset
        self.df['Interaction content'] = self.df['Interaction content'].values.astype('U')
        self.df['Ticket Summary'] = self.df['Ticket Summary'].values.astype('U')
        self.df["y1"] = self.df["Type 1"]
        self.df["y2"] = self.df["Type 2"]
        self.df["y3"] = self.df["Type 3"]
        self.df["y4"] = self.df["Type 4"]
        self.df["x"] = self.df['Interaction content']
        self.df["y"] = self.df["y2"]

        # Remove empty y
        self.df = self.df.loc[(self.df["y"] != '') & (~self.df["y"].isna()),]
        print(self.df.shape)

    def remove_noise(self):
        # Remove noise from Ticket Summary
        noise = "(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
        self.df["ts"] = self.df["Ticket Summary"].str.lower().replace(noise, " ", regex=True).replace(r'\s+', ' ', regex=True).str.strip()
        temp_debug = self.df.loc[:, ["Ticket Summary", "ts", "y"]]

        self.df["ic"] = self.df["Interaction content"].str.lower()   
        # Remove noise from Interaction Content
        noise_1 = [
            "(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
            "(january|february|march|april|may|june|july|august|september|october|november|december)",
            "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
            "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            "\d{2}(:|.)\d{2}",
            "(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
            "dear ((customer)|(user))",
            "dear",
            "(hello)|(hallo)|(hi )|(hi there)",
            "good morning",
            "thank you for your patience ((during (our)? investigation)|(and cooperation))?",
            "thank you for contacting us",
            "thank you for your availability",
            "thank you for providing us this information",
            "thank you for contacting",
            "thank you for reaching us (back)?",
            "thank you for patience",
            "thank you for (your)? reply",
            "thank you for (your)? response",
            "thank you for (your)? cooperation",
            "thank you for providing us with more information",
            "thank you very kindly",
            "thank you( very much)?",
            "i would like to follow up on the case you raised on the date",
            "i will do my very best to assist you",
            "in order to give you the best solution",
            "could you please clarify your request with following information:",
            "in this matter",
            "we hope you(( are)|('re)) doing ((fine)|(well))",
            "i would like to follow up on the case you raised on",
            "we apologize for the inconvenience",
            "sent from my huawei (cell )?phone",
            "original message",
            "customer support team",
            "(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
            "(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
            "canada, australia, new zealand and other countries",
            "\d+",
            "[^0-9a-zA-Z]+",
            "(\s|^).(\s|$)"
        ]

        for noise in noise_1:
            print(noise)
            self.df["ic"] = self.df["ic"].replace(noise, " ", regex=True)

        self.df["ic"] = self.df["ic"].replace(r'\s+', ' ', regex=True).str.strip()
        temp_debug = self.df.loc[:, ["Interaction content", "ic", "y"]]

        print(self.df.y1.value_counts())
        good_y1 = self.df.y1.value_counts()[self.df.y1.value_counts() > 10].index
        temp = self.df.loc[self.df.y1.isin(good_y1)]
        print(temp.shape)

    # Use Case 2: Translate Text to English
    def trans_to_en(self, texts):
        # Translate to English if necessary
        t2t_m = "facebook/m2m100_418M"
        t2t_pipe = pipeline(task='text2text-generation', model=t2t_m)

        model = M2M100ForConditionalGeneration.from_pretrained(t2t_m)
        tokenizer = M2M100Tokenizer.from_pretrained(t2t_m)
        nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid", download_method=DownloadMethod.REUSE_RESOURCES)

        text_en_l = []
        for text in texts:
            if text == "":
                text_en_l.append(text)
                continue

            doc = nlp_stanza(text)
            print(doc.lang)
            if doc.lang == "en":
                text_en_l.append(text)
            else:
                lang = doc.lang
                if lang == "fro":  # Old French
                    lang = "fr"
                elif lang == "la":  # Latin
                    lang = "it"
                elif lang == "nn":  # Norwegian (Nynorsk)
                    lang = "no"
                elif lang == "kmr":  # Kurmanji
                    lang = "tr"

                case = 2
                if case == 1:
                    text_en = t2t_pipe(text, forced_bos_token_id=t2t_pipe.tokenizer.get_lang_id(lang='en'))
                    text_en = text_en[0]['generated_text']
                elif case == 2:
                    tokenizer.src_lang = lang
                    encoded_hi = tokenizer(text, return_tensors="pt")
                    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"))
                    text_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    text_en = text_en[0]
                else:
                    text_en = text

                text_en_l.append(text_en)

                print(text)
                print(text_en)

        return text_en_l

    def prepare_data(self):
        global X_test
        global y_test
        global X_train
        global y_train
        # Use Case 4: Split Data into Training and Testing Sets
        # Prepare training and testing data
        y = self.df.y.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # remove bad test cases from test dataset
        Test_size = 0.20
        y_series = pd.Series(y)
        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
        y_good = y[y_series.isin(good_y_value)]
        X_good = X[y_series.isin(good_y_value)]
        y_bad = y[y_series.isin(good_y_value) == False]
        X_bad = X[y_series.isin(good_y_value) == False]
        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]
        print(f"new_test_size: {new_test_size}")
        X_train, X_test, y_train, y_test = train_test_split(X_good, y_good, test_size=new_test_size, random_state=0)
        X_train = np.concatenate((X_train, X_bad), axis=0)
        y_train = np.concatenate((y_train, y_bad), axis=0)

    def vectorize_text(self):
        # Translate Ticket Summary to English
        self.df["ts_en"] = self.trans_to_en(self.df["ts"].to_list())

        # Use Case 3: Feature Extraction
        # Feature extraction
        tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
        x1 = tfidfconverter.fit_transform(self.df["ic"]).toarray()
        x2 = tfidfconverter.fit_transform(self.df["ts_en"]).toarray()
        global X
        X = np.concatenate((x1, x2), axis=1)

    def model_creation_and_training(self):
        global classifier
        classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
        classifier.fit(X_train, y_train)

    def testing_model(self):
        # Use Case 6: Evaluate the Model
        # Evaluate the model
        y_pred = classifier.predict(X_test)
        p_result = pd.DataFrame(classifier.predict_proba(X_test))
        p_result.columns = classifier.classes_
        print(p_result)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

        # Visualize Confusion Matrix
        # # Encoding target variable 'y'
        label_encoder = LabelEncoder()
        self.df['y_encoded'] = label_encoder.fit_transform(self.df['y'])

        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # Visualize Performance Metrics
        metrics = {
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall
        }

        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

        plt.figure(figsize=(10, 7))
        sns.barplot(x='Metric', y='Value', data=metrics_df)
        plt.ylim(0, 1)
        plt.title('Performance Metrics')
        plt.show()
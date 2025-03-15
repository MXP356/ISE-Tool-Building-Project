## CONFIGURATION:
project = 'pytorch' # either 'pytorch' or 'tensorflow', 'keras', 'incubator-mxnet' or 'caffe'.
REPEAT_TIMES = [5, 10, 30]
IncludeCommentsFromBugReports = False
IncludeCodeSnippetsAndErrorLogsFromBugReports = True
IncludeLabelsFromBugReports = False
Method = 'BERTLogisticRegression' # either 'TFIDFNaiveBayes' or 'BERTLogisticRegression'


RemoveHTMLTags = True
RemoveEmoji = True
RemoveStopWords = True
CleanString = True

UseGridSearchCVForBERTLR = False



USE_LAB1_BASELINE_CONFIG = False # will override all of above configs (except for 'project') to match the Lab 1 baseline. 
if USE_LAB1_BASELINE_CONFIG:
    project = project
    IncludeCommentsFromBugReports = False
    IncludeCodeSnippetsAndErrorLogsFromBugReports = False
    IncludeLabelsFromBugReports = False
    Method = 'TFIDFNaiveBayes'
    RemoveHTMLTags = True
    RemoveEmoji = True
    RemoveStopWords = True
    CleanString = True
    UseGridSearchCVForBERTLR = False





########## 1. Import required libraries ##########
print('Importing libraries...')
import pandas as pd
import numpy as np
import re
import math
import torch

# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel

if Method == 'BERTLogisticRegression':
    print('Getting Pretrained BERT Model...')
    bertPretrainedModelName = 'bert-base-uncased'
    bertTokenizer = BertTokenizer.from_pretrained(bertPretrainedModelName)
    bertModel = BertModel.from_pretrained(bertPretrainedModelName)

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)

# Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Text cleaning & stopwords
if RemoveStopWords:
    print('Downloading Stopwords...')
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    # Stopwords
    NLTK_stop_words_list = stopwords.words('english')
    custom_stop_words_list = ['...']  # You can customize this list as needed
    final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list





########## 2. Define text preprocessing methods ##########
def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)



def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

########## 3. Download & read data ##########
# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
print('Reading Dataset CSV File...')
path = f'datasets/{project}.csv'

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle



def GetTextToAnalyzeFromRow(row):
    text = row['Title']

    if pd.notna(row['Body']):
        text += '. ' + row['Body']

    if IncludeCommentsFromBugReports and row['Comments'] and pd.notna(row['Comments']):
        text += '. ' + row['Comments']

    if IncludeCodeSnippetsAndErrorLogsFromBugReports and row['Codes'] and pd.notna(row['Codes']):
        text += '. ' + row['Codes']

    if IncludeLabelsFromBugReports and row['Labels'] and pd.notna(row['Labels']):
        text += '. ' + row['Labels']

    return text


print('Fetching Text To Analyze...')

pd_all['TextToAnalyze'] = pd_all.apply(
    GetTextToAnalyzeFromRow,
    axis=1
)

# Keep only necessary columns: id, Number, sentiment, text (merged Title+Body + Comments if specified)
pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "TextToAnalyze": "text"
})

DataFileName = 'storage/TextToAnalyze.csv'
pd_tplusb.to_csv(DataFileName, index=False, columns=["id", "Number", "sentiment", "text"])

########## 4. Configure parameters & Start training ##########

# ========== Key Configurations ==========



# 3) Output CSV file name
out_csv_name = f'outputs/{project}{(Method == 'TFIDFNaiveBayes' and '_NB') or (Method == 'GloVeLogisticRegression' and '_GloVe-LR') or (Method == 'BERTLogisticRegression' and '_BERT-LR')}{Method == 'BERTLogisticRegression' and UseGridSearchCVForBERTLR and '-With-GridSearchCV' or ''}{IncludeCommentsFromBugReports and '_CommentsIncluded' or ''}{IncludeCodeSnippetsAndErrorLogsFromBugReports and '_CodeSnippetsAndErrorLogsIncluded' or ''}{IncludeLabelsFromBugReports and '_LabelsIncluded' or ''}{RemoveStopWords == False and '_KeepStopwords' or ''}{RemoveEmoji == False and '_KeepEmoji' or ''}{RemoveHTMLTags == False and '_KeepHTMLTags' or ''}{CleanString == False and '_DontCleanString' or ''}.csv'

# ========== Read and clean data ==========
data = pd.read_csv(DataFileName).fillna('')
text_col = 'text'

# Keep a copy for referencing original data if needed
original_data = data.copy()

print('Preprocessing/Cleaning Text...')

# Text cleaning
if RemoveHTMLTags:
    data[text_col] = data[text_col].apply(remove_html)
if RemoveEmoji:
    data[text_col] = data[text_col].apply(remove_emoji)

if RemoveStopWords:
    data[text_col] = data[text_col].apply(remove_stopwords)

if CleanString:
    data[text_col] = data[text_col].apply(clean_str)



# Lists to store metrics across repeated runs
accuracies  = []
precisions  = []
recalls     = []
f1_scores   = []
auc_values  = []




## BERT EMBEDDINGS CACHE:
if Method == 'BERTLogisticRegression':
    print('Loading Cached BERT Word Embedings...')
    import pickle
    BERT_EMBEDDINGS_CACHE_FILE = 'storage/cached_bert_embeddings.pkl'

    def load_cached_embeddings():
        try:
            with open(BERT_EMBEDDINGS_CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {}


    cachedBERTEmbeddings = load_cached_embeddings()

    def save_cached_embeddings(cache):
        with open(BERT_EMBEDDINGS_CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)



## Store our model in this variable here for use in the Tkinter UI later
model = None

print('Training & Testing Model...')

for repeated_time in range(max(REPEAT_TIMES)):
    REPEAT_NUMBER = repeated_time + 1
    print('REPEAT NUMBER:',REPEAT_NUMBER)
    # --- 4.1 Split into train/test ---
    indices = np.arange(data.shape[0]) ## Notes: data.shape returns a tuple representing the dimensions of the dataframe, and then [0] gives us the number of rows. np.arange(n) gives us an array [0, 1, 2, ... , n-2, n-1]. So it basically creates an array containing indexes of each row.
    train_index, test_index = train_test_split(
        indices, test_size=0.2, random_state=repeated_time
    )

    train_text = data[text_col].iloc[train_index]
    test_text = data[text_col].iloc[test_index]

    y_train = data['sentiment'].iloc[train_index]
    y_test  = data['sentiment'].iloc[test_index]


    if Method == 'TFIDFNaiveBayes':

        # --- 4.2 TF-IDF vectorization ---
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000  # Adjust as needed
        )
        X_train = tfidf.fit_transform(train_text).toarray()
        X_test = tfidf.transform(test_text).toarray()
    

        # ========== Hyperparameter grid ==========
        # We use logspace for var_smoothing: [1e-12, 1e-11, ..., 1]
        params = {
            'var_smoothing': np.logspace(-12, 0, 13)
        }

        # --- 4.3 Naive Bayes model & GridSearch ---
        clf = GaussianNB()
        grid = GridSearchCV(
            clf,
            params,
            cv=5,              # 5-fold CV (can be changed)
            scoring='roc_auc'  # Using roc_auc as the metric for selection
        )
        grid.fit(X_train, y_train)

        # Retrieve the best model
        best_clf = grid.best_estimator_
        best_clf.fit(X_train, y_train)

        # --- 4.4 Make predictions & evaluate ---
        y_pred = best_clf.predict(X_test)
        model = best_clf
    elif Method == 'BERTLogisticRegression':
        ## TODO:
        ## Create word embeddings from BERT
        ## and then 
        ## classify with logistic regression

        def CreateWordEmbeddings(texts, setName):
            embeddings = []
            #batchSize = 1
            
            for i in range(0, len(texts)):#, batchSize):
                

                textToCreateEmbeddingFor = texts[i]
                if textToCreateEmbeddingFor in cachedBERTEmbeddings:
                    embeddings.append(cachedBERTEmbeddings[textToCreateEmbeddingFor])
                    #print('Got BERT embedding from cache!')
                else: 
                    print('Creating BERT word embeddings for bug report number:',i,'/',len(texts),setName)
                    inputsForModel = bertTokenizer(textToCreateEmbeddingFor, return_tensors='pt', padding=True, truncation=True, max_length=512) #texts[i:i+batchSize]
                    
                    with torch.no_grad(): ## I had to do this to stop my PC's memory going to 100%
                        modelOutput = bertModel(**inputsForModel)

                    CLSTokenEmbedding = modelOutput.last_hidden_state[:,0,:].squeeze(0).numpy() ## thjis is so we get the CLS token for each bug report only [CLS token is used for classification]. had to add .squeeze() because it was giving me a 3D array when extracting the CLS embedding, which made logistic regression classifier error
                    #print('CLS Embedding:',CLSEmbedding)
                    cachedBERTEmbeddings[textToCreateEmbeddingFor] = CLSTokenEmbedding ## so that future REPEATs do not have to re-calculate the embeddings again as it takes a long time
                    embeddings.append(CLSTokenEmbedding)
            
            return embeddings
        
        trainingWordEmbeddings = CreateWordEmbeddings(train_text.to_list(),'[Training Set]')
        testingWordEmbeddings = CreateWordEmbeddings(test_text.to_list(),'[Testing Set]')

        save_cached_embeddings(cachedBERTEmbeddings)


        # now classify:

        if UseGridSearchCVForBERTLR:
            clf = LogisticRegression(max_iter=1000)
            params = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs', 'newton-cg'],
            }
            grid = GridSearchCV(
                clf,
                params,
                cv=5,              # 5-fold CV (can be changed)
                scoring='roc_auc'  # Using roc_auc as the metric for selection
            )
            grid.fit(trainingWordEmbeddings, y_train)

            # Retrieve the best model
            best_clf = grid.best_estimator_
            print('Best Params for BERT-LR:', grid.best_params_)
            best_clf.fit(trainingWordEmbeddings, y_train)

            y_pred = best_clf.predict(testingWordEmbeddings)
            model = best_clf
        else:
            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(trainingWordEmbeddings, y_train)
            y_pred = classifier.predict(testingWordEmbeddings)
            model = classifier
        





    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # Precision (macro)
    prec = precision_score(y_test, y_pred, average='macro')
    precisions.append(prec)

    # Recall (macro)
    rec = recall_score(y_test, y_pred, average='macro')
    recalls.append(rec)

    # F1 Score (macro)
    f1 = f1_score(y_test, y_pred, average='macro')
    f1_scores.append(f1)

    # AUC
    # If labels are 0/1 only, this works directly.
    # If labels are something else, adjust pos_label accordingly.
    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    auc_val = auc(fpr, tpr)
    auc_values.append(auc_val)



    if (REPEAT_NUMBER in REPEAT_TIMES):
        # --- 4.5 Aggregate results ---
        final_accuracy  = np.mean(accuracies)
        final_precision = np.mean(precisions)
        final_recall    = np.mean(recalls)
        final_f1        = np.mean(f1_scores)
        final_auc       = np.mean(auc_values)

        print("=== Experiment Results ===")
        print(f"Number of repeats:     {REPEAT_NUMBER}")
        print(f"Average Accuracy:      {final_accuracy:.4f}")
        print(f"Average Precision:     {final_precision:.4f}")
        print(f"Average Recall:        {final_recall:.4f}")
        print(f"Average F1 score:      {final_f1:.4f}")
        print(f"Average AUC:           {final_auc:.4f}")

        # Save final results to CSV (append mode)
        try:
            # Attempt to check if the file already has a header
            existing_data = pd.read_csv(out_csv_name, nrows=1)
            header_needed = False
        except:
            header_needed = True

        df_log = pd.DataFrame(
            {
                'experiment': out_csv_name,
                'repeated_times': [REPEAT_NUMBER],
                'Accuracy': [final_accuracy],
                'Precision': [final_precision],
                'Recall': [final_recall],
                'F1': [final_f1],
                'AUC': [final_auc],


                'CV_list(AUC)': [str(auc_values)],
                'CV_list(Accuracy)': [str(accuracies)],
                'CV_list(Precision)': [str(precisions)],
                'CV_list(Recall)': [str(recalls)],
                'CV_list(F1)': [str(f1_scores)],
            }
        )

        df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)

        print(f"\nResults for REPEAT={REPEAT_NUMBER} have been saved to: {out_csv_name}")


print('Experiment completed.')



from tkinter import *
frame = Tk(screenName='Bug Report Classification Tool')

title = Label(frame, text='Input Bug Report',font=('Arial',20))
title.pack()

classifierTypeLabel = Label(frame, text='Classifier: ' + Method, font=('Arial',8))
classifierTypeLabel.pack()

trainedOnLabel = Label(frame, text='Trained on ' + project + ' bug reports', font=('Arial',8))
trainedOnLabel.pack()


inputBox = Text(frame, height=15, width=25, font=('Arial',16))
inputBox.pack()



def classifyInput():
    print('Classify button clicked')
    userInput = inputBox.get("1.0", "end-1c") 


    ## pre-process user input
    if RemoveHTMLTags:
        userInput = remove_html(userInput)
    if RemoveEmoji:
        userInput = remove_emoji(userInput)

    if RemoveStopWords:
        userInput = remove_stopwords(userInput)

    if CleanString:
        userInput = clean_str(userInput)


    if Method == 'BERTLogisticRegression':
        ## create embedding
        modelInputs = bertTokenizer(userInput, return_tensors='pt', padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            modelOutput = bertModel(**modelInputs)
        
        CLSTOken = modelOutput.last_hidden_state[:, 0, :].squeeze(0).numpy()

        classPrediction = model.predict([CLSTOken])
    elif Method == 'TFIDFNaiveBayes':
        userInputVectorized = tfidf.transform([userInput]).toarray()
        classPrediction = model.predict(userInputVectorized)
    else:
        resultLabel.config(text='Invalid Method Config Setting. Must be BERTLogisticRegression or TFIDFNaiveBayes.')
        return


    if classPrediction[0] == 1:
        resultLabel.config(text='This is a performance related bug report!', fg='green')
    else:
        resultLabel.config(text='This is not a performance related bug report.', fg='red')


classifyButton = Button(frame, text='Classify', font=('Arial',16), command=classifyInput)
classifyButton.pack()

resultLabel = Label(frame, text='', font=('Arial',16))
resultLabel.pack()


print('GUI for bug report classification has been launched')
frame.mainloop()
print('End of file')



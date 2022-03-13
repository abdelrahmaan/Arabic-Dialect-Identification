import numpy as np
import pandas as pd
import streamlit as st
import random

# for reproducibility , to get the same results when evry your run
np.random.seed(2021) 

import re
import string
from collections import Counter


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier



from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.svm import LinearSVC

from pprint import pprint

import sys
import os
import glob
import pickle
import ktrain
from ktrain import text

# #---------------------------------------------------------------------------------#
#                         # Importing fuction for useing
# #---------------------------------------------------------------------------------#

arabic_stop_words = '''
،
ء
ءَ
آ
آب
آذار
آض
آل
آمينَ
آناء
آنفا
آه
آهاً
آهٍ
آهِ
أ
أبدا
أبريل
أبو
أبٌ
أجل
أجمع
أحد
أخبر
أخذ
أخو
أخٌ
أربع
أربعاء
أربعة
أربعمئة
أربعمائة
أرى
أسكن
أصبح
أصلا
أضحى
أطعم
أعطى
أعلم
أغسطس
أفريل
أفعل به
أفٍّ
أقبل
أكتوبر
أل
ألا
ألف
ألفى
أم
أما
أمام
أمامك
أمامكَ
أمد
أمس
أمسى
أمّا
أن
أنا
أنبأ
أنت
أنتم
أنتما
أنتن
أنتِ
أنشأ
أنه
أنًّ
أنّى
أهلا
أو
أوت
أوشك
أول
أولئك
أولاء
أولالك
أوّهْ
أى
أي
أيا
أيار
أيضا
أيلول
أين
أيّ
أيّان
أُفٍّ
ؤ
إحدى
إذ
إذا
إذاً
إذما
إذن
إزاء
إلى
إلي
إليكم
إليكما
إليكنّ
إليكَ
إلَيْكَ
إلّا
إمّا
إن
إنَّ
إى
إياك
إياكم
إياكما
إياكن
إيانا
إياه
إياها
إياهم
إياهما
إياهن
إياي
إيهٍ
ئ
ا
ا?
ا?ى
االا
االتى
ابتدأ
ابين
اتخذ
اثر
اثنا
اثنان
اثني
اثنين
اجل
احد
اخرى
اخلولق
اذا
اربعة
اربعون
اربعين
ارتدّ
استحال
اصبح
اضحى
اطار
اعادة
اعلنت
اف
اكثر
اكد
الآن
الألاء
الألى
الا
الاخيرة
الان
الاول
الاولى
التى
التي
الثاني
الثانية
الحالي
الذاتي
الذى
الذي
الذين
السابق
الف
اللاتي
اللتان
اللتيا
اللتين
اللذان
اللذين
اللواتي
الماضي
المقبل
الوقت
الى
الي
اليه
اليها
اليوم
اما
امام
امس
امسى
ان
انبرى
انقلب
انه
انها
او
اول
اي
ايار
ايام
ايضا
ب
بؤسا
بإن
بئس
باء
بات
باسم
بان
بخٍ
بد
بدلا
برس
بسبب
بسّ
بشكل
بضع
بطآن
بعد
بعدا
بعض
بغتة
بل
بلى
بن
به
بها
بهذا
بيد
بين
بَسْ
بَلْهَ
ة
ت
تاء
تارة
تاسع
تانِ
تانِك
تبدّل
تجاه
تحت
تحوّل
تخذ
ترك
تسع
تسعة
تسعمئة
تسعمائة
تسعون
تسعين
تشرين
تعسا
تعلَّم
تفعلان
تفعلون
تفعلين
تكون
تلقاء
تلك
تم
تموز
تينك
تَيْنِ
تِه
تِي
ث
ثاء
ثالث
ثامن
ثان
ثاني
ثلاث
ثلاثاء
ثلاثة
ثلاثمئة
ثلاثمائة
ثلاثون
ثلاثين
ثم
ثمان
ثمانمئة
ثمانون
ثماني
ثمانية
ثمانين
ثمنمئة
ثمَّ
ثمّ
ثمّة
ج
جانفي
جدا
جعل
جلل
جمعة
جميع
جنيه
جوان
جويلية
جير
جيم
ح
حاء
حادي
حار
حاشا
حاليا
حاي
حبذا
حبيب
حتى
حجا
حدَث
حرى
حزيران
حسب
حقا
حمدا
حمو
حمٌ
حوالى
حول
حيث
حيثما
حين
حيَّ
حَذارِ
خ
خاء
خاصة
خال
خامس
خبَّر
خلا
خلافا
خلال
خلف
خمس
خمسة
خمسمئة
خمسمائة
خمسون
خمسين
خميس
د
دال
درهم
درى
دواليك
دولار
دون
دونك
ديسمبر
دينار
ذ
ذا
ذات
ذاك
ذال
ذانك
ذانِ
ذلك
ذهب
ذو
ذيت
ذينك
ذَيْنِ
ذِه
ذِي
ر
رأى
راء
رابع
راح
رجع
رزق
رويدك
ريال
ريث
رُبَّ
ز
زاي
زعم
زود
زيارة
س
ساء
سابع
سادس
سبت
سبتمبر
سبحان
سبع
سبعة
سبعمئة
سبعمائة
سبعون
سبعين
ست
ستة
ستكون
ستمئة
ستمائة
ستون
ستين
سحقا
سرا
سرعان
سقى
سمعا
سنة
سنتيم
سنوات
سوف
سوى
سين
ش
شباط
شبه
شتانَ
شخصا
شرع
شمال
شيكل
شين
شَتَّانَ
ص
صاد
صار
صباح
صبر
صبرا
صدقا
صراحة
صفر
صهٍ
صهْ
ض
ضاد
ضحوة
ضد
ضمن
ط
طاء
طاق
طالما
طرا
طفق
طَق
ظ
ظاء
ظل
ظلّ
ظنَّ
ع
عاد
عاشر
عام
عاما
عامة
عجبا
عدا
عدة
عدد
عدم
عدَّ
عسى
عشر
عشرة
عشرون
عشرين
عل
علق
علم
على
علي
عليك
عليه
عليها
علًّ
عن
عند
عندما
عنه
عنها
عوض
عيانا
عين
عَدَسْ
غ
غادر
غالبا
غدا
غداة
غير
غين
ـ
ف
فإن
فاء
فان
فانه
فبراير
فرادى
فضلا
فقد
فقط
فكان
فلان
فلس
فهو
فو
فوق
فى
في
فيفري
فيه
فيها
ق
قاطبة
قاف
قال
قام
قبل
قد
قرش
قطّ
قلما
قوة
ك
كأن
كأنّ
كأيّ
كأيّن
كاد
كاف
كان
كانت
كانون
كثيرا
كذا
كذلك
كرب
كسا
كل
كلتا
كلم
كلَّا
كلّما
كم
كما
كن
كى
كيت
كيف
كيفما
كِخ
ل
لأن
لا
لا سيما
لات
لازال
لاسيما
لام
لايزال
لبيك
لدن
لدى
لدي
لذلك
لعل
لعلَّ
لعمر
لقاء
لكن
لكنه
لكنَّ
للامم
لم
لما
لمّا
لن
له
لها
لهذا
لهم
لو
لوكالة
لولا
لوما
ليت
ليرة
ليس
ليسب
م
مئة
مئتان
ما
ما أفعله
ما انفك
ما برح
مائة
ماانفك
مابرح
مادام
ماذا
مارس
مازال
مافتئ
ماي
مايزال
مايو
متى
مثل
مذ
مرّة
مساء
مع
معاذ
معه
معها
مقابل
مكانكم
مكانكما
مكانكنّ
مكانَك
مليار
مليم
مليون
مما
من
منذ
منه
منها
مه
مهما
ميم
ن
نا
نبَّا
نحن
نحو
نعم
نفس
نفسه
نهاية
نوفمبر
نون
نيسان
نيف
نَخْ
نَّ
ه
هؤلاء
ها
هاء
هاكَ
هبّ
هذا
هذه
هل
هللة
هلم
هلّا
هم
هما
همزة
هن
هنا
هناك
هنالك
هو
هي
هيا
هيهات
هيّا
هَؤلاء
هَاتانِ
هَاتَيْنِ
هَاتِه
هَاتِي
هَجْ
هَذا
هَذانِ
هَذَيْنِ
هَذِه
هَذِي
هَيْهات
و
و6
وأبو
وأن
وا
واحد
واضاف
واضافت
واكد
والتي
والذي
وان
واهاً
واو
واوضح
وبين
وثي
وجد
وراءَك
ورد
وعلى
وفي
وقال
وقالت
وقد
وقف
وكان
وكانت
ولا
ولايزال
ولكن
ولم
وله
وليس
ومع
ومن
وهب
وهذا
وهو
وهي
وَيْ
وُشْكَانَ
ى
ي
ياء
يفعلان
يفعلون
يكون
يلي
يمكن
يمين
ين
يناير
يوان
يورو
يوليو
يوم
يونيو
ّأيّان

'''

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations


label_classes = {"LB": "Lebanon","AE": "United Arab Emirates",
                 "PL": "PL","SY": "Syrian",
                 "TN": "Tunisian","BH": "Bahrain",
                 "OM": "Oman","SA": "Saudi Arabia",
                 "MA": "Morocco","DZ": "Algeria",
                 "JO": "Jordan","IQ": "Iraq",
                 "QA": "Qatar","LY": "Libya",
                 "KW": "Kuwait","YE": "Yemen",
                 "EG": "Egypt","SD": "Sudan",
                }
def remove_emojis(text): 
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', text)


def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

def remove_diacritization(text):
    noise = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(noise, '', text)
    return text


def removing_arabic_stopwords(text):
    stemmed_words = " ".join([word for word in text.split() if word not in arabic_stop_words])    
    return stemmed_words

# def stemming(text):
#     ArListem = ArabicLightStemmer()
#     words = word_tokenize(text) 
#     stemed_text = " ".join([ArListem.light_stem(word) for word in words])
#     return stemed_text


def clean_text(text):
    '''
        Make text lowercase, remove text in square brackets,remove links,remove punctuation
        and remove words containing numbers.
    '''

    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    text = re.sub('https?://\S+|www\.\S+', '', text)                # remove urls
    text = re.sub('<.*?>+', '', text)                               # remove tages
    text = re.sub(r'@[^\s]+', ' ', text)                            # Removing @user
    text = re.sub(r'#([^\s]+)', r'\1', text)                        # remove #word with word
    text = re.sub('[%s]' % re.escape(punctuations_list), '', text)  # remove punctuation
    text = re.sub('\n', '', text)                                   # remove new line
    text = re.sub(r'\s+', ' ', text)                                # Removing multiple spaces
    text = re.sub(sequencePattern, seqReplacePattern, text)         # Replace 3 or more consecutive letters by 2 letter.
    text = re.sub(r'\s*[A-Za-z]+\b', '' , text).rstrip()            # Removing English words and make right strip 
    return text


def preprocess_data(text):
    
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    
    # removing stop words
    text = removing_arabic_stopwords(text)

    # remove diacritization
    text = remove_diacritization(text)
    
    # Normalize the text 
    text = normalize_arabic(text)

    # Remove emojis
    text = remove_emojis(text)
    
    # stemming 
#     text = stemming(text)
    

    return text





# #---------------------------------------------------------------------------------#
#                         # Load the model
# #---------------------------------------------------------------------------------#

dir_path = os.path.dirname(os.path.realpath(__file__))
ml_file_path = f"{dir_path}/LinearSVC-ML/LinearSVC_model_and_count_vct.pkl"  

# Load the ML Model and the count vectorizer
with open(ml_file_path, 'rb') as f:
    loaded_count_vect, loaded_LinearSVC_clf = pickle.load(f)


# Load the DL Model 
dl_file_path = f"{dir_path}/MRABERT-DL"  
loaded_marabert_predictor = ktrain.load_predictor(dl_file_path)


# #-------------------------------------------------------------------------------------#
# #-------------------------------Starting Streamlit App--------------------------------#


st.title("Hello world It's an Arabic dialects classification App!")

# Initialize the variable to avoide the same result
input_quote = ""
pred_tags = []

input_quote = st.text_input('Enter Your Arabic text Here!')

is_english = True

# Check if the input is exist not empyt string
if input_quote:

    # Removing English words and make right strip 
    arabic_input = re.sub(r'\s*[A-Za-z]+\b', '' , input_quote).rstrip()
    # print(arabic_input)
    if len(arabic_input.split()) < 3:
        st.write("Please just write at leas 3 Arabic words!")  
    else:
        # Make text preprocessing for the input text
        cleaned_input = preprocess_data(arabic_input)
        
        pred_dialect_ml = loaded_LinearSVC_clf.predict(loaded_count_vect.transform([cleaned_input]))
        pred_dialect_dl = label_classes[loaded_marabert_predictor.predict(cleaned_input)]

        st.write(f"#### The ML Predicted dialect is `{pred_dialect_ml[0]}` for this text `{arabic_input}`")
        st.write(f"#### The DL Predicted dialect is `{pred_dialect_dl}` for this text `{arabic_input}`")

        # df = pd.DataFrame({"Predicted ML Model":pred_dialect_ml,
        #                    "Predicted DL Model":pred_dialect_dl})
        # st.table(df.T)
    
    

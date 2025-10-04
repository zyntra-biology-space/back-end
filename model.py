import requests
import pdfplumber
import os
import pandas as pd
import numpy as np

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset - استخدام البيانات المحلية
try:
    # محاولة تحميل البيانات المحلية
    df_main = pd.read_excel("processed_articles_data.xlsx")
    print(f"✅ تم تحميل البيانات المحلية بنجاح!")
    # print(f"📊 عدد الصفوف: {len(df_main)}")
    # print(f"📋 الأعمدة: {list(df_main.columns)}")
    # print("\n📄 عينة من البيانات:")
    # print(df_main.head())
except FileNotFoundError:
    print("❌ ملف processed_articles_data.xlsx غير موجود")
    print("💡 تأكد من وجود الملف في نفس مجلد الـ notebook")

print("🔍 اختبار البيانات...")
print(f"نوع البيانات: {type(df_main)}")
print(f"شكل البيانات: {df_main.shape}")
print(f"معلومات البيانات:")
print(df_main.info())

df_main.isna().sum()
df_main = df_main.drop(columns=["conclusion"])
df_main.duplicated().sum()
df_main.isna().sum()
# 1️⃣ فلترة الصفوف اللي فيها full_text = NaN
deleted_rows = df_main[df_main["abstract"].isna()]

# 2️⃣ تخزين الصفوف دي في ملف إكسل
deleted_rows.to_excel("deletedData.xlsx", index=False)

# 3️⃣ مسح الصفوف من الداتا الأصلية
df_main = df_main[df_main["abstract"].notna()]

df_main.isna().sum()


print("🔍 اختبار البيانات...")
print(f"نوع البيانات: {type(df_main)}")
print(f"شكل البيانات: {df_main.shape}")
print(f"معلومات البيانات:")
print(df_main.info())


# 1️⃣ خُد العمود اللي فيه النصوص (مثلاً full_text)
texts = df_main["abstract"].tolist()

vectorizer_model = CountVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,  # تجاهل الكلمات النادرة
    max_df=0.95,  # تجاهل الكلمات الشائعة جداً
)
# # # 2️⃣ اعمل الموديل
topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True)

# # 3️⃣ درّب الموديل على النصوص
topics, probs = topic_model.fit_transform(texts)

# 4️⃣ تشوف ملخص التوبيكس

print(topic_model.get_topic_info())
print(topic_model.get_topic_info().Representative_Docs[1])

# شوف أهم الكلمات لكل topic
for i in range(5):  # أول 5 topics
    print(topic_model.get_topic_info().Representative_Docs[i])


# 5️⃣ لو عايز تشوف كلمات كل Topic
# for topic_id in set(topics):
#     print(f"Topic {topic_id}: ", topic_model.get_topic(topic_id))

fig = topic_model.visualize_topics()
fig.show()

# docs_info = topic_model.get_document_info(texts)
# docs_info[docs_info.Topic == -1]

# 2️⃣ خزّن نتيجة كل نص في عمود جديد في الـ DataFrame
df_main["topic"] = topics

# لو عايز تخزن كمان الاحتمالية (probability)
df_main["topic_probability"] = probs

# لو حابب تضيف اسم الـ Topic نفسه (العنوان النصي)
info = topic_model.get_topic_info().set_index("Topic")["Name"]
df_main["topic_name"] = df_main["topic"].map(info)

topic_info = topic_model.get_topic_info().set_index("Topic")
#  اجلب الـ mapping: رقم الـ topic => الوثيقة الممثلة و الـ representation
rep_docs_map = topic_info["Representative_Docs"].to_dict()
representation_map = topic_info["Representation"].to_dict()

#  أضف أعمدة جديدة على حسب رقم الـ topic الخاص بكل نص
df_main["representative_doc"] = df_main["topic"].map(rep_docs_map)
df_main["representation"] = df_main["topic"].map(representation_map)


df_main.info()


# 1️⃣ استخرج الصفوف اللي هتحذفها (outliers)
to_delete = df_main[df_main["topic"] == -1]

# 2️⃣ امسحها من الداتا الأصلية
df_main = df_main[df_main["topic"] != -1]

# 3️⃣ حاول تفتح الملف لو موجود، ولو مش موجود هيعمل جديد
try:
    prev_deleted = pd.read_excel("deletedData.xlsx")
    # دمج القديم مع الجديد
    updated_deleted = pd.concat([prev_deleted, to_delete], ignore_index=True)
except FileNotFoundError:
    # لو الملف مش موجود، استخدم الجديد بس
    updated_deleted = to_delete

# 4️⃣ احفظ كل deleted data في الملف
updated_deleted.to_excel("deletedData.xlsx", index=False)

# كده أي صفوف نضفتها بتتحفظ في ملف مجمع واحد بدون فقدان القديم]

df_main.to_excel("DataWithTopic.xlsx", index=False)


print(df_main.topic_probability.mean())

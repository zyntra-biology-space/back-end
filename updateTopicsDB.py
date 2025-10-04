from pymongo import MongoClient
import certifi
import pandas as pd

# import numpy as np
# import os

MONGO_URI = "mongodb+srv://infocodivera_db_user:m6Uwjdv2f53imWeJ@cluster0.ldqe96m.mongodb.net/?retryWrites=true&w=majority"

try:
    client = MongoClient(
        MONGO_URI, tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=20000
    )
    db = client["nasa_papers"]
    collection = db["articles"]
    print("✅ Connected to MongoDB Atlas successfully")
except Exception as e:
    print(f"❌ MongoDB connection error: {e}")

# افترض إنه عندك topics_to_upload قائمة من الدكشنريز فيها بيانات المواضيع

topic_model_df = pd.read_excel("topic_model_with_explanations.xlsx")
topic_model_to_upload = topic_model_df.to_dict(orient="records")

data_df = pd.read_excel("dataWithTopic.xlsx")
data_to_upload = data_df.to_dict(orient="records")
# كل dict لازم يحتوي على حقل "pmc_id" متوافق مع الموجود في dataWithTopic في MongoDB

for topic in data_to_upload:
    pmc_id = topic.get("pmc_id")
    if pmc_id:  # تأكد أن هناك قيمة
        # تحقق من وجود pmc_id في DataWithTopic collection

        # print(f"topic : {topic}")
        exists = collection.find_one({"pmc_id": pmc_id})
        if exists:
            # print(f"Found pmc_id: {pmc_id} in MongoDB")
            # print(f"topic_model_to_upload  {data_df} ")
            # احدث أو اضف الموضوع في collection مناسبة
            collection.update_one(
                {"pmc_id": pmc_id},
                {
                    "$set": {
                        "topics": topic_model_df.loc[
                            topic["topic"], "public_explanation"
                        ]
                    }
                },
                upsert=True,
            )

            print(f"Updated topic with pmc_id: {pmc_id}")
        else:
            # إذا لم يكن pmc_id موجود، تجاهل هذا الإدخال (skip)
            print(f"Skipped topic with pmc_id: {pmc_id} (not found in MongoDB)")

    else:
        print("Skipped topic with missing pmc_id")

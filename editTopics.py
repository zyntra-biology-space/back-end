import pandas as pd
import google.generativeai as genai


topics_df = pd.read_excel("topic_model.xlsx")
topic_descriptions = []
for idx, row in topics_df.iterrows():
    desc = (
        # f"Topic Name: {row['Name']}\n"
        # f"Top Words: {row['Representation']}\n"
        f"Representative Example: {row['Representative_Docs']}\n"
    )
    topic_descriptions.append(desc)

genai.configure(api_key="AIzaSyAj4iFTHfZpF8oy7k7AyfM5-2VO34SOXqU")
model = genai.GenerativeModel("models/gemini-2.5-flash")

explanations = []
for desc in topic_descriptions:

    prompt = f"""
    You are a scientific assistant.
    Read all of the following texts {desc}. Identify topic labels that best represent the overall content. Output single words or concise key phrases that capture the main subjects, themes, or ideas. Only use words found in the actual documents. Do not write explanations or summaries—just give the most accurate topic labels. You may provide more than one label when appropriate. The output should be suitable for use as website categories or headings.
    """
    response = model.generate_content(prompt)
    explanations.append(response.text)
    print(response.text)

topics_df["public_explanation"] = explanations
topics_df.to_excel("topic_model_with_explanations.xlsx", index=False)

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from groq import Groq
import os


model = SentenceTransformer("all-MiniLM-L6-v2")

# connecting with llm for teh response generate
clint = Groq(
    api_key= os.environ.get("GROQ_API_KEY")
)

#1. create chunk me id + chunk store karna in the form of map
def create_chunk(text,chunk_size=2, overlap=1,source="unknown"):
    paragraph = [
        paragraph.strip()
        for paragraph in text.strip().split("\n")
        if paragraph.strip()
    ]
    chunk_data =[]

    for i in range(0,len(paragraph),chunk_size-overlap):
        chunk_data.append(
            {
                "chunk_id" : i,
                "source":source,
                "chunk": " ".join(paragraph[i:i+chunk_size])
            }
        )

    return chunk_data


# retrival from faiss
def retrival_faiss(question, chunk,index,k=2,temprature= 0.6):
    q_vec = model.encode([question])
    q_vec = np.array(q_vec,dtype="float32")

    dictances, incides = index.search(q_vec,k)
    result =[]

    for distance, i in zip(dictances[0],incides[0]):
        if distance <= temprature:
            result.append(
                        {
                            "distance":float(distance),
                            "chunk_id" :chunk[i]["chunk_id"],
                            "source":chunk[i]["source"],
                            "chunk":chunk[i]["chunk"]
                        }
                    )

    return result


# now biuld the prompt
def build_prompt(question,retrived_chunk):
    context =""

    for item in retrived_chunk:
        context +=f"""
        source:{item["source"]},
        chunk_id:{item["chunk_id"]},
        chunk: {item["chunk"]}
    """

    prompt = f"""
    Answer the question using ONLY provided context.

    If the answer is not prestent in context,
    say : "I don't know based on the provided context"
    Do not use the outside knowledge.

    context:
    {context}

    question:
    {question}

    Answer :
    """

    return prompt



# generate answer function 
def build_answer(prompt):
    resopnse = clint.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )
    return resopnse.choices[0].message.content


#2. text create karna
text = """
Swami Vivekananda was a famous Indian monk, philosopher, and spiritual leader. He was born as Narendranath Datta on 12 January 1863 in Kolkata, India. His parents were Vishwanath Datta and Bhuvaneshwari Devi. From an early age, he was interested in spirituality, philosophy, and learning.

As a young man, Vivekananda became a disciple of Sri Ramakrishna Paramahansa. Ramakrishna had a major influence on Vivekananda's spiritual development and taught him about the importance of experiencing and respecting different paths to God.

Vivekananda became known for his powerful speeches and his ability to explain Indian philosophy to people around the world. He believed that education should develop confidence, strength, character, and compassion in individuals.

In 1893, Vivekananda travelled to the United States and attended the World's Parliament of Religions in Chicago. He delivered a famous speech on 11 September 1893 in which he addressed the audience as brothers and sisters of America. His speech introduced many people in the West to Indian philosophy and Vedanta.

After returning to India, Vivekananda worked to spread the ideas of spirituality, education, social service, and national development. In 1897, he founded the Ramakrishna Mission. The organization focused on education, healthcare, disaster relief, and social service.

Vivekananda strongly believed that serving humanity was an important form of spiritual practice. He encouraged young people to develop physical strength, mental discipline, self-confidence, and a strong sense of purpose.

Swami Vivekananda passed away on 4 July 1902 at Belur Math near Kolkata. His birthday, 12 January, is celebrated in India as National Youth Day because of his influence on young people and his message of courage, character, and service.
"""

#3. print the output
chunk = create_chunk(text=text,chunk_size=2,overlap=1,source="vivekananda.txt")
# print("Created chunk is :")
# print(chunk)
# print()

texts =[item["chunk"] for item in chunk]
print("Converting chunk-data in dictionary to text")
# print(texts)
# print()

# creating texts into embedding
text_vec = model.encode(texts)
text_vec = np.array(text_vec,dtype="float32")
# print("shape :",text_vec.shape)
# print()

#creating Faiss vector databse
index = faiss.IndexFlatL2(text_vec.shape[1])
index.add(text_vec)
# print("Indices:")
# print(index.ntotal)

question = "What was Vivekananda's favorite programming language?"





retrived_context = retrival_faiss(question=question,chunk=chunk,index=index,k=2,temprature= 0.6 )

        
# print('retrived Data from data base is :')
# print(retrived_context)
# print()
# print()

prompt = build_prompt(question=question,retrived_chunk=retrived_context)
print(prompt)


# fetch the llm answer
answer = build_answer(prompt=prompt)
print("Question : ",question)
print("RAG - Answer: ")
print(answer)


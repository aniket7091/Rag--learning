from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from groq import Groq
import os


#1. embeding mordel
model = SentenceTransformer("all-MiniLM-L6-v2")
clint = Groq(
    api_key= os.environ.get("GROQ_API_KEY")
)

#2. chunking function
def data_chunk(text,chunking_size=2,overlap =1):
    sentence =[
        sentence.strip()
        for sentence in text.strip().split("\n")
        if sentence.strip()
    ]
    chunk_data =[]


    for i in range(0,len(sentence),chunking_size-overlap):
        chunk_data.append(" ".join(sentence[i:i+chunking_size]))

    return chunk_data


#4. text or knowledge base
text = """
Chhatrapati Shivaji Maharaj was a great Indian ruler and the founder of the Maratha Empire. He was born on 19 February 1630 at Shivneri Fort in present-day Maharashtra. His father was Shahaji Bhonsle, and his mother was Jijabai. Jijabai played an important role in shaping Shivaji Maharaj's character and taught him values of courage, justice, and devotion.

From a young age, Shivaji Maharaj showed great leadership qualities. He believed that people should be protected and treated fairly. He began capturing important forts in the surrounding regions and gradually established his own independent kingdom.

Shivaji Maharaj was famous for his military strategy and innovative methods of warfare. He effectively used the geography of the Western Ghats to his advantage. His style of warfare is often associated with guerrilla tactics, which helped his smaller forces fight against much larger armies.

He placed great importance on forts because they were essential for protecting his kingdom. Some important forts associated with his rule include Raigad, Pratapgad, Sinhagad, and Rajgad. Raigad later became the capital of the Maratha Empire.

Shivaji Maharaj was also known for his strong administration. He established an organized system of government and appointed ministers to manage different areas of administration. His council of eight ministers was known as the Ashta Pradhan Mandal.

In 1674, Shivaji Maharaj was formally crowned at Raigad Fort. After his coronation, he took the title of Chhatrapati. The coronation marked an important moment in the establishment of the Maratha kingdom.

Shivaji Maharaj respected people of different religions and promoted religious tolerance in his administration. He also gave importance to the protection of women and instructed his soldiers to treat women with respect.

Shivaji Maharaj developed a strong naval force to protect the western coastline. Because of his efforts to strengthen naval power, he is often regarded as one of the pioneers of naval strategy in medieval India.

Chhatrapati Shivaji Maharaj passed away on 3 April 1680 at Raigad Fort. His leadership, military strategies, administration, and vision continue to inspire people across India.
"""


#5. creating chunk
chunk = data_chunk(text= text,chunking_size=2,overlap=1)
print(chunk)
print()
#6. chunking convert into the embedding
res_vec = model.encode(chunk)
res_vec = np.array(res_vec,dtype="float32")

print(res_vec.shape)
print()

#Create FAISS index

index = faiss.IndexFlatL2(res_vec.shape[1])
index.add(res_vec)
print(index.ntotal)
print()

#retrive function
def retrive_faiss(question, chunk,index, k =2):
    query = model.encode([question])
    query = np.array(query,dtype="float32")

    distances, indices = index.search(query,k)

    chunk_data = []
    for distance, i in zip(distances[0],indices[0]):
        chunk_data.append((distance,chunk[i]))

    return chunk_data


#7.question embeddng
question = "What values did Jijabai teach Shivaji Maharaj?"
retrived = retrive_faiss(question=question,chunk=chunk,index=index,k=3)

for i in retrived:
    print(i)
    print()

def build_prompt(question, retrive_chunk):
    context =""
    for distance, chunk in retrive_chunk:
        context +=chunk + "\n"


    prompt = f"""
    Answer the question using the context below.
    
    context =
    {context}

    question = 
    {question}

    Answer :
    """
    return prompt


prompt = build_prompt(question=question,retrive_chunk=retrived)
print(prompt)
print()

# intrigationg the llm
def build_answer(prompt):
    response = clint.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


answer = build_answer(prompt=prompt)

print("Question : ",question)
print("Answer : ",answer)
print()

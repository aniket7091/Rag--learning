# 1. import the transformer model
from sentence_transformers import SentenceTransformer, util
from groq import Groq
import os

clint = Groq(
    api_key = os.environ.get("GROQ_API_KEY")
)

model = SentenceTransformer("all-MiniLM-L6-v2")

#2. make the create-chunk function
def create_chunk(text, chunk_size =2,overlap=1):
    sentences = text.strip().split("\n")
    chunk_data =[]
    for i in range(0,len(sentences),chunk_size-overlap):
        chunk_data.append(" ".join(sentences[i:i+chunk_size]))

    return chunk_data

#3. create teh retrival function
def retrive(question, chunks, doc_vec,k =2):
    queary_vec = model.encode(question)
    result =[]
    for i in range(len(chunks)):
        similarity = util.cos_sim(queary_vec,doc_vec[i]).item()
        result.append((similarity,chunks[i]))

    result.sort(reverse=True)
    return result[:k]


#4. create the build prompt function
def build_prompt(question, result):
    context =""

    for score,doc in result:
        context += doc + "\n"

    prompt = f"""
    Answer the question by using context below.

    context = 
    {context}

    question = 
    {question}

   Answer :
    """
    return prompt


#5. now we are going to geerate the answer 
def generate_answer(prompt):

    response = clint.chat.completions.create(
        model = "openai/gpt-oss-20b",
        messages =[
            {
               "role" : "user",
               "content" : prompt
            }
        ]
    )
    return response.choices[0].message.content
    

#4. give the knowledge base as a text
text = """
Dr. A.P.J. Abdul Kalam was an Indian aerospace scientist and the 11th President of India.
He was born on 15 October 1931 in Rameswaram, Tamil Nadu.
Dr. Kalam played an important role in India's space and missile development programs.
He is popularly known as the Missile Man of India.
He strongly encouraged students to dream big and work hard to achieve their goals.
Dr. Kalam wrote several books, including Wings of Fire and Ignited Minds.
He received the Bharat Ratna, India's highest civilian award, in 1997.
"""

#5. create the embeeding of knowledge base
chunk = create_chunk(text=text,chunk_size=2,overlap =1)
print(chunk)
print()
doc_vec = model.encode(chunk)
print("Embedding vector :",doc_vec.shape)
print()


#6 create the embedding of question
question = "Which award did Dr. Kalam receive in 1997?"
result = retrive(question=question,chunks=chunk,doc_vec=doc_vec,k=3)

#print the prompt output
prompt = build_prompt(question=question,result=result)

print(prompt)
print()

# generate the response 
answer = generate_answer(prompt=prompt)
print("question : ", question)
print()
print("Answer : ",answer)
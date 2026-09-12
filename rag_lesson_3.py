from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, chunk_size,overlap):
    sentances = text.strip().split("\n")
    chunked_data =[]
    for i in range(0,len(sentances),chunk_size-overlap):
        chunked_data.append(" ".join(sentances[i:i+chunk_size]))

    return chunked_data


def retrive(question, chunks,doc_vec,k=2):

    query_vec = model.encode(question)
    result =[]
    for i in range(len(chunks)):
        similrity = util.cos_sim(query_vec,doc_vec[i]).item()
        result.append((similrity,chunks[i]))

    result.sort(reverse= True)

    return result[:k]




# text = """
# Python is a programming language.
# It is easy to learn.
# Python is used in web development.
# Python is popular for machine learning.
# Python has many libraries.
# Python is used in data science.
# """

text = """

Dr. A.P.J. Abdul Kalam was an Indian aerospace scientist and the 11th President of India.
He was born on 15 October 1931 in Rameswaram, Tamil Nadu.
Dr. Kalam played an important role in India's space and missile development programs.
He is popularly known as the Missile Man of India.
He strongly encouraged students to dream big and work hard to achieve their goals.
Dr. Kalam wrote several books, including Wings of Fire and Ignited Minds.
He received the Bharat Ratna, India's highest civilian award, in 1997.
"""

result = chunk_text(text= text, chunk_size=2,overlap=1)

print(result)

res_vec = model.encode(result)
print(res_vec.shape)
print()

# question = "What is Python used for?"

question = "Which award did Dr. Kalam receive in 1997?"

result1 = retrive(question=question, chunks=result,doc_vec=res_vec,k=2)

print("Question : ",question)
for score, doc in result1:
    
    print("Document : ",doc)
    print("Score : ",score)
print("-"*80)





from sentence_transformers import SentenceTransformer, util


# import the model
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrive(question, documents, doc_vec, k=2):
    query_vec = model.encode(question)
    result =[]

    for i in range(len(documents)):
        similarity = util.cos_sim(query_vec,doc_vec[i]).item()
        result.append((similarity,documents[i]))

    result.sort(reverse=True)
    return result[:k]


# ceating the knowledge base
documents = [
    "Python is a programming language.",
    "JavaScript is used for web development.",
    "Python is popular for machine learning."
]

# cnvert the knowledge base into the embedding vector

doc_vec = model.encode(documents)
print("Document embedding : ")
print(doc_vec.shape)
print()




# question
question = [
     "What is Python?",
     "Which language is used for web development?",
     "What is Python used for?"
]


for i in range(len(question)):
    result = retrive(question=question[i],documents=documents,doc_vec=doc_vec,k=2)
    print(question[i])
    for score, doc in result:
        print("Doc :",doc)
        print("Score : ",score)
        print()

    print('-'*80)   

# qu_vec = model.encode(question)
# print(question)
# print(qu_vec)

# # finding the similarity which has higher priority
# # best_score = -1;
# # best_doc =""

# result =[]

# for i in range(len(documents)):
#     similarity = util.cos_sim(qu_vec,doc_vec[i]).item()
#     result.append((similarity,documents[i]))
#     # print( "Document : ",documents[i])
#     # print("Similarity: ", similarity)
#     # print()

#     # if similarity > best_score:
#     #     best_score = similarity
#     #     best_doc = documents[i]

# result.sort( reverse=True)

# for best_score, best_doc in result[:2]:
#     print("best Doc for the question is :",best_doc)
#     print()
#     print("Best score :",best_score)
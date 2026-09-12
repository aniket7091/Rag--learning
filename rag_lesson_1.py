import numpy as np

# initilize the knowledge base
documents = [
    "Python is a programming language.",
    "JavaScript is used for web development.",
    "Python is popular for machine learning."
]

# ask the question
question = "What is Python used for?"


# print the value of documents and question
print(documents)
print()
print(question)


# convert the doc and question into the fack vector and apply the dot prod using numpy

doc_vec = np.array([1,2])
qa_vec = np.array([1,2])

# find the similarity 
similarity = np.dot(qa_vec,doc_vec)
print(similarity)

# now printing the document by using the loop and find the similarity on the fake vector

document_vec = [
    np.array([1,1]),
    np.array([5,1]),
    np.array([1,2])
]

best_score =-1
best_doc = ""

for i in range(len(documents)):
    similarity = np.dot(qa_vec,document_vec[i])

    if similarity > best_score:
        best_score = similarity
        best_doc = documents[i]

print("best Document :")
print(best_doc)
print()
print("best Similarity :")
print(best_score)

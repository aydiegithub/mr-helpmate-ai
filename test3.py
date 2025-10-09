from src.backend.generation_layer import ChatCompletion
from src.artifacts import SystemInstruction
from src.backend.intent_moderation_check import (IntentCheck, ModerationCheck)
from src.database.chromadb_vectorstore import VectorStore
from src.backend.reranker import Reranker



mod_chk = ModerationCheck()
int_cnf = IntentCheck()
chat_completion = ChatCompletion()
vector_store = VectorStore()
reranker = Reranker()

messages = [
    {"role": "system", "content": SystemInstruction.prompt}
]

while True:
    input_message = input("User: ")
    if input_message in ["exit", "bye", "end"]:
        print("Thank you for your time, hope I helped. Bye!")
        print("Chat Terminated....")
        break
    
    if mod_chk.check_moderation(input_message=input_message):
        print("Your Conversation has been flagged!, restarting the conversation.")
        continue
    
    if int_cnf.check_intent(input_message=input_message):
        top_10_documents = vector_store.query_from_db(query=input_message,
                                                  top_k=10)

        print(type(top_10_documents['documents']))
        
        reranked_top_3 = reranker.rerank_documents(documents=top_10_documents['documents'],
                                                   query=input_message,
                                                   top_k=3)
        
        reranked_context = ""
        for score, doc in reranked_top_3:
            reranked_context += doc + "\n\n\n"
        
        messages.append({
            "role": "user", "content": input_message + f"###context:\n" + reranked_context
        })
        
        ai_response = chat_completion.chat_completion(messages=messages)
        if mod_chk.check_moderation(input_message=ai_response):
            print("Your Conversation has been flagged!, restarting the conversation.")
            continue
        
        print("Assistant: ", ai_response)
        print("\n\n")
        
    else:
        print("Your question is out of scope or context, please ask the right question related to the domain.")
        continue
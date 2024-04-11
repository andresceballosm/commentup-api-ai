from flask import Flask, jsonify, request, Response
import openai
import pandas as pd

from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import pinecone

import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api = os.getenv("PINECONE_API")
pinecone_env = os.getenv("PINECONE_ENV=northamerica-northeast1-gcp")

app = Flask(__name__)

app.secret_key = 'myawesomesecretkey'
model_engine = "text-davinci-002"
# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

values = [
    "Commentup is an AI-powered assistant that helps a recruiter find suitable candidates for a position.We will help you filter out the best candidates for your company, saving you time and money.",
    "To see all the positions you have created you must call the following query:getPositions",
    "To see all the positions you have open today, you must call the following query:getPositionsOpen",
    "To see all the positions you have paused today, you must call the following query:getPositionsPaused",
    "To see all the positions you have closed today, you must call the following query:getPositionsClosed",
    "To see the detail for a position id, you must call the following query:getPositionDetail",
    "To see all the candidates or postulations for a position number x, you must call the following query:getAllPostulations",
    "To see all the candidates in status X for a position id, you must call the following query:getPostulationsByStatus",
    "To create or add a new position id, you must call the following query:createPosition",
    "Comment has many interesting features like: 1)Recruiting candidates, 2)Filter the best candidates, 3)Chat, 4)Share information about social network.",
    "At our company, we have developed a cutting-edge that uses AI system that generates customized tests for each job position to accurately assess the skills and qualifications of job candidates.",
    "CommentUp is free!, we would love to hear your feedback, this is only the first feature, the idea is create more with the help of our early adopter as result of your comments. Our goal is to create payment plans in the future, but as an early adopter, you will receive exclusive benefits.",

    "Commentup es un asistente impulsado por inteligencia artificial que ayuda a un reclutador a encontrar candidatos adecuados para un puesto. Lo ayudaremos a filtrar a los mejores candidatos para su empresa, ahorrandole tiempo y dinero",
    "Para ver todas las posiciones que has creado debes llamar a la siguiente query:getPositions.",
    "Para ver todas las posiciones que tiene abiertas hoy, debe llamar a la siguiente query: getPositionsOpen.",
    "Para ver todas las posiciones que ha pausado hoy, debe llamar a la siguiente query:getPositionsPaused",
    "Para ver todas las posiciones que ha cerrado hoy, debe llamar a la siguiente query:getPositionsClosed",
    "Para ver el detalle de una identificaci&#243;n de posici&#243;n, debe llamar a la siguiente query:getPositionDetail",
    "Para ver todos los candidatos o postulaciones para un puesto numero x, debe llamar a la siguiente query:getAllPostulations",
    "Para ver todos los candidatos en estado X para una identificaci&#243;n de posici&#243;n, debe llamar a la siguiente query:getPostulationsByStatus",
    "Para crear o agregar una nueva identificaci&#243;n de posici&#243;n, debe llamar a la siguiente query:createPosition",
    "Comment tiene muchas caracter&#237;sticas interesantes como: 1)Reclutar candidatos, 2)Filtrar a los mejores candidatos, 3)Chatear, 4)Compartir informaci&#243;n sobre la red social",
    "En nuestra empresa, hemos desarrollado un sistema de vanguardia que utiliza inteligencia artificial que genera pruebas personalizadas para cada puesto de trabajo para evaluar con precisi&#243;n las habilidades y calificaciones de los candidatos",
    "CommentUp es gratis! Nos encantar&#237;a escuchar sus comentarios, esta es solo la primera caracter&#237;stica, la idea es crear m&#225;s con la ayuda de nuestros primeros usuarios como resultado de sus comentarios. Nuestro objetivo es crear planes de pago en el futuro, pero como uno de los primeros en adoptar, recibir&#225; beneficios exclusivos"
]

def embed_text(path="texto.csv"):
    conocimiento_df = pd.read_csv(path)
    conocimiento_df['Embedding'] = conocimiento_df['texto'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    conocimiento_df.to_csv('mtg-embeddings.csv')
    return conocimiento_df

def buscar(busqueda, datos, n_resultados=5):
    busqueda_embed = get_embedding(busqueda, engine="text-embedding-ada-002")
    datos["Similitud"] = datos['Embedding'].apply(lambda x: cosine_similarity(x, busqueda_embed))
    datos = datos.sort_values("Similitud", ascending=False)
    return datos.iloc[:n_resultados][["texto", "Similitud", "Embedding"]]

def get_highest_score_url(items):
    print(items)
    highest_score_item = max(items, key=lambda item: item["score"])
 
    if highest_score_item["score"] > 0.8:
        return {"text" : highest_score_item["metadata"]['texto']}
    else:
        return {"text" : "Sorry, I don't know"}

@app.route('/chat', methods=['POST'])
def get_chat():
    # Receiving Data
    question = request.json['question']

    if question:
        pinecone.init(api_key=pinecone_api, environment=pinecone_env)
        index = pinecone.Index("text-index-name")

        vector = openai.Embedding.create(
            input=question,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]

        search_response = index.query(
            namespace="queries-namespace-international-v4",
            top_k=5,
            include_values=True,
            include_metadata=True,
            vector=vector,
        )
        # resp = buscar(question, parrafos, 5)
        
        text = get_highest_score_url(search_response['matches'])
        print(text)
        response = jsonify(text)
        response.status_code = 200
        return response
    else:
        return not_found()


@app.route('/pinecode', methods=['GET'])
def store_embeddings():
    pinecone.init(api_key=pinecone_api, environment=pinecone_env)
    index = pinecone.Index("text-index-name")

    pinecone_vectors = []
    for parrafo in values:
        vector = openai.Embedding.create(
            input=parrafo,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
        pinecone_vectors.append((str(parrafo), vector, {"texto": parrafo}))

    upsert_response = index.upsert(vectors=pinecone_vectors, namespace="queries-namespace-international-v4")
    print(upsert_response)
    return Response('Index Upserted!', mimetype="application/json")

    # Ruta para filtrar usuarios según una descripción
@app.route('/filter_users', methods=['POST'])
def filter_users():
    request_data = request.get_json()
    users = request_data['users']
    description = request_data['description']

    texts = []
    # Obtener embeddings de los usuarios
    for user in users:
        text = f"name:{ user['name']} country:{user['country']} score={user['score']} date:{user['date']}"
        texts.append(text)
    
    parrafos = pd.DataFrame(texts, columns=["texto"])
    parrafos['Embedding'] = parrafos["texto"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002')) # Nueva columna con los embeddings de los parrafos
    parrafos.to_csv('commentup_chat.csv')
    resp = buscar(description, parrafos, 3)
    print('RESP', resp)
    return jsonify({'filtered_users': 'works'})


# @app.route('/users/<id>', methods=['GET'])
# def get_user(id):
#     print(id)
#     user = mongo.db.users.find_one({'_id': ObjectId(id), })
#     response = json_util.dumps(user)
#     return Response(response, mimetype="application/json")


# @app.route('/users/<id>', methods=['DELETE'])
# def delete_user(id):
#     mongo.db.users.delete_one({'_id': ObjectId(id)})
#     response = jsonify({'message': 'User' + id + ' Deleted Successfully'})
#     response.status_code = 200
#     return response


# @app.route('/users/<_id>', methods=['PUT'])
# def update_user(_id):
#     username = request.json['username']
#     email = request.json['email']
#     password = request.json['password']
#     if username and email and password and _id:
#         hashed_password = generate_password_hash(password)
#         mongo.db.users.update_one(
#             {'_id': ObjectId(_id['$oid']) if '$oid' in _id else ObjectId(_id)}, {'$set': {'username': username, 'email': email, 'password': hashed_password}})
#         response = jsonify({'message': 'User' + _id + 'Updated Successfuly'})
#         response.status_code = 200
#         return response
#     else:
#       return not_found()


@app.errorhandler(404)
def not_found(error=None):
    message = {
        'message': 'Resource Not Found ' + request.url,
        'status': 404
    }
    response = jsonify(message)
    response.status_code = 404
    return response


if __name__ == "__main__":
    app.run(debug=True, port=8000)

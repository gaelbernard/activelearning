import json
import sqlite3
import pandas as pd
from itertools import islice
from nltk import edit_distance
from AL import AL
from flask import Flask, redirect, render_template, request, send_from_directory, url_for
from flask_cors import CORS
import os
import time
from txtai.embeddings import Embeddings

#ENTRYPOINT ["gunicorn", "app:app", "--bind", "0.0.0.0:5008", "--timeout", "10000"]
embeddings = Embeddings()

if not os.path.exists(AL.WIKI_DB):
    print ('downloading wiki db, this will take a while (~6go)')
    embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")
    embeddings.save(AL.BASE_WIKI)
    print ('Wiki was downloaded')
else: #modAL==0.4.1
    embeddings.load(AL.BASE_WIKI)

t = time.time()
app = Flask(__name__)
CORS(app)

hugDB = sqlite3.connect(AL.WIKI_DB, check_same_thread=False)

limit = 30000000000000000

# Loading the concept for search
entire_wiki_id = pd.read_sql_query(f"select id from sections LIMIT {limit}", hugDB)['id'].tolist()
entire_wiki_id_lowered = [x.lower() for x in entire_wiki_id]

# Loading the Active Learner
al = AL()

@app.route('/')
def index():
   return render_template('index.html')


def searchName(q):

    global entire_wiki_id

    # Step 1: Fast search
    n_fast = 20
    n_final = 10

    candidates_ids = [entire_wiki_id[entire_wiki_id_lowered.index(x)] for x in list(islice(filter(lambda x: q.lower() in x, entire_wiki_id_lowered), n_fast))]

    if q.lower() in entire_wiki_id_lowered:
        id = entire_wiki_id[entire_wiki_id_lowered.index(q.lower())]
        if id not in candidates_ids:
            candidates_ids.insert(0, id)

    if len(candidates_ids) == 0:
        return []

    txt = str([x for x in candidates_ids])[1:-1].replace("\\'","''")
    data = pd.read_sql_query(f'''select sections.id, data as percentile, indexid, text from documents INNER JOIN sections on sections.id = documents.id where sections.id IN ({txt})''', hugDB).sort_values('percentile', ascending=False).head(n_final)
    data['percentile'] = data['percentile'].astype('str').str.replace('{"percentile": ','',regex=False).str.replace('}','',regex=False).astype(float).round(1)
    data['exact-match'] = data['id'].str.lower() == q.lower()
    data['levensthein-similarity'] = data['id'].apply(lambda x: 1-edit_distance(q, x)/max(len(q), len(x))).round(1)
    data = data.sort_values(['exact-match', 'levensthein-similarity', 'percentile'], ascending=False).head(n_final)
    return data.to_dict(orient='records')


@app.route('/search', methods = ['POST'])
def search():
    r = request.get_json()
    q = r['q']
    return json.dumps(searchName(q))

@app.route('/build', methods = ['POST'])
def build():
    r = request.get_json()
    selected_concepts = pd.DataFrame(json.loads(r['selected_concepts'])).transpose()
    selected_concepts.index = selected_concepts['indexid']
    description = r['description']
    color_mapping = json.loads(r['color_mapping'])

    training_annotation = selected_concepts.to_dict(orient='index')
    testing_annotation = {}

    output = al.load_model(training_annotation, testing_annotation)
    output['description'] = description
    output['color_mapping'] = color_mapping

    return json.dumps(output, indent=4)

@app.route('/train', methods = ['POST'])
def train():
    r = request.get_json()
    training_annotations = pd.DataFrame(json.loads(r['training_annotations'])).transpose().to_dict(orient='index')
    test_annotations = pd.DataFrame(json.loads(r['test_annotations'])).transpose().to_dict(orient='index')
    output = al.load_model(training_annotations, test_annotations)
    output['description'] = r['description']
    output['color_mapping'] = json.loads(r['color_mapping'])

    return json.dumps(output, indent=4)


if __name__ == '__main__':
   app.run(port=5008, host='0.0.0.0', threaded=True)


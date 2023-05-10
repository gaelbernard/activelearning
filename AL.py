import os.path
import numpy as np
import pandas as pd
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import faiss
from sklearn.neural_network import MLPClassifier
import sqlite3
import gc

class AL:
    BASE_WIKI = 'data/wiki'
    WIKI_DB = BASE_WIKI + '/documents'
    EMBEDDING_INDEX = BASE_WIKI + '/embeddings'
    TOP1_EMBEDDING = 'data/top1_embeddings.pickle'
    RANDOM_EMBEDDING = 'data/randoms_embeddings.pickle'

    def __init__(self):
        self.hugDB = sqlite3.connect(AL.WIKI_DB, check_same_thread=False)
        self.index = faiss.read_index(AL.EMBEDDING_INDEX)
        self.index.make_direct_map()
        self.X_pool, self.X_random = self.load_features()

    def load_features(self):
        if not os.path.exists(AL.TOP1_EMBEDDING) or not os.path.exists(AL.TOP1_EMBEDDING):
            print ('Loading features, this will take a while')
            # All
            perc = pd.read_sql_query(f"select indexid, documents.id, data from documents INNER JOIN sections ON sections.id = documents.id", self.hugDB).set_index(['id', 'indexid'])['data'].astype('str').str.replace('{"percentile": ','',regex=False).str.replace('}','',regex=False).astype(float)

            # Export top 1% percentile
            top = pd.DataFrame(perc[perc>0.99].index.tolist(), columns=['id', 'indexid'])
            top_e = {}
            for _, r in top.iterrows():
                top_e[r['id']] = self.index.reconstruct(r['indexid'])
            del top
            pd.DataFrame(top_e).transpose().to_pickle(AL.TOP1_EMBEDDING)
            del top_e

            # Export random ones
            rand = pd.DataFrame(perc.index.tolist(), columns=['id', 'indexid']).sample(100000)
            del perc

            rand_e = {}
            for _, r in rand.iterrows():
                rand_e[r['id']] = self.index.reconstruct(r['indexid'])
            del rand
            pd.DataFrame(rand_e).transpose().to_pickle(AL.RANDOM_EMBEDDING)

            del rand_e
            gc.collect()


        return pd.read_pickle(AL.TOP1_EMBEDDING), pd.read_pickle(AL.RANDOM_EMBEDDING)


    def load_model(self, training_annotations, test_annotations):

        o = {}
        o['training_annotations'] = training_annotations
        o['test_annotations'] = test_annotations
        X = np.array([self.index.reconstruct(int(k)) for k in training_annotations.keys()])
        y = np.array([str(x['label']) for x in training_annotations.values()])

        learner = ActiveLearner(
            estimator=MLPClassifier(max_iter=1000, hidden_layer_sizes=(300,100,)),
            query_strategy=uncertainty_sampling,
            X_training=X, y_training=y
        )

        query_idx, query_inst = learner.query(self.X_pool)
        id = query_inst.index.tolist()[0]
        id_str = str([id])[1:-1].replace("\\'", "''")

        meta = pd.read_sql_query(f"select indexid, text from sections where id = {id_str}", self.hugDB).to_dict(orient='records')[0]
        o['next_question'] = {
            'id': id,
            'indexid': str(meta['indexid']),
            'text': str(meta['text']),
        }
        rand = self.X_random
        rand_score = pd.DataFrame(learner.predict_proba(rand), index=rand.index)

        # Samples
        min_r = 1 / rand_score.shape[0]
        certainty = (rand_score.max(axis=1) - min_r) / (1 - min_r)
        uncertain = certainty < 0.8
        prediction = pd.Series(learner.predict(rand), index=rand.index)
        prediction.loc[uncertain] = 'Uncertain'
        o['distrib'] = {}
        sorted_label = list(set([x['label'] for x in training_annotations.values()]))
        sorted_label.append('Uncertain')

        samples = []
        preds = []
        for label in sorted_label:
            ldf = prediction[prediction==label]
            s = min(ldf.shape[0], 8)
            samples.extend(ldf.sample(s).index.tolist())
            preds.extend([label]*s)
            o['distrib'][label] = round((ldf.shape[0])/prediction.shape[0], 4)

        id_str = str(samples)[1:-1].replace("\\'", "''")

        samples = pd.read_sql_query(f"select indexid, text, id from sections where id in ({id_str})", self.hugDB).set_index('id').loc[samples,:].reset_index()
        samples.index = samples['indexid'].tolist()
        samples['label'] = preds
        o['samples_prediction'] = samples.to_dict(orient='index')

        return o

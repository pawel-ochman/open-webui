import chromadb
from chromadb.config import Settings
from chromadb.api import ClientAPI, BaseAPI
from chromadb.utils.batch_utils import create_batches

from apps.rag.vector.clients import VectorClient, VectorCollection

class ChromaCollection(VectorCollection):

    def __init__(self, collection, api: BaseAPI):
        super().__init__(collection.name)
        self.collection = collection
        self.api = api

    def add(self, documents, ids, embeddings, metadatas):
        for batch in create_batches(
            api=self.api,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        ):
            self.collection.add(*batch)

    def get(self):
        return self.collection.get()

    def query(self, query_embeddings, n_results):
        return self.collection.query(query_embeddings=query_embeddings, n_results=n_results)



class ChromaClientBase(VectorClient):
    def __init__(self, api: ClientAPI):
        super().__init__()

        self.client = api

    def get_collection(self, name) -> VectorCollection:
        return ChromaCollection(self.client.get_collection(name), self.client)

    def list_collections(self):
        return [ChromaCollection(collection, self.client) for collection in self.client.list_collections()]

    def delete_collection(self, name: str):
        self.client.delete_collection(name)

    def create_collection(self, name):
        return ChromaCollection(self.client.create_collection(name), self.client)

    def reset(self):
        self.client.reset()

    def get_or_create_collection(self, name):
        return ChromaCollection(self.client.get_or_create_collection(name), self.client)


class ChromaClient(ChromaClientBase):
    def __init__(self, path, tenant, database):
        super().__init__(chromadb.PersistentClient(
            path=path,
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
            tenant=tenant,
            database=database,
        ))

class ChromaClientWeb(ChromaClientBase):
    def __init__(self, host, port, headers, ssl, tenant, database):
        super().__init__(chromadb.HttpClient(
            host=host,
            port=port,
            headers=headers,
            ssl=ssl,
            tenant=tenant,
            database=database,
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        ))
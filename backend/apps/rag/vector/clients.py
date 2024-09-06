
class VectorCollection:

    def __init__(self, name):
        self.name = name

    def add(self, documents, ids, embeddings, metadatas):
        raise NotImplementedError('add')

    def get(self):
        raise NotImplementedError('get')

    def query(self, query_embeddings, n_results):
        raise NotImplementedError('query')

class VectorClient:
    def __init__(self):
        pass

    def get_collection(self, name) -> VectorCollection:
        raise NotImplementedError('get_collection')

    def list_collections(self) -> list[VectorCollection]:
        raise NotImplementedError('list_collections')

    def delete_collection(self, collection):
        raise NotImplementedError('delete_collection')

    def create_collection(self, collection):
        raise NotImplementedError('create_collection')

    def reset(self):
        raise NotImplementedError('reset_collection')

    def get_or_create_collection(self, collection):
        raise NotImplementedError('collection')

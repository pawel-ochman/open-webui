import json
import uuid
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from pydantic import Field
from pydantic.deprecated.class_validators import root_validator

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel

from ollama._client import Client
from ollama._types import  RequestError, Mapping, Any, Options
from typing import AnyStr, Optional, Union, Sequence, overload, List

from apps.rag.vector.clients import VectorClient, VectorCollection

from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import CollectionInfo
from qdrant_client.http.models import VectorParams, Distance

from logging import getLogger

log = getLogger(__name__)


class OllamaClientFix(Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def embed(
            self,
            model: str = '',
            input: Union[str, Sequence[AnyStr]] = '',
            truncate: bool = True,
            options: Optional[Options] = None,
            keep_alive: Optional[Union[float, str]] = None,
    ) -> Mapping[str, Any]:
        if not model:
            raise RequestError('must provide a model')

        response = self._request(
            'POST',
            '/api/embed',
            json={
                'model': model,
                'input': input,
                'truncate': truncate,
                'options': options or {},
                'keep_alive': keep_alive,
            },
        )

        j = response.json()

        return j

class OllamaEmbeddingsFix(Embeddings):

    _client: Client = Field(default=None)
    model: str = ''

    def __init__(self, model, client: OllamaClientFix):
        self._client = client
        self.model = model
        super().__init__()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        response= self._client.embed(self.model, texts)

        embedded_docs = response["embeddings"]

        return embedded_docs

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embedded_docs = (self._client.embed(self.model, texts))[
            "embeddings"
        ]
        return embedded_docs

    async def aembed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return (await self.aembed_documents([text]))[0]


class QDrantCollection(VectorCollection):
    def __init__(self, name: str, client: QdrantClient):
        super().__init__(name)
        self.name = name
        self.store = QdrantVectorStore(
            client=client,
            collection_name=name,
            embedding=OllamaEmbeddingsFix(
                model="nomic-embed-text",
                client=OllamaClientFix(host="172.24.2.24:11434")
            )
        )

    def add(self, documents, ids, embeddings, metadatas):
        items = []

        for i, doc in enumerate(documents):
            obj = {
                "id": str(uuid.uuid4()),
                "embedding": embeddings[i],
                "page_content": documents[i],
                "metadata": metadatas[i]
            }
            items.append(Document(**obj))

        self.store.add_documents(items)

    def get(self):
        points, _ = self.store.client.scroll(collection_name=self.name)
        docs = [Document(**{"id": doc.id, **doc.payload}) for doc in points]
        return {
            "documents": [doc.page_content for doc in docs],
            "metadatas": [doc.metadata for doc in docs],
            "ids": [doc.id for doc in docs]
        }

    def query(self, query_embeddings: List[List[float]], n_results):
        search_result = self.store.client.search(
            collection_name=self.name,
            query_vector=query_embeddings[0],
            limit=n_results  # Number of nearest neighbors to return
        )
        docs = [Document(**{"id": doc.id, **doc.payload}) for doc in search_result]
        res = {
            "documents": [[doc.page_content for doc in docs]],
            "metadatas": [[doc.metadata for doc in docs]],
            "ids": [[doc.id for doc in docs]]
        }
        return res

class QDrantClient(VectorClient):
    def __init__(self, url, port):
        super().__init__()
        log.info(f"Creating QDrantClient with url={url}, port={port}")
        self.client = QdrantClient(url=url, port=port)


    def get_collection(self, name) -> VectorCollection:
        return QDrantCollection(name=name, client=self.client)

    def list_collections(self):
        collections = self.client.http.collections_api.get_collections()
        return [self.get_collection(col.name) for col in collections.result.collections]

    def delete_collection(self, name: str):
        self.client.delete_collection(name)

    def create_collection(self, name):
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        return self.get_collection(name)

    def reset(self):
        pass

    def get_or_create_collection(self, name):
        if not self.client.collection_exists(collection_name=name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

        return self.get_collection(name)
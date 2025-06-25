"""
Run once per dataset refresh:

    python -m app.ingest
"""

from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import get_settings

settings = get_settings()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLIT_SIZE = 800
SPLIT_OVERLAP = 100


def iter_documents(root: Path) -> Iterable:
    """Yield LangChain Documents from .txt/.md/.pdf files."""
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            loader = TextLoader(str(path))
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            print(f"⏩  Skipping unsupported file: {path.name}")
            continue

        for doc in loader.load():
            doc.metadata["source"] = str(path.relative_to(root))
            yield doc


def run() -> None:
    print(f"🔍  Scanning {DATA_DIR}")
    raw_docs = list(iter_documents(DATA_DIR))
    print(f"📄  Loaded {len(raw_docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SPLIT_SIZE, chunk_overlap=SPLIT_OVERLAP
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"✂️   Split into {len(chunks)} chunks")

    store = Chroma.from_documents(
        chunks,
        embedding=OpenAIEmbeddings(api_key=settings.openai_api_key),
        persist_directory=settings.chroma_dir,
    )
    store.persist()
    print(f"✅  Embeddings stored in {settings.chroma_dir}/")


if __name__ == "__main__":
    run()

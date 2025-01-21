import os
import warnings
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import numpy as np
import argparse

warnings.filterwarnings("ignore")
os.environ['OPENAI_API_KEY'] = '****'
os.environ['LANGCHAIN_API_KEY'] = '****'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'task'

# PDF 로드 함수
def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return [doc.page_content for doc in docs]
    except Exception as e:
        print(f"PDF 로드 중 오류 발생: {e}")
        return None

# 텍스트 Splitter 함수
def split_text(documents, splitter_type="recursive", chunk_size=500, chunk_overlap=0):
    try:
        # splitter_type에 따라 텍스트 분리기 선택
        if splitter_type == "character":
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif splitter_type == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            raise ValueError("지원되지 않는 Splitter 타입입니다.")
        
        texts = text_splitter.create_documents(documents)
        return [text.page_content for text in texts]
    except Exception as e:
        print(f"텍스트 분리 중 오류 발생: {e}")
        return None

# Embedding 함수
def initialize_embedding(model_name="intfloat/multilingual-e5-large-instruct"):
    try:
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return hf_embeddings
    except Exception as e:
        print(f"Embedding 초기화 중 오류 발생: {e}")
        return None

# Search_query 함수
def search_query(query, texts, embeddings):
    try:
        # 문서와 질의의 Embedding 계산
        embedded_query = embeddings.embed_query(query)
        embedded_documents = embeddings.embed_documents(texts)

        if len(embedded_query) == 0 or len(embedded_documents) == 0:
            raise ValueError("Embedding 결과가 비어 있습니다.")

        # 유사도 계산 (코사인 유사도)
        scores = np.dot(np.array(embedded_query), np.array(embedded_documents).T)
        
        # scores는 1차원 배열이므로 벡터를 반환하기 위해 처리합니다.
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)  # 2D로 변환하여 처리

        sorted_idx = np.argsort(scores[0])[::-1]  # 높은 점수 순으로 정렬

        # 결과를 저장할 리스트
        results = []

        # 상위 5개 결과를 JSON 형식으로 저장
        for i, idx in enumerate(sorted_idx[:5]):
            result = {
                "질문": query,
                "답변": texts[idx],
                "유사도 점수": float(scores[0][idx])  # 유사도 점수 추가
            }
            results.append(result)

        # JSON 파일로 결과 저장
        with open('search_results_2.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"결과가 'search_results.json' 파일에 저장되었습니다.")
        return sorted_idx

    except Exception as e:
        print(f"검색 중 오류 발생: {e}")
        return None

# 메인 실행 부분
if __name__ == "__main__":
    FILE_PATH = "C:/Users/kowm6/Desktop/test.pdf"  # 파일 경로 직접 입력
    model_name = "intfloat/multilingual-e5-large-instruct"

    # PDF 로드
    documents = load_pdf(FILE_PATH)

    if documents:
        # 텍스트 분리 (separator 제거, chunk_size와 chunk_overlap만 사용)
        texts = split_text(documents, splitter_type="recursive", chunk_size=500, chunk_overlap=0)
        if texts:
            hf_embeddings = initialize_embedding(model_name)
            if hf_embeddings:
                search_query(
                    """
                    2. What are the main differences between the pre-training data and the SFT data in the data distribution visualization of Figure 1?
                
                    """,
                    texts,
                    hf_embeddings,
                )

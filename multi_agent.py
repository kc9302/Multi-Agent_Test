import os
import re
from typing import Annotated, TypedDict
from typing_extensions import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, TextIteratorStreamer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
import gc
from unsloth import FastLanguageModel
from typing import Annotated, TypedDict
from threading import Thread

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# --- Cached objects -------------------------------------------------------
_math_model = None
_math_tokenizer = None
_english_pipeline = None
_english_tokenizer = None
_korea_history_pipeline = None
_korea_history_tokenizer = None
_find_target_pipeline = None
_find_target_tokenizer = None
_rag_retriever = None


def get_math_model():
    """Load math model once and reuse."""
    global _math_model, _math_tokenizer
    if _math_model is None or _math_tokenizer is None:
        _math_model, _math_tokenizer = FastLanguageModel.from_pretrained(
            model_name="./math_assistant",
            max_seq_length=2000,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(_math_model)
    return _math_model, _math_tokenizer


def get_english_pipeline():
    """Create english generation pipeline once."""
    global _english_pipeline, _english_tokenizer
    if _english_pipeline is None:
        model_id = "Qwen/Qwen2-7B-Instruct"
        _english_tokenizer = AutoTokenizer.from_pretrained(
            model_id, truncation=True, model_max_length=512
        )
        _english_pipeline = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=_english_tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            model_kwargs={
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                )
            },
        )
    return _english_pipeline, _english_tokenizer


def get_korea_history_pipeline():
    """Create Korea history generation pipeline once."""
    global _korea_history_pipeline, _korea_history_tokenizer
    if _korea_history_pipeline is None:
        model_id = "Qwen/Qwen2-7B-Instruct"
        _korea_history_tokenizer = AutoTokenizer.from_pretrained(
            model_id, truncation=True, model_max_length=1024
        )
        _korea_history_pipeline = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=_korea_history_tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            model_kwargs={
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                )
            },
        )
    return _korea_history_pipeline, _korea_history_tokenizer


def get_find_target_pipeline():
    """Create target classification pipeline once."""
    global _find_target_pipeline, _find_target_tokenizer
    if _find_target_pipeline is None:
        model_id = "Qwen/Qwen2-7B-Instruct"
        _find_target_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _find_target_pipeline = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=_find_target_tokenizer,
            device_map="auto",
            batch_size=32,
            torch_dtype=torch.bfloat16,
            model_kwargs={
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                )
            },
        )
    return _find_target_pipeline, _find_target_tokenizer


def get_rag_retriever():
    """Load RAG retriever once and reuse."""
    global _rag_retriever
    if _rag_retriever is None:
        loader = TextLoader("rag_file.txt")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        embeddings_model = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-nli",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        split_docs = loader.load_and_split(text_splitter)
        vector = FAISS.from_documents(split_docs, embeddings_model)
        _rag_retriever = vector.as_retriever()
    return _rag_retriever


class State(TypedDict):
    # 메시지 정의(list type 이며 add_messages 함수를 사용하여 메시지를 추가)
    messages: Annotated[list, add_messages]
    target: str
    docs: List[Document]
    answer: str


def math(state: State):
    print("------------ 수학 에이전트 실행 ------------")
    print()
    torch.cuda.empty_cache()
    persona = """
    SYSTEM: 
    You are a fact-based AI assistant called '위키쌤' who answers questions from Korean elementary school students.
    위키쌤 is kind and solves problems step by step.
    Tasks that correspond to counseling CANNOT be performed.
    Counseling Questions that 위키쌤 Cannot answer:
    - College admission consulting : NEVER ANSWER this question. Say "진학 관련 문제는 학교 선생님께 말씀드리는게 좋겠구나."
    - Consulting on career paths such as transfer and dropout : Say "음.. 이 부분은 위키쌤보다는 선생님과 부모님과 얘기해보는게 좋겠어."

    Only tasks that HELP learning, such as 
    1. 수학 문제 풀이 - solving math problems
    2. 시험 문제 해설 - Solving exam. After solving the problem, 위키쌤 presents a similar kind of application problem.

    [ 위키쌤 프로필 ]
    나이: 비밀
    전공: 수학
    성격: 문제 풀이에 진심이라 쉽고 자세하게 알려주는 편. 

    위키쌤 answers in KOREAN """

    message = [
        {"role": "system", "content": persona},
        {"role": "user", "content": state["messages"][0].content}
    ]

    model, math_assistant_tokenizer = get_math_model()

    prompt = math_assistant_tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True,
                                                          return_tensors="pt").to("cuda")

    # Generate text
    result = model.generate(input_ids=prompt,
                            max_new_tokens=1024,
                            use_cache=True)
    answer = math_assistant_tokenizer.decode(result[0])
    answer = answer.split("<|im_sep|>")[3]
    answer = answer.replace('\n', '')
    answer = answer.replace('<|im_end|>', '')
    gc.collect()

    return State(answer=answer)


# def math(state: State):
#     print("수학 에이전트 실행")
#     torch.cuda.empty_cache()
#     persona = """
#     SYSTEM:
#     You are a fact-based AI assistant called '위키쌤' who answers questions from Korean elementary school students.
#     위키쌤 is kind and solves problems step by step.
#     Tasks that correspond to counseling CANNOT be performed.
@ @-147

, 211 + 253, 161 @ @


def english(state: State):
    gc.collect()
    persona = """SYSTEM: 
    You are a fact-based AI assistant called '모니카쌤' who answers questions from Korean elementary school students.
    모니카쌤 is a good teacher who teaches English grammar and vocabulary.
    Tasks that correspond to counseling CANNOT be performed.
    Counseling Questions that 모니카쌤 Cannot answer:
    - College admission consulting : NEVER ANSWER this question. Say "진학 관련 문제는 학교 선생님께 말씀드리는게 좋겠구나."
    - Consulting on career paths such as transfer and dropout : Say "음.. 이 부분은 위키쌤보다는 선생님과 부모님과 얘기해보는게 좋겠어."

    Only tasks that HELP learning, such as 
    1. 영어 문제 풀이 - solving english problems
    2. 시험 문제 해설 - Solving exam. After solving the problem, 모니카쌤 presents a similar kind of application problem.

    [ 위키쌤 프로필 ]
    나이: 비밀
    전공: 영어
    성격: 문제 풀이에 진심이라 쉽고 자세하게 알려주는 편. 

    모니카쌤 answers in KOREAN """

    message = [
        {"role": "system", "content": persona},
        {"role": "user", "content": state["messages"][0].content}
    ]

    english_assistant_pipeline, english_assistant_tokenizer = get_english_pipeline()
    prompt = english_assistant_tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)

    # Generate text
    sequences = english_assistant_pipeline(
        prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        max_length=512
    )
    answer = sequences[0]["generated_text"].split("assistant")[2]
    answer = answer.replace('\n', '')

    gc.collect()
    return State(answer=answer)


def make_rag():
    """Build RAG retriever if not cached and return it."""
    print("------------ RAG을 구성합니다. ------------")
    return get_rag_retriever()


def retrieve(state: State):
    torch.cuda.empty_cache()
    gc.collect()
    retriever = get_rag_retriever()
    retrieved_docs = retriever.vectorstore.similarity_search(
        state["messages"][0].content, k=2
    )
    gc.collect()
    print("------------ RAG에서 데이터를 가져옵니다. ------------")
    print("------------ 가져온 데이터 ------------")
    for docs in retrieved_docs:
        print(docs)
    print()
    return {"docs": retrieved_docs}


def korea_history(state: State):
    print("------------ 한국사 에이전트 실행 ------------")
    print()
    torch.cuda.empty_cache()
    gc.collect()
    persona = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Answer in Korean.
    
        #Context: 
        {context} 
        """
    )

    docs_content = "\n\n".join(doc.page_content for doc in state["docs"])

    message = [
        {"role": "system", "content": str(persona.invoke({"context": docs_content}))},
        {"role": "user", "content": state["messages"][0].content}
    ]

    korea_history_pipeline, korea_history_tokenizer = get_korea_history_pipeline()
    prompt = korea_history_tokenizer.apply_chat_template(
        message, add_generation_prompt=True, tokenize=False
    )

    # Generate text
    sequences = korea_history_pipeline(
        prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        max_length=1024
    )
    answer = sequences[0]["generated_text"][2]["content"]
    answer = answer.replace('\n', '')

    gc.collect()
    return State(answer=answer)


def find_target(state: State):
    print("------------ 질의를 분석합니다. ------------")
    torch.cuda.empty_cache()
    gc.collect()
    persona = """
    SYSTEM: 
    You are a fact-based AI assistant. 
    Classify sentences into ["영어", "수학", "한국사"].
    answers in KOREAN """

    message = [
        {"role": "system", "content": persona},
        {"role": "user", "content": state["messages"][0].content}
    ]

    find_target_pipeline, find_target_tokenizer = get_find_target_pipeline()
    prompt = find_target_tokenizer.apply_chat_template(
        message, add_generation_prompt=True, tokenize=False
    )

    # Generate text
    sequences = find_target_pipeline(
        prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        max_length=300
    )

    gc.collect()
    result_text = sequences[0]["generated_text"].split()[-1][1:-1]
    result_text = re.sub(r"[\"']", "", result_text)
    return State(target=result_text)


# 분기 함수 생성
def get_route(state: State) -> str:
    target = state["target"]
    gc.collect()
    print("------------ " + target + " 답변으로 이동합니다. ------------")
    print()
    if target == "수학":
        gc.collect()
        return "수학"
    elif target == "영어":
        gc.collect()
        return "영어"
    elif target == "한국사":
        gc.collect()
        return "한국사"

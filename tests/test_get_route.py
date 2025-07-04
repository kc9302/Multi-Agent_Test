import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Create dummy modules for heavy dependencies
module_attrs = {
    'torch': {},
    'langchain.text_splitter': {'RecursiveCharacterTextSplitter': object},
    'langchain.document_loaders': {'TextLoader': object},
    'langchain_huggingface': {'HuggingFaceEmbeddings': object},
    'langchain_community.vectorstores': {'FAISS': object},
    'langchain_core.prompts': {'PromptTemplate': object},
    'langchain_core.documents': {'Document': object},
    'langgraph.graph.message': {'add_messages': lambda x: x},
    'unsloth': {'FastLanguageModel': object},
}

for name, attrs in module_attrs.items():
    mod = types.ModuleType(name)
    for attr, val in attrs.items():
        setattr(mod, attr, val)
    sys.modules[name] = mod

# Dummy transformers module with minimal attributes
transformers_dummy = types.ModuleType('transformers')
transformers_dummy.AutoModelForCausalLM = object
transformers_dummy.AutoTokenizer = object
def dummy_pipeline(*args, **kwargs):
    return lambda *a, **k: None
transformers_dummy.pipeline = dummy_pipeline
class BitsAndBytesConfig: pass
transformers_dummy.BitsAndBytesConfig = BitsAndBytesConfig
class TextIteratorStreamer: pass
transformers_dummy.TextIteratorStreamer = TextIteratorStreamer
sys.modules['transformers'] = transformers_dummy

from multi_agent import get_route


def test_get_route_math():
    assert get_route({"target": "수학"}) == "수학"


def test_get_route_english():
    assert get_route({"target": "영어"}) == "영어"


def test_get_route_history():
    assert get_route({"target": "한국사"}) == "한국사"


def test_get_route_unknown():
    assert get_route({"target": "과학"}) is None
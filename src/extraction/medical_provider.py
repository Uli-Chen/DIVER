"""Official DeepSeek provider setup for Medical; credentials stay process-local."""
import os
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
PREFIX = 'tmp_medical_'
BASE_URL = 'https://api.deepseek.com'
MODEL = 'deepseek-v4-flash'


def environment():
    from dotenv import load_dotenv
    load_dotenv(PROJECT / '.env', override=True)
    values = {k: os.environ.get(PREFIX+k, '').strip() for k in ('api_key','api_base','chat_model')}
    if not values['api_key'] or values['api_base'].rstrip('/') != BASE_URL or values['chat_model'] != MODEL:
        raise RuntimeError('Medical requires tmp_medical_api_key/api_base/chat_model for the approved DeepSeek endpoint')
    if os.environ.get('GRAPHRAG_EMBEDDING_MODEL') != 'Qwen/Qwen3-Embedding-8B' or os.environ.get('GRAPHRAG_EMBEDDING_API_BASE','').rstrip('/') != 'https://api.siliconflow.cn/v1':
        raise RuntimeError('Original SiliconFlow embedding index configuration must be retained')
    if not os.environ.get('GRAPHRAG_EMBEDDING_API_KEY'):
        raise RuntimeError('Missing original embedding credential')
    # These aliases exist only in this process and its children; never edit the
    # normal AGEA_/GRAPHRAG_ entries in the shared .env.
    os.environ.update(AGEA_API_KEY=values['api_key'],GRAPHRAG_API_KEY=values['api_key'],
        AGEA_API_BASE=BASE_URL,GRAPHRAG_API_BASE=BASE_URL,AGEA_CHAT_MODEL=MODEL,
        GRAPHRAG_CHAT_MODEL=MODEL,QUERY_GENERATOR=MODEL,AGEA_LLM_PROVIDER='openai_compatible',
        AGEA_THINKING_CONTROL_STYLE='thinking_disabled',AGEA_QUERY_MAX_TOKENS='1024',
        PYTHONPATH=str(PROJECT/'src'),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0')


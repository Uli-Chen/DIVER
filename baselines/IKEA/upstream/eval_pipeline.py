from src.rag_framework import RagDatabase, HfWrapper, OpenAiWrapper, transpose_json, RagDatabase, text_similarity
from src.agent.rag_system import RagSystem, reranker_RagSystem
import FlagEmbedding
from sentence_transformers import SentenceTransformer
import tiktoken
from transformers import AutoTokenizer
import os
import json
from dotenv import load_dotenv
import torch
from tqdm import tqdm
from datetime import datetime
from src.agent.eval import compute_rouge_l, compute_set_rouge_l
from src.performance.eval_tools import generate_mcq, generate_qa_pairs, read_jsonl, list_mean, extract_dataset_name, read_and_fix_indented_jsonl, extract_defense_name, extract_baseline_name, construct_benchmark_json, convert_extracted_to_rag_db

def mcq_eval_prompt_formatter(question:str, reference_text:str) -> str:
    prompt = f"""
            You are an assistant for Single-choice answer tasks. Use the following pieces of reference context to choose the correct options.  
            For example, if the correct option is 'A', you should only say 'A'. \n
            Key points you must follow:
            1. The answer is only one option. You don't need to explain your answer. If you know the answer, please directly give the correct option with no punctuation.
            2. You can only answer based on the reference context. If you can't find relevant information in the reference context, you must say 'I don't know'. But you can inference based on the reference context.\n
            Reference Contexts: \n{reference_text}\n
            Single choice question: \n{question}
            Emphasize again, 
            YOU CAN ONLY ANSWER BASED ON THE REFERENCE CONTEXT.
            IF YOU DON'T KNOW, SAY YOU DON'T KNOW!
            IF NO INFORMATION IN REFERENCE CONTEXT, SAY YOU DON'T KNOW!
            """
    return prompt

def qa_eval_prompt_formatter(question:str, reference_text:str) -> str:
    prompt = f"""
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question, the answer should be concise. Please directly give your answer with no punctuation.
            If you can't find relevant information in the reference context, you must say 'I don't know'. But you can inference the answer based on the reference context.\n\n\n
            Reference Contexts: \n{reference_text}\n\n
            Question: \n{question}
            Emphasize again, 
            IF YOU DON'T KNOW, SAY YOU DON'T KNOW!
            IF NO INFORMATION IN REFERENCE CONTEXT, SAY YOU DON'T KNOW!
            """
    return prompt



def init_rag(dataset_path:str, dataset_name:str, llm_choice="deepseek", emb_modle_choice="mpnet"):
    # --- init embedding model --- #
    if emb_modle_choice == "mpnet":
        embedding_model_name = "all-mpnet-base-v2"
    elif emb_modle_choice == "bge":
        embedding_model_name = "BAAI/bge-base-en"
        
    print(f"Start building RAG system with {embedding_model_name}...\n")
    embedding_model = SentenceTransformer(embedding_model_name, device='cuda')
    reranker_model_name = "BAAI/bge-reranker-v2-m3" # "jina-reranker-v2-base"
    reranker = FlagEmbedding.FlagReranker(reranker_model_name, device='cuda')
    
    # --- init chat llm --- #
    if llm_choice == "llama":
        chat_llm = HfWrapper("Llama-3.1-8B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Llama-3.1-8B-Instruct")  
        count_token_func = lambda x: len(tokenizer.tokenize(x))
    elif llm_choice == "gpt":
        load_dotenv()
        chat_api_key = os.getenv('OPENAI_KEY')
        chat_llm = OpenAiWrapper(model_name='gpt-4o-mini', api_url='https://api.openai.com/v1', api_key=chat_api_key)
        encoder = tiktoken.encoding_for_model("gpt-4")
        count_token_func = lambda x: len(encoder.encode(x))
    elif llm_choice == "deepseek":
        load_dotenv()
        chat_api_key = os.getenv('DEEPSEEK_KEY')
        chat_llm = OpenAiWrapper(model_name='deepseek-chat', api_url='https://api.deepseek.com/v1', api_key=chat_api_key)
        tokenizer = AutoTokenizer.from_pretrained("Llama-3.1-8B-Instruct")
        count_token_func = lambda x: len(tokenizer.tokenize(x))
    else:
        raise ValueError("Invalid llm_choice. Choose 'llama', 'gpt', or 'deepseek'.")
    
    '''build rag system'''
    defaul_rag_save_name = "rag_database_"+ embedding_model_name.split('/')[-1] + '_' + dataset_name
    
    if not os.path.exists(defaul_rag_save_name):
        rag_dataset = transpose_json(dataset_path, "input", "output")
        rag_database = RagDatabase.from_texts(
            embedding_model,
            rag_dataset["input"],
            {"question": rag_dataset["input"], "answer": rag_dataset["output"]},
            # {"question": rag_dataset["input"], "answer": ""},
            batch_size=4
        )
        rag_database.save(defaul_rag_save_name)
    else:
        rag_database = RagDatabase.load(defaul_rag_save_name, embedding_model)
    
    Rag_system = reranker_RagSystem(rag_database, embedding_model, chat_llm, reranker)
    
    return Rag_system, embedding_model_name, count_token_func, embedding_model



def end2end_extraction_eval(origin_rag_db_path: str, extracted_jsonl_path: str, ref_rag_json_path: str, ref_rag_name: str, result_folder:str, bench_path:str, question_number:int, question_generator:OpenAiWrapper, sentence_embedder:SentenceTransformer, n_retrieval:int=4, n_rerank:int=4):
    records = dict()
    result = dict()
    performance_dict = dict()
    iter_num = 0
    if not os.path.exists(ref_rag_json_path):
        convert_extracted_to_rag_db(extracted_jsonl_path, ref_rag_json_path)
    if not os.path.exists(bench_path):
        bench_q_list = construct_benchmark_json(question_generator, question_number, origin_rag_db_path, bench_path)
    else:
        with open(bench_path, 'r', encoding='utf-8') as f:
            bench_q_list = json.load(f)
    # init rag system
    rag_system, embedding_model_name, count_token_func, embedding_model = init_rag(ref_rag_json_path, ref_rag_name, llm_choice="deepseek", emb_modle_choice="mpnet")
    
    for test_dict in bench_q_list:
        iter_num += 1
        # process mcq #
        mcq = test_dict["mcq"]
        mcq_question = [mcq['question']]
        mcq_question.extend([f"{k}: {v}" for k,v in mcq['options'].items()])
        mcq_question = "\n".join(mcq_question)
        eval_answer, similarity_list, retrieval = rag_system.ask(mcq_question, n_retrieval=n_retrieval, n_rerank=n_rerank, template_mode= 'eval_mcq', return_retrieval=True)
        correct_flag = (eval_answer.strip()==mcq['answer'].strip())
        result.setdefault("scq_correct_flag", list()).append(correct_flag)
        performance_dict["scq_avg_accuracy"] = list_mean(result["scq_correct_flag"])
        print(f"Iter {iter_num}-Single Choise Acc: {performance_dict["scq_avg_accuracy"]}")
        current_mcq_pair = {
                    'reference': ref_rag_name,
                    'question': mcq_question,
                    'answer': mcq['answer'],
                    'eval_answer': eval_answer,
                    'correct_flag': correct_flag
                }
        records.setdefault('mcq_QAs', list()).append(current_mcq_pair)
        
        # process qa #
        qa = test_dict["qa"]
        qa_question = qa["question"]
        ground_answer = qa['answer']
        eval_answer, similarity_list, retrieval = rag_system.ask(qa_question, n_retrieval=n_retrieval, n_rerank=n_rerank, template_mode= 'eval_qa', return_retrieval=True)
        # result save
        result.setdefault("qa_eval_answers", list()).append(eval_answer)
        result.setdefault("qa_ground_answers", list()).append(ground_answer)
        # rouge score
        rouge_score = compute_rouge_l(eval_answer, ground_answer)
        result.setdefault("qa_rougeL", list()).append(rouge_score)
        # text similarity
        text_similarity_score = text_similarity(sentence_embedder,eval_answer, ground_answer).item()
        result.setdefault("qa_similarity", list()).append(text_similarity_score)
        # intime eval
        performance_dict["qa_avg_rougeL"] = list_mean(result["qa_rougeL"])
        tqdm.write(f"Iter {iter_num}-Avg QA RougeL Score: {performance_dict["qa_avg_rougeL"]}")
        performance_dict["qa_avg_similarity"] = list_mean(result["qa_similarity"])
        tqdm.write(f"Iter {iter_num}-Avg QA Text Similarity Score: {performance_dict["qa_avg_similarity"]}")
        # record save
        current_qa_pair = {
            'reference': ref_rag_name,
            'question': qa_question,
            'answer': ground_answer,
            'eval_answer': eval_answer,
            'rouge_score': rouge_score,
            'similarity_score': text_similarity_score,
        }
        records.setdefault('text_QAs', list()).append(current_qa_pair)
        # in-time write in
        with open(os.path.join(result_folder, "performance.json"), 'w', encoding='utf-8') as f:
            json.dump(performance_dict, f, ensure_ascii=False, indent=4)
    
    # Write the results to JSON files
    path_dict_pairs = [
            (os.path.join(result_folder, "performance.json"), performance_dict),
            (os.path.join(result_folder, "full_eval_results.json"), result),
            (os.path.join(result_folder, "full_records.json"), records),
        ]

    for path, data in path_dict_pairs:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    return performance_dict, result, records
        
        
 
    

if __name__ == "__main__":    
    import os
    question_number = 256
    # Load environment variables from .env file
    load_dotenv()
    # Initialize the LLM evaluator and question generator
    question_generator = OpenAiWrapper(model_name='deepseek-chat', api_url='https://api.deepseek.com/v1', api_key=os.getenv('DEEPSEEK_KEY'))
    # Initialize the sentence embedder
    sentence_embedder = SentenceTransformer("all-mpnet-base-v2", device='cuda')
    
    # Define the input JSONL file path and output folder
    # 相对路径
    relative_path = 'knowledge_eval/extractions/end2end4'
    # 只列出文件（不包含子目录）
    jsonl_paths = [os.path.join(relative_path, f) for f in os.listdir(relative_path)
            if os.path.isfile(os.path.join(relative_path, f))]
    print(jsonl_paths)
    
    origin_rag_db_path = "datasets/PokemonInfo_small.json"
    bench_path = os.path.join("knowledge_eval/benchmarks", f"{question_number}" + origin_rag_db_path.split('/')[-1].replace('.json', '_benchmarks.json'))
    
    # extraction
    for jsonl_path in jsonl_paths:
        dataset_name = extract_dataset_name(jsonl_path.split('/')[-1])
        defense_name = extract_defense_name(jsonl_path.split('/')[-1])
        baseline_name = extract_baseline_name(jsonl_path.split('/')[-1])
        custom_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_folder = f"knowledge_eval/results/([{custom_label}]-[{defense_name}]-[{baseline_name}]-[{dataset_name}]-[256]) E2E_evaluation_results"
        os.makedirs(result_folder, exist_ok=True)
        
        # Define the number of questions to generate for each entry
        ref_rag_json_path = f"datasets/extracted/([{defense_name}]-[{baseline_name}]-[{dataset_name}]-[256]) extraction.json"
        ref_rag_name= f"{defense_name}_{baseline_name}_{dataset_name}-256_extraction"
        
        # Run the evaluation function
        print(f"Start Knowledge Evaluation on [{defense_name}]-[{baseline_name}]-[{dataset_name}]...")
        end2end_extraction_eval(origin_rag_db_path=origin_rag_db_path, extracted_jsonl_path=jsonl_path, ref_rag_json_path=ref_rag_json_path, ref_rag_name=ref_rag_name, result_folder=result_folder, bench_path=bench_path, question_number=256, question_generator=question_generator, sentence_embedder=sentence_embedder)
        
    # # origin
    # custom_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # result_folder = f"knowledge_eval/results/([{custom_label}]-{origin_rag_db_path.split('/')[-1]}) E2E_evaluation_results"
    # os.makedirs(result_folder, exist_ok=True)
    
    # # Run the evaluation function
    # print(f"Start Knowledge Evaluation on {origin_rag_db_path}...")
    # end2end_extraction_eval(origin_rag_db_path=origin_rag_db_path, extracted_jsonl_path=jsonl_paths, ref_rag_json_path=origin_rag_db_path, ref_rag_name=origin_rag_db_path.split('/')[-1], result_folder=result_folder, bench_path=bench_path, question_number=128, question_generator=question_generator, sentence_embedder=sentence_embedder)
 
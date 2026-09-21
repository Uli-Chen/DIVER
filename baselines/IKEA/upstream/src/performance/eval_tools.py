# 将 ./src 目录添加到 Python 搜索路径中
from locale import strcoll
from operator import length_hint
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import re
from src.rag_framework import OpenAiWrapper
import json
from typing import List, Dict, final
from tqdm import tqdm

# --- MCQ generation --- #
def generate_MCQ_generation_prompt(reference_text:str, num_questions:int) -> str:
    prompt = f"""
                You are given the following reference text:

                {reference_text}

                Please read it carefully. Then create exactly {num_questions} multiple-choice questions based on the information presented.

                For each question:
                1. Write the question clearly.
                2. Provide four answer options labeled "A", "B", "C", "D".
                3. Identify one correct answer from among these options.
                4. Make sure the correct answer is accurate and derived from the reference text.

                Your answer should be in the following JSON array format:

                [
                    {{
                        "question": "Your question here...",
                        "options": {{
                        "A":"Option A",
                        "B":"Option B",
                        "C":"Option C",
                        "D":"Option D"
                        }},
                        "answer": "A"   // Label of the correct option
                    }},
                    ...
                ]

                Each question must be self-contained and refer back to the content of the provided text so that the correct answer is verifiable from that text. Do not include any additional explanations outside the JSON structure.
                """
    
    return prompt
    
    
def parse_mcq_output(response_text: str)->List[Dict[str,str]]:
    """
    解析 LLM 输出的 JSON 格式的多项选择题，并返回一个列表。
    列表中的每个元素是一个字典，包含 question、options 和 answer 三个字段。
    
    :param response_text: 来自 LLM 的 JSON 字符串 (例如：
        [
          {
            "question": "Your question here...",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": "A"
          },
          ...
        ]
    :return: 成功解析后的 MCQ 列表；
             若解析失败或不符合结构，返回空列表。
    """
    json_content = extract_json_codeblock(response_text)
    
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError as e:
        print(json_content)
        print("JSON 解码失败，错误信息：", e)
        return []

    # 检查整体是否为列表
    if not isinstance(data, list):
        print("数据结构错误：最外层应该是一个列表。")
        return []

    parsed_questions = []
    for index, item in enumerate(data):
        # 每个条目必须是字典
        if not isinstance(item, dict):
            print(f"第 {index} 个条目不是字典类型，已跳过。")
            continue
        
        # 检查必要字段
        missing_keys = [k for k in ("question", "options", "answer") if k not in item]
        if missing_keys:
            print(f"第 {index} 个条目缺少必要字段: {missing_keys}，已跳过。")
            continue

        # 检查 options 字段的结构（需为长度为 4 的列表）
        options = item["options"]
        if not isinstance(options, dict) or len(options) != 4:
            print(f"第 {index} 个条目的 'options' 字段无效(必须是含 4 个选项的dict), 已跳过。")
            continue

        # 若通过以上检查，则将有效条目加入列表
        parsed_questions.append(
            {
                "question": item["question"],
                "options": item["options"],
                "answer": item["answer"],
            }
        )
    
    return parsed_questions


def extract_json_codeblock(text: str) -> str:
    """使用正则表达式提取代码块中的JSON内容"""
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # 取最后一个匹配的代码块（通常是最新的响应）
        raw_json = matches[-1].strip()
        # 处理可能存在的行注释
        return "\n".join([line for line in raw_json.split("\n") if not line.strip().startswith("//")])
    return text  # 回退到整个文本


def generate_mcq(llm_generator:OpenAiWrapper, reference_context:str, num_questions:int=1) -> List[Dict[str, str]]:
    mcq_prompt = generate_MCQ_generation_prompt(reference_context, num_questions)
    mcq_list_str = llm_generator.ask(mcq_prompt)
    mcq_list = parse_mcq_output(mcq_list_str)
    
    return mcq_list

# --- generate QA --- #

def generate_qa_prompt(reference_text: str, num_questions: int) -> str:
    """Generate prompt for QA pair generation."""
    return f"""
    Generate {num_questions} question-answer pairs based EXCLUSIVELY on the following text:

    {reference_text}

    Requirements:
    1. Questions must be factual and answerable from the text
    2. Answers must be concise (1-2 sentences) 
    3. Use this JSON format:
    [
        {{
            "question": "clear question",
            "answer": "exact answer from text"
        }}
    ]
    4. Ensure JSON syntax validity
    5. Do NOT include explanations or markdown formatting
    6. Questions should cover different aspects of the text
    """

def parse_qa_response(response_text: str) -> List[Dict[str, str]]:
    """Parse LLM response containing QA pairs."""
    json_str = extract_json_block(response_text)
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {str(e)}")
        return []

    if not isinstance(data, list):
        print("Top-level structure must be a list")
        return []

    valid_pairs = []
    for idx, entry in enumerate(data):
        if not validate_qa_entry(entry, idx):
            continue
        valid_pairs.append({
            "question": entry["question"].strip(),
            "answer": entry["answer"].strip()
        })
    
    return valid_pairs

def validate_qa_entry(entry: Dict, idx: int) -> bool:
    """Validate individual QA entry structure."""
    if not isinstance(entry, dict):
        print(f"Entry {idx}: Invalid type, expected dict")
        return False
    
    required_keys = {"question", "answer"}
    if missing := required_keys - entry.keys():
        print(f"Entry {idx}: Missing keys {missing}")
        return False
    
    if not isinstance(entry["question"], str) or not entry["question"].strip():
        print(f"Entry {idx}: Invalid question")
        return False
        
    if not isinstance(entry["answer"], str) or not entry["answer"].strip():
        print(f"Entry {idx}: Invalid answer")
        return False
    
    return True

def extract_json_block(text: str) -> str:
    """Extract JSON content from markdown code block."""
    pattern = r"(?i)```json\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Handle multiple code blocks by selecting last match
        raw_json = matches[-1].strip()
        # Remove inline comments
        return re.sub(r"//.*", "", raw_json)
    return text

def generate_qa_pairs(llm_client:OpenAiWrapper, references: List[str], num_pairs: int = 1) -> List[Dict[str, str]]:
    """Generate QA pairs from reference texts."""
    context = "\n".join(references)
    prompt = generate_qa_prompt(context, num_pairs)
    response = llm_client.ask(prompt)
    return parse_qa_response(response)



# --- utils --- #
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error on line {i}: {e}")
    return data

def read_and_fix_indented_jsonl(input_path: str, output_path: str=None) -> List[Dict]:
    data_list = []
    buffer = ""

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            buffer += line
            try:
                obj = json.loads(buffer)
                data_list.append(obj)
                buffer = ""
            except json.JSONDecodeError:
                continue  # JSON 未拼接完整，继续读下一行
    
    if output_path:
        # 写入修复后的标准 JSONL 文件
        with open(output_path, "w", encoding="utf-8") as outfile:
            for item in data_list:
                outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    return data_list

def extract_dataset_name(filename):
    """
    从文件名中提取第一个圆括号 () 内最后一个方括号 [] 中的数据集名称。
    """
    match = re.search(r'\(([^()]*)\)', filename)
    if match:
        inner_content = match.group(1)
        brackets = re.findall(r'\[([^\[\]]+)\]', inner_content)
        if brackets:
            return brackets[-1]
    return None

import re

def extract_defense_name(filename):
    """
    从文件名中提取第一个圆括号 () 后第一个方括号 [] 中的内容。
    """
    match = re.search(r'\([^()]*\)(.*)', filename)  # 找到第一个 () 后的部分
    if match:
        remainder = match.group(1)
        bracket_match = re.search(r'\[([^\[\]]+)\]', remainder)
        if bracket_match:
            return bracket_match.group(1)
    return None

def extract_baseline_name(filename):
    """
    从文件名中提取第一个圆括号 () 后第二个方括号 [] 中的内容。
    """
    match = re.search(r'\([^()]*\)(.*)', filename)  # 提取第一个圆括号后的内容
    if match:
        remainder = match.group(1)
        brackets = re.findall(r'\[([^\[\]]+)\]', remainder)  # 找出所有方括号内容
        if len(brackets) >= 2:
            return brackets[1]  # 返回第二个方括号内容（索引从0开始）
    return None

def list_mean(list:List):
    return sum(list)/len(list) if len(list)>0 else 0

# End to end rag evaluation
import random
def sample_json(input_path, n) -> List[str]:
    # 加载 JSON 文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 如果是 .jsonl 格式则换成逐行读取

    # 随机采样 n 条
    sampled = random.sample(data, min(n, len(data)))

    # 构造目标格式
    result = [item['input'] for item in sampled]

    return result

def construct_benchmark_json(llm_generator: OpenAiWrapper, n:int, input_path:str, output_path:str=None) -> List[dict[dict[str]]]:# -> list:# -> list:
    content_list = sample_json(input_path, n)
    final_list = []
    for content in content_list:
        bench_dict={"mcq":{}, "qa":{}}
        bench_dict["mcq"] = generate_mcq(llm_generator, content, 1)[0]
        bench_dict["qa"] = generate_qa_pairs(llm_generator, [content], 1)[0]
        final_list.append(bench_dict)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_list, f, ensure_ascii=False)
        print(f"Benchmark JSON saved to {output_path}")
    return final_list

def convert_extracted_to_rag_db(extracted_jsonl:str, output_json:str):
    input_path = extracted_jsonl    # 原始 jsonl 文件路径
    output_path = output_json   # 目标 json 文件路径

    results = []
    dicts_ls = read_and_fix_indented_jsonl(extracted_jsonl)
    for i in tqdm(range(len(dicts_ls))):
        data = dicts_ls[i]
        if data.get("reject_flag", False):  # 跳过 reject_flag 为 true 的项
            continue
        answer = data.get("answer", "")
        results.append({
            "instruction": "",
            "input": answer,
            "output": ""
        })
    # 保存为 JSON 数组格式
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)

    print(f"转换完成，共保存 {len(results)} 条记录。")

if __name__ == "__main__":
    filename = "U-([2025-04-10 02:17:50]-[bge-base-en]-[llama]-[HealthCareMagic-100k]) [implicit]-[softmax]-[with_mutation]-[256] mutation_pipeline_result.jsonl"
    print(extract_dataset_name(filename))  # 输出: HealthCareMagic-100k

# if __name__ == "__main__":
#     sample_text = ["""
#     The Nile is a major north-flowing river in northeastern Africa,
#     and is commonly regarded as the longest river in the world.
#     It is approximately 6,650 km long and runs through several countries,
#     including Uganda, Ethiopia, and Egypt, before it empties into the Mediterranean Sea.
#     """]
#     retrievals = [
#         "“Yes,” said Hermione, now turning the fragile pages as if examining rotting entrails, “because it warns Dark wizards how strong they have to make the enchantments on them. From all that I’ve read, what Harry did to Riddle’s diary was one of the few really foolproof ways of destroying a Horcrux.” What significance does the destruction of Harry's diary hold?The destruction of Harry's diary was an important event because it was one of the few successful ways to destroy a Horcrux. This warned Dark wizards of the strength required to protect their own Horcruxes.",
#         "“The diary,” said Riddle. `My diary. Little Ginny’s been writing in it for months and months, telling me all her pitiful worries and woes — how her brothers tease her, how she had to come to school with secondhand robes and books, how —” Riddle’s eyes glinted “— how she didn’t think famous, good, great Harry Potter would ever like her…” How does Tom Riddle use the information from the diary?Tom Riddle uses the information from the diary to learn about Ginny Weasley's vulnerabilities and weaknesses.",
#         "It wasn’t until they had reached Professor Flitwick’s class that Harry noticed something rather odd about Riddle’s diary. All his other books were drenched in scarlet ink. The diary, however, was as clean as it had been before the ink bottle had smashed all over it. He tried to point this out to Ron, but Ron was having trouble with his wand again; large purple bubbles were blossoming out of the end, and he wasn’t much interested in anything else. How does Harry's diary compare to his other books?Harry notices that his diary is clean while his other books are soaked in scarlet ink.",
#         "Harry couldn’t explain, even to himself, why he didn’t just throw Riddle’s diary away. The fact was that even though he knew the diary was blank, he kept absentmindedly picking it up and turning the pages, as though it were a story he wanted to finish. And while Harry was sure he had never heard the name T. M. Riddle before, it still seemed to mean something to him, almost as though Riddle was a friend he’d had when he was very small, and had half-forgotten. But this was absurd. He’d never had friends before Hogwarts, Dudley had made sure of that. Why can't Harry bring himself to get rid of Tom Marvolo Riddle's diary?Despite knowing the diary is blank, Harry keeps picking it up and reading it, suggesting an unconscious connection to Riddle, possibly from a past memory he's forgotten."
#     ]
#     chat_api_key = "YOUR_API_KEY"
#     chat_llm = OpenAiWrapper(model_name='deepseek-chat', api_url='https://api.deepseek.com/v1', api_key=chat_api_key)
#     qs_ls = generate_mcq(chat_llm,retrievals,4)
#     qa_ls = generate_qa_pairs(chat_llm,retrievals,4)
#     print(qa_ls)
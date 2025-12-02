import json
import os
import sys
import requests
from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables (e.g. OPENAI_API_KEY)
load_dotenv()

# --- Internalized DictionaryLoader ---

@dataclass
class DictionaryEntry:
    """Dictionary entry: ID and related surface forms"""
    id: str
    preferred_label: str
    synonyms: List[str]
    
    def all_labels(self) -> List[str]:
        """Return all surface forms (preferred + synonyms)"""
        labels = [self.preferred_label]
        labels.extend(self.synonyms)
        return [label.strip() for label in labels if label.strip()]

class DictionaryLoader:
    """Biomedical Dictionary Loader (MeSH, HGNC, NCBIGene) - API Only Version"""
    
    def __init__(self):
        pass
    
    def get_entry(self, id_str: str) -> Optional[DictionaryEntry]:
        """Return entry for ID (Always fetch from API)"""
        id_str = id_str.strip()
        if not id_str:
            return None
        return self._fetch_from_api(id_str)

    def _fetch_from_api(self, id_str: str) -> Optional[DictionaryEntry]:
        """Fetch ID info from external APIs"""
        try:
            if id_str.lower().startswith('mesh:'):
                # MeSH API via E-utilities (Two-step: esearch -> esummary)
                mesh_id = id_str.split(':')[-1]
                
                # Step 1: Get UID from D-number
                search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=mesh&term={mesh_id}&retmode=json&retmax=1"
                resp = requests.get(search_url, timeout=5)
                uid = None
                
                if resp.status_code == 200:
                    s_data = resp.json()
                    id_list = s_data.get("esearchresult", {}).get("idlist", [])
                    if id_list:
                        uid = id_list[0]
                
                if uid:
                    # Step 2: Get details using UID
                    summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=mesh&id={uid}&retmode=json"
                    resp = requests.get(summary_url, timeout=5)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        result_block = data.get('result', {})
                        if uid in result_block:
                            info = result_block[uid]
                            # ds_meshterms contains synonyms
                            mesh_terms = info.get('ds_meshterms', [])
                            
                            # The first term is usually the preferred one, but let's check
                            preferred = mesh_terms[0] if mesh_terms else mesh_id
                            synonyms = mesh_terms[1:] if len(mesh_terms) > 1 else []
                            
                            return DictionaryEntry(id=id_str, preferred_label=preferred, synonyms=synonyms)
                
                # Fallback to id.nlm.nih.gov if E-utilities fails
                url_fallback = f"https://id.nlm.nih.gov/mesh/{mesh_id}.json"
                resp = requests.get(url_fallback, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    label = ""
                    if 'label' in data:
                        label = data['label']['@value']
                    elif 'http://www.w3.org/2000/01/rdf-schema#label' in data:
                        label = data['http://www.w3.org/2000/01/rdf-schema#label']['@value']
                    
                    if label:
                        return DictionaryEntry(id=id_str, preferred_label=label, synonyms=[])
                        
            elif id_str.lower().startswith('hgnc:'):
                # HGNC REST API
                url = f"https://rest.genenames.org/fetch/hgnc_id/{id_str}"
                headers = {'Accept': 'application/json'}
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data['response']['numFound'] > 0:
                        doc = data['response']['docs'][0]
                        symbol = doc.get('symbol', '')
                        alias_symbol = doc.get('alias_symbol', [])
                        alias_name = doc.get('alias_name', [])
                        synonyms = alias_symbol + alias_name
                        return DictionaryEntry(id=id_str, preferred_label=symbol, synonyms=synonyms)

            elif id_str.lower().startswith('geneid:'):
                # NCBI E-utilities
                gene_id = id_str.split(':')[-1]
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id={gene_id}&retmode=json"
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if 'result' in data and gene_id in data['result']:
                        info = data['result'][gene_id]
                        name = info.get('name', '')
                        other_aliases = info.get('otheraliases', '')
                        synonyms = [s.strip() for s in other_aliases.split(',') if s.strip()] if other_aliases else []
                        return DictionaryEntry(id=id_str, preferred_label=name, synonyms=synonyms)
                        
        except Exception as e:
            print(f"API Fetch Error for {id_str}: {e}")
            
        return None

# --- Internalized Pipeline Components ---

class PipelineState(TypedDict):
    qid: str
    text: str  # Input context + question
    entities: List[str]  # Extracted entity names
    candidates: Dict[str, List[Dict[str, str]]]  # term -> list of {id, label}
    resolved_ids: List[str]  # Final selected IDs
    final_answer: str

def node_reason(state: PipelineState) -> PipelineState:
    """Generate final answer using resolved IDs and definitions."""
    print("--- [Node] Reason ---")
    
    ids = state["resolved_ids"]
    text = state["text"]
    
    # 1. Fetch Definitions for Grounding
    dl = DictionaryLoader()
    definitions = []
    for pid in ids:
        entry = dl.get_entry(pid)
        if entry:
            # Provide ID, Name, and Synonyms to help the model identify the entity
            labels = ", ".join(entry.all_labels()[:5])
            definitions.append(f"ID: {pid} | Name: {entry.preferred_label} | Synonyms: {labels}")
            
    knowledge_context = "\n".join(definitions) if definitions else "No specific entity definitions found."
    
    # 2. Reasoning with LLM
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    
    prompt = (
        f"Context & Question:\n\"\"\"\n{text}\n\"\"\"\n\n"
        f"Grounded Knowledge (Candidate Entities):\n{knowledge_context}\n\n"
        "Task: Identify the specific entity that answers the question based on the Context and Grounded Knowledge.\n"
        "The answer is likely one of the entities listed in the Grounded Knowledge.\n"
        "You MUST provide the Entity ID if the answer matches a candidate.\n"
        "If none of the candidates match, provide the best answer you can.\n\n"
        "Required Output Format:\n"
        "Final Answer: [ID] Preferred Label\n"
        "Explanation: Brief explanation of why this entity is the answer."
    )
    
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        final_answer = resp.content.strip()
    except Exception as e:
        print(f"Reasoning Error: {e}")
        final_answer = "Error in reasoning"
        
    return {"final_answer": final_answer}

def load_dataset(silver_path: str, pubmed_path: str) -> List[Dict[str, Any]]:
    print(f"Loading Silver Standard from {silver_path}...")
    with open(silver_path, "r", encoding="utf-8") as f:
        silver_data = json.load(f)
        
    print(f"Loading PubMedQA text from {pubmed_path}...")
    pubmed_text_map = {}
    with open(pubmed_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            # Construct text as Context + Question
            text = f"Context: {entry['context']}\nQuestion: {entry['question']}"
            pubmed_text_map[entry['qid']] = text
            
    # Merge text into silver data
    data = []
    for item in silver_data:
        qid = item['qid']
        if qid in pubmed_text_map:
            # Extract resolved_ids from the first step (assuming single step for now or taking all)
            # Silver standard structure: steps -> [{ids: [...]}]
            resolved_ids = []
            if item.get('steps'):
                for step in item['steps']:
                    resolved_ids.extend(step.get('ids', []))
            
            # Deduplicate
            resolved_ids = list(set(resolved_ids))
            
            data.append({
                "qid": qid,
                "text": pubmed_text_map[qid],
                "resolved_ids": resolved_ids,
                "gold_final_answer": item.get("final_answer"), # This is likely an ID
                "question": pubmed_text_map[qid].split("Question: ")[1] if "Question: " in pubmed_text_map[qid] else ""
            })
            
    print(f"Loaded {len(data)} examples for evaluation.")
    return data

def node_reason_baseline(state: PipelineState) -> PipelineState:
    """
    Baseline reasoning: Provided with Candidate IDs but NO definitions/synonyms.
    This tests if the *content* of the knowledge (synonyms) helps, or just having the list is enough.
    """
    ids = state["resolved_ids"]
    text = state["text"]
    
    # Baseline: Just list ID and Name, NO Synonyms/Definitions
    dl = DictionaryLoader()
    candidates = []
    for pid in ids:
        entry = dl.get_entry(pid)
        if entry:
            candidates.append(f"ID: {pid} | Name: {entry.preferred_label}")
            
    knowledge_context = "\n".join(candidates) if candidates else "No candidates found."
    
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    
    prompt = (
        f"Context & Question:\n\"\"\"\n{text}\n\"\"\"\n\n"
        f"Candidates:\n{knowledge_context}\n\n"
        "Task: Identify the specific entity that answers the question based on the Context.\n"
        "Select the best matching entity from the Candidates list.\n"
        "You MUST provide the Entity ID if the answer matches a candidate.\n\n"
        "Required Output Format:\n"
        "Final Answer: [ID] Preferred Label\n"
        "Explanation: Brief explanation."
    )
    
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        final_answer = resp.content.strip()
    except Exception as e:
        print(f"Reasoning Error: {e}")
        final_answer = "Error"
        
    return {"final_answer": final_answer}

def evaluate():
    silver_path = "output/silver_standard.json"
    pubmed_path = "output/pubmedqa_filtered.jsonl"
    
    data = load_dataset(silver_path, pubmed_path)
    
    results = []
    
    print("\n--- Starting Evaluation (Entity-Aware QA) ---")
    for i, example in enumerate(data):
        print(f"Processing {i+1}/{len(data)}: {example['qid']}")
        
        state: PipelineState = {
            "qid": example['qid'],
            "text": example['text'],
            "entities": [],
            "candidates": {},
            "resolved_ids": example['resolved_ids'],
            "final_answer": ""
        }
        
        # 1. Run Grounded Reasoning (node_reason) - Has Synonyms
        try:
            grounded_output = node_reason(state)
            grounded_ans = grounded_output["final_answer"]
        except Exception as e:
            print(f"Error in node_reason: {e}")
            grounded_ans = "Error"

        # 2. Run Baseline Reasoning (node_reason_baseline) - No Synonyms
        try:
            baseline_output = node_reason_baseline(state)
            baseline_ans = baseline_output["final_answer"]
        except Exception as e:
            print(f"Error in node_reason_baseline: {e}")
            baseline_ans = "Error"
            
        results.append({
            "qid": example['qid'],
            "question": example['question'],
            "gold_answer": example['gold_final_answer'],
            "grounded_answer": grounded_ans,
            "baseline_answer": baseline_ans
        })
        
    # Analysis
    print("\n--- Evaluation Results ---")
    correct_grounded = 0
    correct_baseline = 0
    
    import re
    def extract_id(ans_str):
        # Look for [mesh:...] or just mesh:...
        match = re.search(r"\[(mesh:[A-Z0-9]+)\]", ans_str, re.IGNORECASE)
        if match: return match.group(1).lower()
        # Fallback: look for mesh:D...
        match = re.search(r"(mesh:[A-Z0-9]+)", ans_str, re.IGNORECASE)
        if match: return match.group(1).lower()
        return "none"
            
    for res in results:
        g_id = extract_id(res['grounded_answer'])
        b_id = extract_id(res['baseline_answer'])
        gold = res['gold_answer'].lower()
        
        if g_id == gold: correct_grounded += 1
        if b_id == gold: correct_baseline += 1
        
        print(f"QID: {res['qid']}")
        print(f"  Gold: {res['gold_answer']}")
        print(f"  Grounded: {g_id} (Raw: {res['grounded_answer'][:50]}...)")
        print(f"  Baseline: {b_id} (Raw: {res['baseline_answer'][:50]}...)")
        print("-" * 30)
        
    print(f"\nTotal Examples: {len(results)}")
    print(f"Grounded Accuracy (Exact Match): {correct_grounded}/{len(results)} ({correct_grounded/len(results)*100:.2f}%)")
    print(f"Baseline Accuracy (Exact Match): {correct_baseline}/{len(results)} ({correct_baseline/len(results)*100:.2f}%)")
    
    # Save results
    with open("output/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to output/evaluation_results.json")

if __name__ == "__main__":
    evaluate()

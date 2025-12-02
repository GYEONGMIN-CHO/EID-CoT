from __future__ import annotations

from typing import List, Optional, TypedDict, Dict
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.normalization.dictionaries import DictionaryLoader
from src.clients.openai_client import OpenAIEntityExtractor

# Load env
load_dotenv()

# --- State Definition ---
class PipelineState(TypedDict):
    qid: str
    text: str  # Input context + question
    entities: List[str]  # Extracted entity names
    candidates: Dict[str, List[Dict[str, str]]]  # term -> list of {id, label}
    resolved_ids: List[str]  # Final selected IDs
    final_answer: str

# --- Nodes ---

def node_extract(state: PipelineState) -> PipelineState:
    """Extract entity names from text."""
    print("--- [Node] Extract ---")
    extractor = OpenAIEntityExtractor()
    entities = extractor.extract_entities(state["text"])
    return {"entities": entities}

def node_retrieve(state: PipelineState) -> PipelineState:
    """Retrieve candidates for each entity."""
    print("--- [Node] Retrieve ---")
    dl = DictionaryLoader()
    candidates = {}
    
    for term in state["entities"]:
        # Fetch Top-5 candidates
        cands = dl.search_candidates(term, limit=5)
        candidates[term] = cands
            
    return {"candidates": candidates}

def node_disambiguate(state: PipelineState) -> PipelineState:
    """Select the best ID among candidates based on context."""
    print("--- [Node] Disambiguate ---")
    
    resolved = []
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    
    for term, cands in state["candidates"].items():
        if not cands:
            continue
        
        if len(cands) == 1:
            # Only one candidate, accept it
            resolved.append(cands[0]["id"])
            continue
            
        # Multiple candidates: Ask LLM
        # Format candidates for prompt
        cand_str = "\n".join([f"{i+1}. {c['label']} (ID: {c['id']}, Source: {c['source']})" for i, c in enumerate(cands)])
        
        prompt = (
            f"Context: \"{state['text']}\"\n\n"
            f"Entity Term: \"{term}\"\n\n"
            f"Candidates:\n{cand_str}\n"
            f"{len(cands)+1}. None of the above\n\n"
            "Task: Identify which candidate matches the Entity Term in the given Context.\n"
            "If none match perfectly or the candidates are irrelevant, choose 'None of the above'.\n"
            "Output ONLY the ID of the selected candidate. If 'None of the above', output 'NONE'."
        )
        
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            selected_id = resp.content.strip()
            
            if "NONE" in selected_id.upper():
                continue
            
            # Simple validation: check if selected_id is in candidates
            valid_ids = {c['id'] for c in cands}
            # Handle cases where LLM might output extra text
            import re
            match = re.search(r"(mesh:D\d+|HGNC:\d+|GeneID:\d+)", selected_id)
            if match:
                clean_id = match.group(1)
                if clean_id in valid_ids:
                    resolved.append(clean_id)
                else:
                    # Fallback: pick first if LLM hallucinated an ID, but maybe we should skip?
                    # Let's skip if it picked something weird to be safe.
                    pass
            else:
                # Fallback
                pass
                
        except Exception as e:
            print(f"Disambiguation Error for {term}: {e}")
            # If error, maybe skip or pick first? Let's skip to be safe.
            pass
        
    return {"resolved_ids": list(set(resolved))}

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

# --- Graph Construction ---

def build_graph():
    workflow = StateGraph(PipelineState)
    
    workflow.add_node("extract", node_extract)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("disambiguate", node_disambiguate)
    workflow.add_node("reason", node_reason)
    
    workflow.set_entry_point("extract")
    
    workflow.add_edge("extract", "retrieve")
    workflow.add_edge("retrieve", "disambiguate")
    workflow.add_edge("disambiguate", "reason")
    workflow.add_edge("reason", END)
    
    return workflow.compile()

if __name__ == "__main__":
    # Test run
    app = build_graph()
    initial_state = {
        "qid": "test-1",
        "text": "The patient was diagnosed with Amblyopia.",
        "entities": [],
        "candidates": {},
        "resolved_ids": [],
        "final_answer": ""
    }
    result = app.invoke(initial_state)
    print("\n=== Result ===")
    print(f"Entities: {result['entities']}")
    print(f"Resolved IDs: {result['resolved_ids']}")

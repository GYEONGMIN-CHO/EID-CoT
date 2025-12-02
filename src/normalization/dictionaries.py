from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from src.utils.config import load_config


@dataclass
class DictionaryEntry:
    """사전 엔트리: ID와 관련된 모든 표면형들을 저장"""
    id: str
    preferred_label: str
    synonyms: List[str]
    
    def all_labels(self) -> List[str]:
        """모든 표면형을 반환 (preferred + synonyms)"""
        labels = [self.preferred_label]
        labels.extend(self.synonyms)
        return [label.strip() for label in labels if label.strip()]


class DictionaryLoader:
    """생물의학 사전 로더 (MeSH, HGNC, NCBIGene)"""
    
    def __init__(self):
        self.config = load_config()
        self._mesh_cache: Optional[Dict[str, DictionaryEntry]] = None
        self._hgnc_cache: Optional[Dict[str, DictionaryEntry]] = None
        self._ncbigene_cache: Optional[Dict[str, DictionaryEntry]] = None
        self._extra_synonyms_loaded = False
    
    def load_mesh(self, tsv_path: Optional[str] = None) -> Dict[str, DictionaryEntry]:
        """MeSH 사전 로드
        
        TSV 포맷: id,preferred_label,synonyms
        예: mesh:D003110,Colorectal Neoplasms,"colon cancer,colorectal cancer"
        """
        if self._mesh_cache is not None:
            return self._mesh_cache
            
        path = tsv_path or self.config.dict_mesh_tsv
        if not path or not os.path.exists(path):
            print(f"Warning: MeSH 사전 파일이 없습니다: {path}")
            return {}
            
        entries = {}
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                entry_id = row['id'].strip()
                preferred = row['preferred_label'].strip()
                synonyms_str = row.get('synonyms', '').strip()
                
                # synonyms 파싱 (쉼표로 구분)
                synonyms = []
                if synonyms_str:
                    synonyms = [s.strip() for s in synonyms_str.split(',') if s.strip()]
                
                entries[entry_id] = DictionaryEntry(
                    id=entry_id,
                    preferred_label=preferred,
                    synonyms=synonyms
                )
        
        self._mesh_cache = entries
        print(f"MeSH 사전 로드 완료: {len(entries)} 엔트리")
        return entries
    
    def load_hgnc(self, tsv_path: Optional[str] = None) -> Dict[str, DictionaryEntry]:
        """HGNC 사전 로드
        
        TSV 포맷: id,symbol,synonyms
        예: HGNC:1097,TP53,"p53,TRP53"
        """
        if self._hgnc_cache is not None:
            return self._hgnc_cache
            
        path = tsv_path or self.config.dict_hgnc_tsv
        if not path or not os.path.exists(path):
            print(f"Warning: HGNC 사전 파일이 없습니다: {path}")
            return {}
            
        entries = {}
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                entry_id = row['id'].strip()
                symbol = row['symbol'].strip()
                synonyms_str = row.get('synonyms', '').strip()
                
                # synonyms 파싱
                synonyms = []
                if synonyms_str:
                    synonyms = [s.strip() for s in synonyms_str.split(',') if s.strip()]
                
                entries[entry_id] = DictionaryEntry(
                    id=entry_id,
                    preferred_label=symbol,
                    synonyms=synonyms
                )
        
        self._hgnc_cache = entries
        print(f"HGNC 사전 로드 완료: {len(entries)} 엔트리")
        return entries
    
    def load_ncbigene(self, tsv_path: Optional[str] = None) -> Dict[str, DictionaryEntry]:
        """NCBIGene 사전 로드
        
        TSV 포맷: id,symbol,synonyms
        예: GeneID:7157,TP53,"p53,TRP53"
        """
        if self._ncbigene_cache is not None:
            return self._ncbigene_cache
            
        path = tsv_path or self.config.dict_ncbigene_tsv
        if not path or not os.path.exists(path):
            print(f"Warning: NCBIGene 사전 파일이 없습니다: {path}")
            return {}
            
        entries = {}
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                entry_id = row['id'].strip()
                symbol = row['symbol'].strip()
                synonyms_str = row.get('synonyms', '').strip()
                
                # synonyms 파싱
                synonyms = []
                if synonyms_str:
                    synonyms = [s.strip() for s in synonyms_str.split(',') if s.strip()]
                
                entries[entry_id] = DictionaryEntry(
                    id=entry_id,
                    preferred_label=symbol,
                    synonyms=synonyms
                )
        
        self._ncbigene_cache = entries
        print(f"NCBIGene 사전 로드 완료: {len(entries)} 엔트리")
        return entries
    
    def build_inverted_index(self) -> List[Dict]:
        """BM25용 역인덱스 구축
        
        Returns:
            List[Dict]: 각 문서는 {'id': str, 'text': str} 형태
        """
        docs = []
        
        # 필요 시 확장 동의어 로드
        self._maybe_load_extra_synonyms()

        # MeSH 추가
        mesh_entries = self.load_mesh()
        for entry in mesh_entries.values():
            for label in entry.all_labels():
                docs.append({
                    'id': entry.id,
                    'text': label.lower().strip(),
                    'type': 'mesh'
                })
        
        # HGNC 추가
        hgnc_entries = self.load_hgnc()
        for entry in hgnc_entries.values():
            for label in entry.all_labels():
                docs.append({
                    'id': entry.id,
                    'text': label.lower().strip(),
                    'type': 'hgnc'
                })
        
        # NCBIGene 추가
        ncbigene_entries = self.load_ncbigene()
        for entry in ncbigene_entries.values():
            for label in entry.all_labels():
                docs.append({
                    'id': entry.id,
                    'text': label.lower().strip(),
                    'type': 'ncbigene'
                })
        
        print(f"역인덱스 구축 완료: {len(docs)} 문서")
        return docs

    def get_entry(self, id_str: str) -> Optional[DictionaryEntry]:
        """ID에 해당하는 엔트리를 반환 (항상 API 조회)"""
        id_str = id_str.strip()
        if not id_str:
            return None
            
        # 로컬 캐시 무시하고 항상 API 조회
        return self._fetch_from_api(id_str)

    def _fetch_from_api(self, id_str: str) -> Optional[DictionaryEntry]:
        """외부 API를 통해 ID 정보 조회"""
        import requests
        
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

    def search_candidates(self, term: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        용어(Term)에 대한 후보 ID 리스트를 반환 (Top-K).
        Returns: [{'id': 'mesh:D0000', 'label': 'Term', 'source': 'MeSH'}, ...]
        """
        candidates = []
        term = term.strip()
        if not term:
            return []
            
        import requests
        
        # 1. MeSH Search
        try:
            # esearch
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=mesh&term={term}&retmode=json&retmax={limit}"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                uids = data.get("esearchresult", {}).get("idlist", [])
                if uids:
                    uids_str = ",".join(uids)
                    sum_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=mesh&id={uids_str}&retmode=json"
                    s_resp = requests.get(sum_url, timeout=3)
                    if s_resp.status_code == 200:
                        s_data = s_resp.json()
                        result = s_data.get('result', {})
                        for uid in uids:
                            if uid in result:
                                item = result[uid]
                                mesh_ui = item.get('ds_meshui', '')
                                terms = item.get('ds_meshterms', [])
                                label = terms[0] if terms else mesh_ui
                                
                                if mesh_ui:
                                    candidates.append({
                                        "id": f"mesh:{mesh_ui}",
                                        "label": label,
                                        "source": "MeSH"
                                    })
        except Exception as e:
            print(f"MeSH Candidate Search Error: {e}")

        # 2. HGNC Search
        try:
            url = f"https://rest.genenames.org/search/{term}"
            headers = {'Accept': 'application/json'}
            resp = requests.get(url, headers=headers, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                docs = data.get('response', {}).get('docs', [])
                for doc in docs[:limit]:
                    candidates.append({
                        "id": doc['hgnc_id'],
                        "label": doc['symbol'],
                        "source": "HGNC"
                    })
        except Exception as e:
            print(f"HGNC Candidate Search Error: {e}")
            
        return candidates

    def search_id_by_term(self, term: str) -> Optional[str]:
        """용어(Term)를 검색하여 가장 적절한 표준 ID를 반환"""
        term = term.strip()
        if not term:
            return None
            
        import requests
        
        # 1. MeSH Search
        try:
            # E-utilities esearch
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=mesh&term={term}&retmode=json&retmax=1"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                id_list = data.get("esearchresult", {}).get("idlist", [])
                if id_list:
                    uid = id_list[0]
                    # Get UI (D-number) from summary
                    sum_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=mesh&id={uid}&retmode=json"
                    s_resp = requests.get(sum_url, timeout=3)
                    if s_resp.status_code == 200:
                        s_data = s_resp.json()
                        if 'result' in s_data and uid in s_data['result']:
                            mesh_ui = s_data['result'][uid].get('ds_meshui', '')
                            if mesh_ui:
                                return f"mesh:{mesh_ui}"
        except Exception as e:
            print(f"MeSH Search Error for '{term}': {e}")

        # 2. HGNC Search
        try:
            url = f"https://rest.genenames.org/search/{term}"
            headers = {'Accept': 'application/json'}
            resp = requests.get(url, headers=headers, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data['response']['numFound'] > 0:
                    # Return the first hit's HGNC ID
                    return data['response']['docs'][0]['hgnc_id']
        except Exception as e:
            print(f"HGNC Search Error for '{term}': {e}")

        # 3. NCBI Gene Search (Fallback)
        try:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&term={term}[Gene Name]&retmode=json&retmax=1"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                id_list = data.get("esearchresult", {}).get("idlist", [])
                if id_list:
                    return f"GeneID:{id_list[0]}"
        except Exception as e:
            print(f"NCBI Gene Search Error for '{term}': {e}")
            
        return None

    def _maybe_load_extra_synonyms(self) -> None:
        """추가 동의어 TSV를 각 사전에 주입 (선택적)"""
        if self._extra_synonyms_loaded:
            return
        path = self.config.dict_extra_synonyms_tsv
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    entry_id = row['id'].strip()
                    synonyms_str = row.get('synonyms', '').strip()
                    if not synonyms_str:
                        continue
                    synonyms = [s.strip() for s in synonyms_str.split(',') if s.strip()]
                    # 대상 사전을 찾아 주입
                    if entry_id.lower().startswith('mesh:'):
                        entries = self.load_mesh()
                    elif entry_id.lower().startswith('hgnc:'):
                        entries = self.load_hgnc()
                    elif entry_id.lower().startswith('geneid:'):
                        entries = self.load_ncbigene()
                    else:
                        continue
                    if entry_id in entries:
                        # 중복 없이 병합
                        before = set(entries[entry_id].synonyms)
                        for s in synonyms:
                            if s not in before:
                                entries[entry_id].synonyms.append(s)
            self._extra_synonyms_loaded = True
            print("추가 동의어 로드 완료")
        except Exception as e:
            print(f"Warning: 추가 동의어 로드 실패: {e}")




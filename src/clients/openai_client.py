from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.config import load_config

ID_PATTERN = re.compile(r"(mesh:D\d{6}|HGNC:\d+|GeneID:\d+|DB\d{5})")


class OpenAIEntityExtractor:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1-nano") -> None:
        # Ensure .env is loaded
        load_dotenv()
        cfg = load_config()
        key = api_key or (cfg.openai_api_key if cfg else None) or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=key)
        self.model = model

    def extract_entities(self, text: str, extra_instructions: Optional[str] = None, *,
                         gpt5_reasoning_effort: Optional[str] = None,
                         gpt5_verbosity: Optional[str] = None,
                         gpt5_max_output_tokens: Optional[int] = None) -> List[str]:
        instr = (
            "You are a biomedical entity extraction expert. "
            "Your task is to extract all important biomedical entities (Diseases, Chemicals, Genes) from the input text.\n"
            "Extract the specific names as they appear or in their canonical form.\n\n"
            "Examples:\n"
            "Input: \"TP53 mutations are frequently observed in colorectal cancer.\"\n"
            "Output: [\"TP53\", \"colorectal cancer\"]\n\n"
            "Input: \"Aspirin usage is associated with reduced risk of myocardial infarction.\"\n"
            "Output: [\"Aspirin\", \"myocardial infarction\"]\n\n"
            "Return ONLY a JSON array of strings. No prose."
        )
        if extra_instructions:
            instr = instr + "\n" + extra_instructions

        prompt = f"Input: {text}\nOutput:"
        # GPT-5 계열은 chat.completions의 일부 파라미터가 호환되지 않을 수 있어 responses API 사용을 우선 시도
        text_out = ""
        try:
            if self.model.startswith("gpt-5"):
                # Responses API
                kwargs = {
                    "model": self.model,
                    "input": f"{instr}\n\n{prompt}",
                }
                if gpt5_reasoning_effort:
                    kwargs["reasoning"] = {"effort": gpt5_reasoning_effort}
                if gpt5_verbosity:
                    kwargs["text"] = {"verbosity": gpt5_verbosity}
                if gpt5_max_output_tokens:
                    kwargs["max_output_tokens"] = int(gpt5_max_output_tokens)
                resp = self.client.responses.create(**kwargs)
                # Prefer text output; fallback to combined text
                text_out = getattr(resp, "output_text", None) or ""
                if not text_out and getattr(resp, "output", None):
                    try:
                        # new SDK may provide a list in resp.output
                        text_out = "".join(getattr(x, "content", "") for x in resp.output)
                    except Exception:
                        text_out = ""
            else:
                # 기존 Chat Completions API
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": instr},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=128,
                )
                text_out = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            # 실패 시 빈 텍스트로 이어서 파싱 시도(빈 결과 반환될 수도 있음)
            text_out = text_out or ""
        
        # Try JSON parse first
        entities = self._parse_json_array(text_out)
        return self._dedup(entities)

    @staticmethod
    def _parse_json_array(text: str) -> List[str]:
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass
        # try to locate first [...] block
        start = text.find("[")
        end = text.rfind("]")
        if 0 <= start < end:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, list):
                    return [str(x) for x in data]
            except Exception:
                return []
        return []

    @staticmethod
    def _dedup(seq: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out



"""Answer generation with deterministic numeric guardrail verification (Phase 4 - Task 4.4)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol

from src.application.dto.search_dto import AnswerRequestDTO, AnswerResponseDTO
from src.application.use_cases.hybrid_search import HybridSearchUseCase
from src.domain.entities.query import SearchQuery
from src.domain.entities.response import Citation, SearchResult
from src.domain.entities.table import TableBlock
from src.domain.exceptions import FinDocError, HallucinationError
from src.domain.interfaces.i_validator import INumericValidator

_RETIRED_GEMINI_MODELS = frozenset(
    {
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
    }
)


class LLMClientProtocol(Protocol):
    """Port protocol for LLM answer synthesis from retrieved financial context."""

    def generate_answer(self, query: str, context: str) -> str:
        """Synthesize a factual answer strictly from the provided financial context."""


class GeminiLLMClient:
    """Zero-dependency Google Gemini API client using standard library HTTP transport."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.5-flash-lite",
        timeout_seconds: int = 20,
    ) -> None:
        self._api_key = api_key.strip()
        cleaned_model = model_name.strip() or "gemini-3.5-flash-lite"
        if cleaned_model in _RETIRED_GEMINI_MODELS:
            cleaned_model = "gemini-3.5-flash-lite"
        self._model_name = cleaned_model
        self._timeout = timeout_seconds

    def _call_model(self, model_name: str, full_prompt: str) -> str:
        """Execute HTTP POST request against a specific Gemini model identifier."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={self._api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        candidates = body.get("candidates", [])
        if not candidates:
            raise FinDocError("Gemini API returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(str(p.get("text", "")) for p in parts).strip()

    def generate_answer(self, query: str, context: str) -> str:
        """Call Gemini `generateContent` endpoint with strict zero-hallucination instructions."""
        if not self._api_key:
            raise FinDocError("GEMINI_API_KEY is not configured")

        system_prompt = (
            "Bạn là chuyên gia phân tích Báo cáo Tài chính. "
            "NHIỆM VỤ: Trả lời câu hỏi dựa DUY NHẤT vào NGỮ CẢNH được cung cấp dưới đây. "
            "QUY TẮC BẤT BIẾN: "
            "1. Tuyệt đối KHÔNG suy đoán hoặc bịa đặt bất kỳ con số nào ngoài ngữ cảnh. "
            "2. Giữ nguyên chính xác các con số như trong bảng biểu gốc. "
            "3. Nếu ngữ cảnh không chứa thông tin để trả lời, hãy nêu rõ không tìm thấy dữ liệu."
        )
        full_prompt = f"{system_prompt}\n\n=== NGỮ CẢNH ===\n{context}\n\n=== CÂU HỎI ===\n{query}"

        candidate_models = [self._model_name]
        if "gemini-3.5-flash-lite" not in candidate_models:
            candidate_models.append("gemini-3.5-flash-lite")

        last_exc: Exception | None = None
        for model in candidate_models:
            try:
                return self._call_model(model, full_prompt)
            except (urllib.error.URLError, TimeoutError, FinDocError) as exc:
                last_exc = exc

        raise FinDocError(
            f"Gemini API request failed: {last_exc}",
            details={"model": self._model_name},
        ) from last_exc


class AnswerWithVerifyUseCase:
    """Orchestrates Hybrid Search -> LLM Answer Synthesis -> Deterministic Numeric Guardrail."""

    def __init__(
        self,
        hybrid_search: HybridSearchUseCase,
        validator: INumericValidator,
        llm_client: LLMClientProtocol,
    ) -> None:
        self._hybrid_search = hybrid_search
        self._validator = validator
        self._llm_client = llm_client

    @staticmethod
    def _build_context_and_tables(
        results: list[SearchResult],
    ) -> tuple[str, list[TableBlock], list[Citation]]:
        """Assemble context prompt string, ground-truth TableBlocks, and Citations."""
        context_blocks: list[str] = []
        source_tables: list[TableBlock] = []
        citations: list[Citation] = []

        for idx, res in enumerate(results, start=1):
            chunk = res.chunk
            context_blocks.append(
                f"--- Nguồn #{idx} (Trang {chunk.page_number}) ---\n{chunk.content}"
            )
            if chunk.table_block is not None:
                source_tables.append(chunk.table_block)
            else:
                source_tables.append(
                    TableBlock(
                        page_number=chunk.page_number,
                        title=chunk.section_title,
                        headers=["Content"],
                        rows=[[chunk.content]],
                        raw_markdown=chunk.content,
                    )
                )
            citations.append(res.citation or Citation.from_chunk(chunk))

        return "\n\n".join(context_blocks), source_tables, citations

    def execute(
        self,
        query: SearchQuery,
        strict_guardrail: bool = False,
        extra_source_tables: list[TableBlock] | None = None,
    ) -> AnswerResponseDTO:
        """Retrieve context, synthesize answer via LLM, and verify numbers via NumericGuardrail."""
        results = self._hybrid_search.execute(query)
        context_str, source_tables, citations = self._build_context_and_tables(results)
        if extra_source_tables:
            source_tables.extend(extra_source_tables)

        if not results:
            empty_answer = "Không tìm thấy dữ liệu phù hợp trong Báo cáo Tài chính để trả lời."
            report = self._validator.validate(answer=empty_answer, source_tables=source_tables)
            return AnswerResponseDTO(
                query=query.query_text,
                answer=empty_answer,
                citations=[],
                results=[],
                validation_report=report,
            )

        generated_answer = self._llm_client.generate_answer(
            query=query.query_text,
            context=context_str,
        )
        validation_report = self._validator.validate(
            answer=generated_answer,
            source_tables=source_tables,
        )

        if strict_guardrail and not validation_report.is_valid:
            raise HallucinationError(
                validation_report.message,
                unverified_numbers=validation_report.hallucinated_numbers,
            )

        return AnswerResponseDTO(
            query=query.query_text,
            answer=generated_answer,
            citations=citations,
            results=results,
            validation_report=validation_report,
        )

    def execute_dto(self, request: AnswerRequestDTO) -> AnswerResponseDTO:
        """Execute question answering and numeric verification from an AnswerRequestDTO."""
        return self.execute(
            query=request.to_search_query(),
            strict_guardrail=request.strict_guardrail,
        )

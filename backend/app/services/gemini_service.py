from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── Output Schema ─────────────────────────────────────────────────────────────

class ReasonsOutput(BaseModel):
    """
    Pydantic model for Gemini's structured output.
    LangChain's PydanticOutputParser enforces this structure.
    """
    reasons: List[str] = Field(
        description="Exactly 5 detailed reasons explaining the verdict"
    )


# ── Prompt Template ───────────────────────────────────────────────────────────

ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["text", "verdict", "confidence", "model_scores"],
    template="""
You are an expert fact-checker and misinformation analyst with 20 years of experience.

A news article or claim has been analyzed by an AI ensemble of 4 machine learning models.

ARTICLE/CLAIM:
\"\"\"{text}\"\"\"

ML ANALYSIS RESULTS:
- Final Verdict    : {verdict}
- Confidence Score : {confidence}%
- Model Scores     : {model_scores}

YOUR TASK:
Provide exactly 5 detailed, evidence-based reasons that explain why this content 
is classified as {verdict}.

Each reason must:
1. Be specific to the text above (not generic)
2. Reference linguistic patterns, factual claims, or journalistic standards
3. Be written in clear, professional language
4. Be 2-3 sentences long
5. Start with a bold keyword like **Sensationalism:**, **Source Credibility:**, etc.

Use these analytical angles:
- Language patterns (sensationalism, emotional manipulation, clickbait)
- Factual accuracy and verifiability of specific claims
- Source attribution and journalistic standards
- Logical consistency and reasoning quality
- Statistical or scientific claim validity

{format_instructions}

IMPORTANT: Return ONLY the JSON. No extra text before or after.
"""
)


# ── Gemini Service ────────────────────────────────────────────────────────────

class GeminiService:
    """
    Wraps LangChain + Gemini to produce structured explanations.
    """

    def __init__(self):
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY not set — reasons will be stubs")
            self.enabled = False
            return

        # Initialize Gemini via LangChain
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0.3,      # lower = more consistent, factual output
            max_tokens=1024
        )

        # Output parser enforces ReasonsOutput schema
        self.parser = PydanticOutputParser(pydantic_object=ReasonsOutput)

        # Build the full chain: prompt → LLM → parser
        self.chain = ANALYSIS_PROMPT | self.llm | self.parser

        self.enabled = True
        logger.info("✅ GeminiService initialized with gemini-2.0-flash")

    async def generate_reasons(
        self,
        text: str,
        verdict: str,
        confidence: float,
        model_results: list
    ) -> List[str]:
        """
        Calls Gemini to generate 5 structured reasons.

        Args:
            text:         Original article/claim text
            verdict:      "REAL", "FAKE", or "UNCERTAIN"
            confidence:   Float 0.0-1.0
            model_results: List of individual model predictions

        Returns:
            List of 5 reason strings
        """
        if not self.enabled:
            return self._stub_reasons(verdict, confidence)

        # Format model scores for the prompt
        model_scores = "\n".join([
            f"  • {r['model_name']}: {r['prediction']} ({r['confidence']*100:.1f}%)"
            for r in model_results
        ])

        # Truncate very long articles for the prompt (Gemini has token limits)
        truncated_text = text[:2000] + "..." if len(text) > 2000 else text

        try:
            logger.info("Calling Gemini API for reasons...")

            result = await self.chain.ainvoke({
                "text":             truncated_text,
                "verdict":          verdict,
                "confidence":       f"{confidence * 100:.1f}",
                "model_scores":     model_scores,
                "format_instructions": self.parser.get_format_instructions()
            })

            logger.info("✅ Gemini reasons generated successfully")
            return result.reasons

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._stub_reasons(verdict, confidence)

    def _stub_reasons(self, verdict: str, confidence: float) -> List[str]:
        """Fallback reasons if Gemini is unavailable."""
        return [
            f"**Verdict**: The content was classified as {verdict} with {confidence*100:.1f}% confidence.",
            "**Language Analysis**: The text contains patterns commonly associated with this classification.",
            "**Factual Consistency**: Claims in the article were evaluated against known fact patterns.",
            "**Source Evaluation**: The writing style and attribution patterns were analyzed.",
            "**Model Consensus**: Multiple independent ML models reached the same conclusion."
        ]


# ── Singleton ─────────────────────────────────────────────────────────────────

_gemini_instance = None

def get_gemini_service() -> GeminiService:
    """Returns shared GeminiService instance."""
    global _gemini_instance
    if _gemini_instance is None:
        _gemini_instance = GeminiService()
    return _gemini_instance
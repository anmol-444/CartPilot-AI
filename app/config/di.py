"""Dependency injection — module-level singletons wired together.

One instance of each component, shared across all requests.
Tests import these directly and call order_repo.reset() between cases.
"""

import os

from app.agent.graph import AgentGraph
from app.agent.skills import AgentSkills
from app.llm.client import LlmClient
from app.llm.guardrail import Guardrail
from app.llm.response_generator import OrderDraftGenerator, QaResponseGenerator
from app.llm.skill_router import SkillRouter
from app.order.repository import InMemoryOrderRepository
from app.order.service import OrderService
from app.product.repository import ChromaProductRepository
from app.product.service import ProductService

# Repositories (infrastructure)
# Product search is backed by ChromaDB (semantic retrieval). The collection is
# populated offline by the indexer (`make index`); the client connects lazily.
product_repo = ChromaProductRepository(
    host=os.environ.get("CHROMA_HOST", "localhost"),
    port=int(os.environ.get("CHROMA_PORT", "8000")),
    collection_name=os.environ.get("CHROMA_COLLECTION", "products"),
    embedding_model=os.environ.get("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
    top_k=int(os.environ.get("RAG_TOP_K", "3")),
)
order_repo = InMemoryOrderRepository()

# Domain services
product_service = ProductService(product_repo)
order_service = OrderService(order_repo)

# LLM client — one Gemini chat model shared across all LLM components
llm_client = LlmClient(
    api_key=os.environ["GOOGLE_API_KEY"],
    model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
)

# LLM components — each receives the same Gemini chat model instance
skill_router = SkillRouter(llm_client.chat_openai)
qa_generator = QaResponseGenerator(llm_client.chat_openai)
order_generator = OrderDraftGenerator(llm_client.chat_openai)
guardrail = Guardrail(llm_client.chat_openai)

# Agent skills — receives everything it needs via constructor
skills = AgentSkills(
    skill_router=skill_router,
    qa_generator=qa_generator,
    order_generator=order_generator,
    product_service=product_service,
    order_service=order_service,
)

# Agent graph builder — wires skills into the LangGraph graph
agent_graph_builder = AgentGraph(skills=skills)

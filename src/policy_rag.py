"""
Policy RAG - Retrieval Augmented Generation for Policy Questions

Uses ChromaDB to retrieve relevant policy sections and generate accurate answers.

Usage:
    from src.policy_rag import PolicyRAG
    rag = PolicyRAG()
    answer = rag.answer_policy_question("What is your refund policy?")
"""

import os
import json
from typing import List, Dict, Optional, Tuple


class PolicyRAG:
    """RAG system for answering policy questions using ChromaDB"""
    
    def __init__(
        self,
        chroma_db_path: str = "data/embeddings/chroma_db",
        policies_json_path: str = "data/policies/policies.json"
    ):
        self.chroma_db_path = chroma_db_path
        self.policies_json_path = policies_json_path
        self.chromadb_available = False
        self.policies_collection = None
        
        self._init_chromadb()
        self._load_json_fallback()
    
    def _init_chromadb(self):
        """Initialize ChromaDB connection"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            if not os.path.exists(self.chroma_db_path):
                print(f"ChromaDB path not found: {self.chroma_db_path}")
                return
            
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get policies collection
            try:
                self.policies_collection = self.chroma_client.get_collection("policies")
                self.chromadb_available = True
                print(f"✅ PolicyRAG: ChromaDB connected ({self.policies_collection.count()} documents)")
            except Exception as e:
                print(f"⚠️ Policies collection not found: {e}")
                self.chromadb_available = False
                
        except ImportError:
            print("⚠️ ChromaDB not installed")
        except Exception as e:
            print(f"⚠️ ChromaDB error: {e}")
    
    def _load_json_fallback(self):
        """Load policies from JSON as fallback"""
        self.policies_json = []
        if os.path.exists(self.policies_json_path):
            try:
                with open(self.policies_json_path, 'r', encoding='utf-8') as f:
                    self.policies_json = json.load(f)
                print(f"✅ PolicyRAG: JSON fallback loaded ({len(self.policies_json)} policies)")
            except Exception as e:
                print(f"⚠️ Failed to load policies JSON: {e}")
    
    def retrieve_relevant_sections(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant policy sections from ChromaDB using semantic search.
        
        Args:
            query: User's question
            n_results: Number of results to return
            
        Returns:
            List of relevant policy sections with metadata
        """
        results = []
        
        # Try ChromaDB first (semantic search)
        if self.chromadb_available and self.policies_collection:
            try:
                search_results = self.policies_collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                if search_results and search_results['documents']:
                    for i, doc in enumerate(search_results['documents'][0]):
                        metadata = search_results['metadatas'][0][i] if search_results['metadatas'] else {}
                        distance = search_results['distances'][0][i] if search_results['distances'] else 0
                        
                        results.append({
                            "content": doc,
                            "policy_name": metadata.get('policy_name', 'Unknown'),
                            "section_title": metadata.get('section_title', ''),
                            "type": metadata.get('type', 'policy_section'),
                            "relevance_score": 1 - distance,  # Convert distance to similarity
                            "source": "chromadb"
                        })
                    
                    print(f"📚 Retrieved {len(results)} sections from ChromaDB")
                    return results
                    
            except Exception as e:
                print(f"⚠️ ChromaDB query error: {e}")
        
        # Fallback to JSON keyword search
        return self._keyword_search_json(query, n_results)
    
    def _keyword_search_json(self, query: str, n_results: int = 5) -> List[Dict]:
        """Fallback keyword search in JSON policies"""
        if not self.policies_json:
            return []
        
        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if len(w) >= 3]
        
        # Determine which policy is most relevant
        policy_scores = {}
        
        # Keywords mapping to policies
        policy_keywords = {
            "refund_policy": ["refund", "return", "exchange", "money", "back", "cancel"],
            "shipping_policy": ["ship", "shipping", "deliver", "delivery", "track", "arrive", "courier"],
            "terms_of_service": ["terms", "service", "condition", "agreement", "legal", "privacy"]
        }
        
        for policy_id, kws in policy_keywords.items():
            score = sum(1 for kw in kws if kw in query_lower)
            if score > 0:
                policy_scores[policy_id] = score
        
        # Get most relevant policy
        if policy_scores:
            best_policy_id = max(policy_scores, key=policy_scores.get)
        else:
            best_policy_id = None
        
        results = []
        
        for policy in self.policies_json:
            if best_policy_id and policy.get('policy_id') != best_policy_id:
                continue
            
            # Add full policy
            results.append({
                "content": policy.get('content', '')[:2000],
                "policy_name": policy.get('policy_name', 'Unknown'),
                "section_title": "Full Policy",
                "type": "full_policy",
                "relevance_score": 0.8,
                "source": "json"
            })
            
            # Add relevant sections
            for section in policy.get('sections', []):
                section_content = section.get('content', '')
                section_title = section.get('title', '')
                
                # Check if section is relevant
                section_text = (section_title + ' ' + section_content).lower()
                relevance = sum(1 for kw in keywords if kw in section_text)
                
                if relevance > 0:
                    results.append({
                        "content": f"{section_title}\n\n{section_content}",
                        "policy_name": policy.get('policy_name', 'Unknown'),
                        "section_title": section_title,
                        "type": "policy_section",
                        "relevance_score": min(0.9, 0.5 + relevance * 0.1),
                        "source": "json"
                    })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        print(f"📚 Retrieved {len(results[:n_results])} sections from JSON")
        return results[:n_results]
    
    def format_context_for_llm(self, sections: List[Dict]) -> str:
        """Format retrieved sections as context for LLM"""
        if not sections:
            return "No relevant policy information found."
        
        context_parts = []
        
        for i, section in enumerate(sections[:3]):  # Top 3 most relevant
            policy_name = section.get('policy_name', 'Policy')
            section_title = section.get('section_title', '')
            content = section.get('content', '')
            
            header = f"[{policy_name}"
            if section_title:
                header += f" - {section_title}"
            header += "]"
            
            context_parts.append(f"{header}\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def answer_policy_question(
        self,
        question: str,
        llm_client=None,
        model: str = "gpt-4o-mini"
    ) -> Tuple[str, List[Dict]]:
        """
        Answer a policy question using RAG.
        
        Args:
            question: User's policy question
            llm_client: OpenAI client (optional, will create if not provided)
            model: LLM model to use
            
        Returns:
            (answer, retrieved_sections)
        """
        # Retrieve relevant sections
        sections = self.retrieve_relevant_sections(question, n_results=5)
        
        if not sections:
            return (
                "I couldn't find specific policy information about that. "
                "Please contact us at hello@bynoemie.com for assistance.",
                []
            )
        
        # Format context
        context = self.format_context_for_llm(sections)
        
        # If no LLM client, try to create OpenAI client
        if llm_client is None:
            try:
                import os
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY", "")
                if api_key:
                    llm_client = OpenAI(api_key=api_key)
            except:
                pass
        
        if llm_client is None:
            # Return formatted context without LLM
            top_section = sections[0]
            return (
                "**{}**\n\n{}...".format(top_section['policy_name'], top_section['content'][:1000]),
                sections
            )
        
        # Generate answer using LLM
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful customer service assistant for ByNoemie fashion boutique.
Answer the customer's question based ONLY on the policy information provided.
Be concise, friendly, and professional.
If the information doesn't fully answer the question, suggest contacting hello@bynoemie.com.
Format your response clearly with bullet points if listing multiple items."""
                    },
                    {
                        "role": "user",
                        "content": "Customer Question: {}\n\nRelevant Policy Information:\n{}\n\nPlease answer the customer's question based on this policy information.".format(question, context)
                    }
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            return (answer, sections)
            
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            # Return context without LLM enhancement
            top_section = sections[0]
            return (
                f"**{top_section['policy_name']}**\n\n{top_section['content'][:1000]}",
                sections
            )
    
    def get_policy_summary(self, policy_type: str) -> Optional[str]:
        """Get a summary of a specific policy"""
        policy_map = {
            "refund": "refund_policy",
            "return": "refund_policy",
            "shipping": "shipping_policy",
            "delivery": "shipping_policy",
            "terms": "terms_of_service"
        }
        
        policy_id = policy_map.get(policy_type.lower(), policy_type)
        
        for policy in self.policies_json:
            if policy.get('policy_id') == policy_id:
                return f"**{policy['policy_name']}**\n\n{policy['content'][:1500]}..."
        
        return None


# Convenience function
def create_policy_rag() -> PolicyRAG:
    """Create and return a PolicyRAG instance"""
    return PolicyRAG()

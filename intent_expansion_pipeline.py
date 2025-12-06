import json
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"Error importing ML libraries: {e}")
    print("Please install requirements: pip install sentence-transformers scikit-learn pandas numpy")

try:
    import google.generativeai as genai
except ImportError:
    genai = None

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CLUSTERING_DISTANCE_THRESHOLD = 0.5
MIN_CLUSTER_SIZE = 3
LLM_MODEL_NAME = "gemini-2.5-flash"

class IntentExpansionPipeline:
    def __init__(self, input_file: str, api_key: Optional[str] = None):
        self.input_file = input_file
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.data = self._load_data()
        self.messages = self.data.get("customer_messages", [])
        self.intent_mapper = self.data.get("intent_mapper", [])
        self.use_cases = self._format_use_cases()
        self.embeddings = None
        self.clusters = {}
        
        if self.api_key and genai:
            genai.configure(api_key=self.api_key)
        else:
            print("WARNING: No API key provided or google-generativeai not installed. LLM steps will be mocked/skipped.")

    def _load_data(self) -> Dict:
        with open(self.input_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _format_use_cases(self):
        lines = []
        for primary in self.intent_mapper:
            lines.append(f"\n{primary['primary_intent_name']} ({primary['primary_intent_id']}):")
            for secondary in primary['secondary_intents']:
                lines.append(f"  - {secondary['id']}: {secondary['description']}")
        return '\n'.join(lines)

    def preprocess_and_embed(self):
        print("Loading embedding model...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        texts = []
        for msg in self.messages:
            text = f"Context: {msg.get('history', '')}\nCustomer: {msg.get('current_human_message', '')}"
            texts.append(text)
        
        print(f"Generating embeddings for {len(texts)} messages...")
        self.embeddings = model.encode(texts, show_progress_bar=True)
        self.texts = texts
        return self.embeddings

    def cluster_messages(self):
        print("Clustering messages...")
        
        embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=CLUSTERING_DISTANCE_THRESHOLD,
            metric='euclidean',
            linkage='ward'
        )
        labels = clustering.fit_predict(embeddings_norm)
        
        self.clusters = {}
        for idx, label in enumerate(labels):
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(idx)
            
        print(f"Found {len(self.clusters)} clusters.")
        
        self.clusters = {k: v for k, v in self.clusters.items() if len(v) >= MIN_CLUSTER_SIZE}
        print(f"Retained {len(self.clusters)} clusters with size >= {MIN_CLUSTER_SIZE}.")

    def get_representative_samples(self, n_per_cluster=3) -> List[str]:
        """
        Selects representative messages from each cluster to ensure coverage of all patterns.
        This replaces random sampling with stratified sampling based on clusters.
        """
        samples = []
        print(f"Selecting representative samples (top {n_per_cluster} from each cluster)...")
        
        for cluster_id, indices in self.clusters.items():
            # In a real scenario, we might pick the one closest to centroid.
            # Here we just pick the first few as they are arbitrary in Agglomerative without centroid computation.
            # To improve, we could compute centroid of this cluster and find closest.
            # For speed, we'll take the first few.
            selected_indices = indices[:n_per_cluster]
            for i in selected_indices:
                samples.append(self.messages[i]['current_human_message'])
        
        print(f"Selected {len(samples)} representative messages from {len(self.clusters)} clusters.")
        return samples

    def _call_llm(self, prompt: str) -> Optional[str]:
        if not self.api_key or not genai:
            return None
        
        try:
            model = genai.GenerativeModel(LLM_MODEL_NAME)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"LLM Call failed: {e}")
            return None

    def identify_issues(self, representative_samples: List[str]):
        """
        Analyzes the representative samples to find overloaded/missing intents.
        Adapted from prac.py logic.
        """
        prompt = f"""Analyze customer support data for intent hierarchy issues. Return ONLY valid JSON, no other text.

CURRENT INTENT HIERARCHY:
{self.use_cases}

REPRESENTATIVE SAMPLE MESSAGES (Selected via Clustering):
{json.dumps(representative_samples, indent=2)}

ANALYZE:
1. Which intents are overloaded (too broad)?
2. What missing intents do you see?
3. Which intents have unclear boundaries?

Return exactly this JSON format:
{{
  "overloaded_intents": [
    {{
      "intent": "primary::secondary",
      "reason": "why overloaded",
      "proposed_split": [
        {{"name": "Name", "id": "id", "description": "desc", "examples": ["ex1"]}}
      ],
      "risk_score": 5
    }}
  ],
  "missing_intents": [
    {{
      "parent_primary": "primary_id",
      "name": "Name",
      "id": "id",
      "description": "desc",
      "justification": "why",
      "examples": ["ex1"],
      "confidence": 0.8
    }}
  ],
  "unclear_boundaries": [
    {{
      "intent_pair": ["intent1", "intent2"],
      "issue": "what's unclear",
      "recommendation": "how to fix"
    }}
  ]
}}"""
        
        print("Analyzing intent issues (LLM)...")
        response = self._call_llm(prompt)
        if not response:
            return {"overloaded_intents": [], "missing_intents": [], "unclear_boundaries": []}
        
        try:
            text = response.strip()
            if '```' in text:
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
            text = text.strip()
            return json.loads(text)
        except Exception as e:
            print(f"Parse error in identify_issues: {e}")
            return {"overloaded_intents": [], "missing_intents": [], "unclear_boundaries": []}

    def validate_proposals(self, analysis):
        """
        Validates the proposed changes. Adapted from prac.py.
        """
        proposals = analysis.get('overloaded_intents', []) + [
            {'proposed_split': [m]} for m in analysis.get('missing_intents', [])
        ]
        
        if not proposals:
            return {"validated": []}
        
        prompt = f"""Validate these intent proposals. Return ONLY valid JSON, no other text.

CURRENT HIERARCHY:
{self.use_cases}

PROPOSALS:
{json.dumps(proposals, indent=2)}

Evaluate:
1. Is definition unambiguous?
2. Can small LLM handle it?
3. Does it overlap with existing?
4. Is complexity justified?

Return exactly this JSON format:
{{
  "validated": [
    {{
      "proposal": {{}},
      "approved": true,
      "concerns": ["concern1"],
      "final_recommendation": "reasoning"
    }}
  ]
}}"""
        
        print("Validating proposals (LLM)...")
        response = self._call_llm(prompt)
        if not response:
            return {"validated": []}
        
        try:
            text = response.strip()
            if '```' in text:
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
            text = text.strip()
            return json.loads(text)
        except Exception as e:
            print(f"Parse error in validate_proposals: {e}")
            return {"validated": []}

    def run(self, output_file: str):
        self.preprocess_and_embed()
        self.cluster_messages()
        
       
        samples = self.get_representative_samples(n_per_cluster=3)
        
      
        analysis = self.identify_issues(samples)
        
        print(f"\nIssues found:")
        print(f"  Overloaded intents: {len(analysis.get('overloaded_intents', []))}")
        print(f"  Missing intents: {len(analysis.get('missing_intents', []))}")
        print(f"  Unclear boundaries: {len(analysis.get('unclear_boundaries', []))}")
        
        
        validation = self.validate_proposals(analysis)
        approved = [v for v in validation.get('validated', []) if v.get('approved')]
        
        report = {
            "summary": {
                "total_messages": len(self.messages),
                "clusters_found": len(self.clusters),
                "representative_samples_analyzed": len(samples),
                "issues_found": len(analysis.get('overloaded_intents', [])) + len(analysis.get('missing_intents', [])),
                "approved_recommendations": len(approved)
            },
            "analysis": analysis,
            "validation": validation,
            "approved_recommendations": approved
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"Report generated: {output_file}")
        
        return report

def main():
    input_path = "inputs_for_assignment.json"
    output_path = "intent_expansion_report.json"
    
    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found.")
        return

    pipeline = IntentExpansionPipeline(input_path)
    
    try:
        pipeline.run(output_path)
            
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

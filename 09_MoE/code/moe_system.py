import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from typing import List, Dict, Tuple
import re
import warnings
from datetime import datetime
import json
from collections import defaultdict

warnings.filterwarnings('ignore')

class Expert:
    """
    Domain-specific expert for question answering
    """
    
    def __init__(self, name: str, domain: str, model_name: str = "distilbert-base-cased-distilled-squad"):
        self.name = name
        self.domain = domain
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.qa_pipeline = pipeline(
                "question-answering", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            self.model_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load expert {name}: {e}")
        
        self.performance_score = 1.0
        self.usage_count = 0
        self.domain_accuracy = defaultdict(list)
    
    def predict(self, question: str, context: str) -> Dict:
        """Generate answer for the given question"""
        self.usage_count += 1
        
        try:
            result = self.qa_pipeline(
                question=question, 
                context=context,
                max_answer_len=150,
                handle_impossible_answer=True
            )
            
            if result['score'] < 0.1 or result['answer'].strip() == '':
                result['answer'] = self._generate_fallback_answer(context)
                result['score'] = max(result['score'], 0.3)
            
            return {
                'answer': result['answer'],
                'score': min(result['score'] * self.performance_score, 1.0),
                'expert': self.name,
                'domain': self.domain,
                'timestamp': datetime.now()
            }
        except Exception as e:
            return {
                'answer': self._generate_fallback_answer(context),
                'score': 0.2,
                'expert': self.name,
                'domain': self.domain,
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    def _generate_fallback_answer(self, context: str) -> str:
        """Generate fallback answer when model confidence is low"""
        sentences = [s.strip() for s in context.split('.') if s.strip()]
        return sentences[0] + '.' if sentences else "Based on available information, this requires specialized analysis."
    
    def update_performance(self, feedback_score: float, domain: str = None):
        """Update expert performance based on feedback"""
        alpha = 0.1
        self.performance_score = (1 - alpha) * self.performance_score + alpha * feedback_score
        
        if domain:
            self.domain_accuracy[domain].append(feedback_score)
    
    def get_stats(self) -> Dict:
        """Get expert statistics"""
        return {
            'name': self.name,
            'domain': self.domain,
            'performance_score': self.performance_score,
            'usage_count': self.usage_count,
            'domain_accuracy': dict(self.domain_accuracy)
        }

class Router:
    """
    Intelligent router for expert selection
    """
    
    def __init__(self, experts: List[Expert]):
        self.experts = experts
        self.domain_keywords = self._build_domain_keywords()
        self.routing_history = []
    
    def _build_domain_keywords(self) -> Dict[str, List[str]]:
        """Build domain keyword mapping with Portuguese support"""
        return {
            'medical': [
                'covid', 'symptom', 'medical', 'health', 'disease', 'treatment', 
                'hospital', 'doctor', 'patient', 'medicine', 'virus', 'fever',
                'cough', 'headache', 'pain', 'infection', 'vaccine', 'pandemic',
                'sintoma', 'médico', 'saúde', 'doença', 'tratamento', 'hospital',
                'paciente', 'medicamento', 'vírus', 'febre', 'tosse', 'dor'
            ],
            'legal': [
                'law', 'legal', 'right', 'contract', 'court', 'lawyer', 'constitution',
                'lei', 'jurídico', 'direito', 'contrato', 'tribunal', 'advogado'
            ],
            'technical': [
                'technology', 'programming', 'software', 'computer', 'code', 'python',
                'tecnologia', 'programação', 'software', 'computador', 'código'
            ],
            'scientific': [
                'science', 'research', 'experiment', 'study', 'scientific', 'data',
                'ciência', 'pesquisa', 'experimento', 'estudo', 'científico'
            ],
            'business': [
                'business', 'company', 'market', 'management', 'strategy', 'profit',
                'negócio', 'empresa', 'mercado', 'gestão', 'estratégia'
            ]
        }
    
    def calculate_domain_similarity(self, question: str, domain: str) -> float:
        """Calculate similarity between question and domain"""
        question_lower = question.lower()
        domain_keywords = self.domain_keywords.get(domain, [])
        
        if not domain_keywords:
            return 0.0
        
        matches = sum(1 for keyword in domain_keywords if keyword in question_lower)
        similarity = matches / len(domain_keywords)
        
        # Boost for important terms
        important_terms = {
            'medical': ['covid', 'symptom', 'sintoma', 'doença'],
            'legal': ['law', 'lei', 'direito', 'legal'],
            'technical': ['python', 'programming', 'programação'],
            'scientific': ['science', 'ciência', 'pesquisa'],
            'business': ['business', 'negócio', 'empresa']
        }
        
        if domain in important_terms:
            for term in important_terms[domain]:
                if term in question_lower:
                    similarity += 0.3
                    break
        
        return min(similarity, 1.0)
    
    def route_question(self, question: str) -> List[Tuple[Expert, float]]:
        """Route question to appropriate experts"""
        expert_scores = []
        
        for expert in self.experts:
            similarity = self.calculate_domain_similarity(question, expert.domain)
            final_score = similarity * expert.performance_score
            expert_scores.append((expert, final_score))
        
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        
        self.routing_history.append({
            'question': question,
            'scores': [(exp.name, score) for exp, score in expert_scores[:3]],
            'timestamp': datetime.now()
        })
        
        return expert_scores

class MixtureOfExperts:
    """
    Main Mixture of Experts system
    """
    
    def __init__(self):
        self.experts = self._initialize_experts()
        self.router = Router(self.experts)
        self.history = []
        self.performance_history = []
        self.knowledge_base = self._initialize_knowledge_base()
    
    def _initialize_experts(self) -> List[Expert]:
        """Initialize domain experts"""
        experts_config = [
            {"name": "Medical Expert", "domain": "medical"},
            {"name": "Legal Expert", "domain": "legal"},
            {"name": "Technical Expert", "domain": "technical"},
            {"name": "Scientific Expert", "domain": "scientific"},
            {"name": "Business Expert", "domain": "business"}
        ]
        
        experts = []
        for config in experts_config:
            try:
                expert = Expert(config["name"], config["domain"])
                experts.append(expert)
            except Exception as e:
                print(f"Warning: Failed to initialize {config['name']}: {e}")
                continue
                
        return experts
    
    def _initialize_knowledge_base(self) -> Dict[str, str]:
        """Initialize domain-specific knowledge"""
        return {
            'medical': """
            COVID-19 is an infectious disease caused by the SARS-CoV-2 virus. Common symptoms include 
            fever, cough, fatigue, loss of taste or smell, sore throat, headache, muscle pain, 
            difficulty breathing, and gastrointestinal issues. Severe cases can lead to pneumonia, 
            acute respiratory distress syndrome, and other complications. Prevention measures include 
            vaccination, mask-wearing, social distancing, and hand hygiene.
            """,
            'legal': """
            Legal systems are based on constitutions, statutes, regulations, and judicial precedents.
            Key principles include rule of law, justice, equality before the law, and due process.
            Legal proceedings involve courts, judges, attorneys, and various procedural rules.
            """,
            'technical': """
            Technology involves hardware components like processors, memory, and storage devices,
            and software including operating systems, applications, and programming languages.
            Key concepts include algorithms, data structures, networks, databases, and cybersecurity.
            """,
            'scientific': """
            The scientific method involves observation, hypothesis formation, experimentation, and conclusion.
            Research follows rigorous methodologies and undergoes peer review before publication.
            Major scientific disciplines include physics, chemistry, biology, and earth sciences.
            """,
            'business': """
            Business operations encompass management, marketing, finance, human resources, and strategy.
            Key concepts include profit maximization, market share, competitive advantage, and ROI.
            Business models define how organizations create, deliver, and capture value.
            """
        }
    
    def detect_domain(self, question: str) -> str:
        """Detect the primary domain of the question"""
        domain_scores = {}
        
        for domain in ['medical', 'legal', 'technical', 'scientific', 'business']:
            score = self.router.calculate_domain_similarity(question, domain)
            domain_scores[domain] = score
        
        best_domain, best_score = max(domain_scores.items(), key=lambda x: x[1])
        return best_domain if best_score > 0.1 else 'medical'
    
    def ask_question(self, question: str, top_k: int = 3) -> Dict:
        """Process a question through the MoE system"""
        if not question.strip():
            return {'error': 'Empty question provided'}
        
        # Detect domain and get context
        domain = self.detect_domain(question)
        context = self.knowledge_base.get(domain, self.knowledge_base['medical'])
        
        # Route to experts
        expert_scores = self.router.route_question(question)
        selected_experts = expert_scores[:top_k]
        
        # Get predictions
        predictions = []
        weights = []
        
        for expert, weight in selected_experts:
            prediction = expert.predict(question, context)
            predictions.append(prediction)
            weights.append(weight)
        
        # Aggregate results
        if predictions:
            best_idx = np.argmax([p['score'] for p in predictions])
            best_prediction = predictions[best_idx]
            
            final_answer = {
                'answer': best_prediction['answer'],
                'score': best_prediction['score'],
                'expert': best_prediction['expert'],
                'domain': domain,
                'all_predictions': predictions,
                'expert_weights': weights,
                'timestamp': datetime.now()
            }
        else:
            final_answer = {
                'answer': "No experts could answer this question.",
                'score': 0.0,
                'expert': 'None',
                'domain': domain,
                'timestamp': datetime.now()
            }
        
        # Store history
        self.history.append({
            'question': question,
            'answer': final_answer,
            'timestamp': datetime.now()
        })
        
        return final_answer
    
    def provide_feedback(self, question: str, rating: float, correct_answer: str = None):
        """Provide feedback to improve system"""
        for entry in self.history:
            if entry['question'] == question:
                expert_name = entry['answer']['expert']
                expert = next((e for e in self.experts if e.name == expert_name), None)
                
                if expert:
                    # Calculate similarity with correct answer if provided
                    similarity = 1.0
                    if correct_answer:
                        similarity = self._calculate_answer_similarity(
                            entry['answer']['answer'], correct_answer
                        )
                    
                    feedback_score = rating * similarity
                    expert.update_performance(feedback_score, entry['answer']['domain'])
                
                self.performance_history.append({
                    'question': question,
                    'rating': rating,
                    'timestamp': datetime.now()
                })
                break
    
    def _calculate_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate similarity between two answers"""
        words1 = set(answer1.lower().split())
        words2 = set(answer2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        total_questions = len(self.history)
        avg_confidence = np.mean([entry['answer']['score'] for entry in self.history]) if self.history else 0
        
        expert_stats = {}
        for expert in self.experts:
            expert_stats[expert.name] = expert.get_stats()
        
        return {
            'system_metrics': {
                'total_questions': total_questions,
                'average_confidence': avg_confidence,
                'active_experts': len(self.experts),
                'total_feedback': len(self.performance_history)
            },
            'expert_performance': expert_stats,
            'domain_distribution': self._get_domain_distribution()
        }
    
    def _get_domain_distribution(self) -> Dict:
        """Get distribution of questions by domain"""
        domains = [entry['answer']['domain'] for entry in self.history]
        return {domain: domains.count(domain) for domain in set(domains)}
    
    def get_recent_history(self, limit: int = 10) -> List[Dict]:
        """Get recent question history"""
        return self.history[-limit:] if self.history else []

# Singleton instance for the application
moe_system = MixtureOfExperts()

if __name__ == "__main__":
    # Test the system
    test_questions = [
        "Quais os sintomas do COVID-19?",
        "How does Python handle memory management?",
        "What are the basic principles of business management?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = moe_system.ask_question(question)
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['score']:.3f}")
        print(f"Expert: {result['expert']}")
        print(f"Domain: {result['domain']}")
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from moe_system import moe_system

# Page configuration
st.set_page_config(
    page_title="Mixture of Experts Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .expert-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #00cc96; }
    .confidence-medium { color: #ffa15a; }
    .confidence-low { color: #ef553b; }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitInterface:
    def __init__(self):
        self.moe_system = moe_system
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'questions_asked' not in st.session_state:
            st.session_state.questions_asked = 0
        if 'current_question' not in st.session_state:
            st.session_state.current_question = ""
        if 'last_answer' not in st.session_state:
            st.session_state.last_answer = None
    
    def render_sidebar(self):
        """Render the sidebar with system information"""
        with st.sidebar:
            st.title("🤖 System Controls")
            
            # System statistics
            stats = self.moe_system.get_system_stats()
            
            st.subheader("📊 System Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Questions Processed", stats['system_metrics']['total_questions'])
                st.metric("Active Experts", stats['system_metrics']['active_experts'])
            with col2:
                st.metric("Avg Confidence", f"{stats['system_metrics']['average_confidence']:.3f}")
                st.metric("Feedback Received", stats['system_metrics']['total_feedback'])
            
            st.subheader("🔧 Configuration")
            top_k = st.slider("Number of Experts to Consult", 1, 5, 3)
            auto_detect = st.checkbox("Auto Domain Detection", value=True)
            
            st.subheader("🎯 Expert Performance")
            for expert_name, expert_stats in stats['expert_performance'].items():
                with st.expander(f"{expert_name} ({expert_stats['domain']})"):
                    st.progress(expert_stats['performance_score'])
                    st.write(f"Performance: {expert_stats['performance_score']:.3f}")
                    st.write(f"Usage Count: {expert_stats['usage_count']}")
    
    def render_main_interface(self):
        """Render the main Q&A interface"""
        st.markdown('<div class="main-header">🤖 Mixture of Experts Q&A System</div>', unsafe_allow_html=True)
        
        # Main question input
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input(
                "💭 Ask your question:",
                placeholder="e.g., What are the symptoms of COVID-19?",
                key="question_input"
            )
        with col2:
            ask_button = st.button("Ask Experts", type="primary", use_container_width=True)
        
        if ask_button and question:
            self.process_question(question)
        
        # Display last answer if available
        if st.session_state.last_answer:
            self.display_answer(st.session_state.last_answer)
        
        # Recent history
        self.display_recent_history()
    
    def process_question(self, question: str):
        """Process a question and display results"""
        with st.spinner("🔍 Consulting experts..."):
            # Add slight delay for better UX
            time.sleep(0.5)
            
            result = self.moe_system.ask_question(question)
            
            if 'error' in result:
                st.error(f"Error: {result['error']}")
                return
            
            st.session_state.current_question = question
            st.session_state.last_answer = result
            st.session_state.questions_asked += 1
            
            # Show success message
            st.success("✅ Answer generated successfully!")
    
    def display_answer(self, answer_data: dict):
        """Display the answer with detailed information"""
        st.markdown("---")
        
        # Main answer section
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("🎯 Answer")
            st.info(answer_data['answer'])
        with col2:
            confidence = answer_data['score']
            confidence_color = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.4 else "confidence-low"
            st.markdown(f'<div class="metric-card"><h4>Confidence</h4><p class="{confidence_color}">{confidence:.3f}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h4>Expert</h4><p>{answer_data["expert"]}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h4>Domain</h4><p>{answer_data["domain"]}</p></div>', unsafe_allow_html=True)
        
        # Expert contributions
        if 'all_predictions' in answer_data:
            st.subheader("👥 Expert Contributions")
            for i, prediction in enumerate(answer_data['all_predictions']):
                with st.expander(f"{prediction['expert']} (Score: {prediction['score']:.3f})"):
                    st.write(f"**Answer:** {prediction['answer']}")
                    st.write(f"**Domain:** {prediction['domain']}")
                    st.write(f"**Confidence:** {prediction['score']:.3f}")
        
        # Feedback section
        self.render_feedback_section(answer_data)
    
    def render_feedback_section(self, answer_data: dict):
        """Render the feedback section"""
        st.markdown("---")
        st.subheader("💬 Provide Feedback")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("👍 Helpful", use_container_width=True):
                self.moe_system.provide_feedback(
                    st.session_state.current_question, 
                    rating=1.0
                )
                st.success("Thank you for your feedback!")
        
        with col2:
            if st.button("👎 Not Helpful", use_container_width=True):
                correct_answer = st.text_input("What should the correct answer be?")
                if correct_answer:
                    self.moe_system.provide_feedback(
                        st.session_state.current_question,
                        rating=0.0,
                        correct_answer=correct_answer
                    )
                    st.success("Thank you for the correction!")
        
        with col3:
            if st.button("📊 Rate Answer", use_container_width=True):
                rating = st.slider("Rate this answer (0-1)", 0.0, 1.0, 0.5, 0.1)
                if st.button("Submit Rating"):
                    self.moe_system.provide_feedback(
                        st.session_state.current_question,
                        rating=rating
                    )
                    st.success("Rating submitted!")
    
    def display_recent_history(self):
        """Display recent question history"""
        if not self.moe_system.history:
            return
        
        st.markdown("---")
        st.subheader("📚 Recent Questions")
        
        recent_history = self.moe_system.get_recent_history(5)
        
        for i, entry in enumerate(reversed(recent_history)):
            with st.expander(f"Q: {entry['question']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Answer:** {entry['answer']['answer'][:100]}...")
                with col2:
                    st.write(f"**Expert:** {entry['answer']['expert']}")
                    st.write(f"**Confidence:** {entry['answer']['score']:.3f}")
                    st.write(f"**Time:** {entry['timestamp'].strftime('%H:%M:%S')}")
    
    def render_analytics_dashboard(self):
        """Render analytics dashboard in a separate tab"""
        st.header("📈 System Analytics")
        
        stats = self.moe_system.get_system_stats()
        
        # Expert performance chart
        expert_data = []
        for expert_name, expert_stats in stats['expert_performance'].items():
            expert_data.append({
                'Expert': expert_name,
                'Domain': expert_stats['domain'],
                'Performance': expert_stats['performance_score'],
                'Usage': expert_stats['usage_count']
            })
        
        if expert_data:
            df = pd.DataFrame(expert_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance chart
                fig_perf = px.bar(
                    df, x='Expert', y='Performance', 
                    title='Expert Performance Scores',
                    color='Performance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_perf, use_container_width=True)
            
            with col2:
                # Usage chart
                fig_usage = px.pie(
                    df, values='Usage', names='Expert',
                    title='Expert Usage Distribution'
                )
                st.plotly_chart(fig_usage, use_container_width=True)
            
            # Domain distribution
            domain_data = []
            for domain, count in stats['domain_distribution'].items():
                domain_data.append({'Domain': domain, 'Count': count})
            
            if domain_data:
                df_domain = pd.DataFrame(domain_data)
                fig_domain = px.bar(
                    df_domain, x='Domain', y='Count',
                    title='Question Distribution by Domain',
                    color='Domain'
                )
                st.plotly_chart(fig_domain, use_container_width=True)
    
    def run(self):
        """Main application runner"""
        # Create tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["🎯 Q&A Interface", "📊 Analytics", "ℹ️ About"])
        
        with tab1:
            self.render_sidebar()
            self.render_main_interface()
        
        with tab2:
            self.render_analytics_dashboard()
        
        with tab3:
            self.render_about_page()

    def render_about_page(self):
        """Render the about page"""
        st.header("ℹ️ About Mixture of Experts System")
        
        st.markdown("""
        ### 🤖 System Overview
        
        This Mixture of Experts (MoE) Question Answering System combines multiple specialized AI models 
        to provide accurate, domain-specific answers to your questions.
        
        ### 🎯 How It Works
        
        1. **Domain Detection**: The system analyzes your question to determine the most relevant domain
        2. **Expert Routing**: Questions are routed to the most appropriate domain experts
        3. **Answer Generation**: Each expert generates an answer based on their specialization
        4. **Result Aggregation**: The system selects the best answer based on confidence scores
        
        ### 👥 Available Experts
        
        - **Medical Expert**: Healthcare, symptoms, diseases, medical advice
        - **Legal Expert**: Laws, regulations, legal concepts
        - **Technical Expert**: Technology, programming, software development
        - **Scientific Expert**: Research, experiments, scientific concepts
        - **Business Expert**: Business strategies, management, market analysis
        
        ### 🔧 Technical Architecture
        
        - **Backend**: Python with PyTorch and Hugging Face Transformers
        - **Frontend**: Streamlit for interactive web interface
        - **Models**: DistilBERT-based question answering models
        - **Routing**: Intelligent domain detection and expert selection
        
        ### 📈 Continuous Learning
        
        The system improves over time through user feedback, adapting expert performance scores
        to provide increasingly accurate responses.
        """)

# Main application
if __name__ == "__main__":
    app = StreamlitInterface()
    app.run()
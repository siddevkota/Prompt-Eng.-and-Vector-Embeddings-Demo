"""
Sample data for testing the embeddings demo
Contains various text samples about different topics
"""

# AI and Machine Learning Topics
AI_ML_TEXTS = [
    "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
    "Deep learning uses neural networks with multiple layers to process complex patterns.",
    "Natural language processing helps computers understand and generate human language.",
    "Computer vision enables machines to interpret and understand visual information.",
    "Reinforcement learning trains agents through rewards and penalties.",
    "Transfer learning applies knowledge gained from one task to another related task.",
    "Supervised learning uses labeled data for training predictive models.",
    "Unsupervised learning discovers hidden patterns in unlabeled data.",
    "Neural networks are inspired by the structure of biological neurons in the brain.",
    "Transformers revolutionized natural language processing with attention mechanisms."
]

# Science Topics
SCIENCE_TEXTS = [
    "Quantum mechanics describes the behavior of matter at atomic scales.",
    "DNA carries genetic information in all living organisms.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The theory of relativity revolutionized our understanding of space and time.",
    "Evolution explains the diversity of life through natural selection.",
    "Climate change is affecting ecosystems worldwide.",
    "Black holes are regions of spacetime with extreme gravitational effects.",
    "Antibiotics fight bacterial infections by targeting specific cellular processes.",
    "Renewable energy sources include solar, wind, and hydroelectric power.",
    "The periodic table organizes elements by their atomic structure and properties."
]

# Technology Topics
TECH_TEXTS = [
    "Cloud computing delivers computing services over the internet.",
    "Blockchain technology provides decentralized and transparent record-keeping.",
    "5G networks offer faster speeds and lower latency than previous generations.",
    "Internet of Things connects physical devices to the digital world.",
    "Cybersecurity protects systems and data from digital attacks.",
    "Virtual reality creates immersive digital environments.",
    "Quantum computing uses quantum mechanics for advanced computation.",
    "Edge computing processes data closer to where it's generated.",
    "Artificial intelligence powers smart assistants and recommendation systems.",
    "Big data analytics extracts insights from massive datasets."
]

# Sample Document for RAG
SAMPLE_DOCUMENT = """
Introduction to Artificial Intelligence

Artificial Intelligence (AI) has become one of the most transformative technologies of the 21st century. 
It encompasses a wide range of techniques and approaches that enable machines to perform tasks that 
typically require human intelligence.

Machine Learning Fundamentals

Machine learning is a crucial subset of AI that focuses on developing algorithms that can learn from 
and make predictions or decisions based on data. Unlike traditional programming, where rules are 
explicitly coded, machine learning algorithms identify patterns in data and improve their performance 
over time.

There are three main types of machine learning:
1. Supervised Learning: Uses labeled data to train models
2. Unsupervised Learning: Finds patterns in unlabeled data
3. Reinforcement Learning: Learns through trial and error

Deep Learning and Neural Networks

Deep learning represents a significant advancement in machine learning, utilizing neural networks with 
multiple layers to process complex patterns. These networks are inspired by the structure and function 
of the human brain, consisting of interconnected nodes that process information in layers.

Applications of deep learning include:
- Image recognition and computer vision
- Natural language processing
- Speech recognition
- Autonomous vehicles
- Medical diagnosis

Natural Language Processing

Natural Language Processing (NLP) is a branch of AI that focuses on the interaction between computers 
and human language. Recent breakthroughs in NLP, particularly with transformer models like GPT and BERT, 
have revolutionized how machines understand and generate text.

NLP enables:
- Language translation
- Sentiment analysis
- Text summarization
- Question answering
- Chatbots and virtual assistants

Computer Vision

Computer vision gives machines the ability to interpret and understand visual information from the world. 
This field has seen tremendous progress with deep learning, enabling applications from facial recognition 
to autonomous driving.

Key computer vision tasks include:
- Object detection and recognition
- Image classification
- Semantic segmentation
- Pose estimation
- Visual question answering

Ethical Considerations

As AI systems become more prevalent, ethical considerations are increasingly important. Issues such as 
bias in algorithms, privacy concerns, transparency, and the impact on employment require careful 
consideration and responsible development practices.

The Future of AI

The field of AI continues to evolve rapidly, with ongoing research in areas such as:
- Artificial General Intelligence (AGI)
- Explainable AI
- AI safety and alignment
- Quantum machine learning
- Neuromorphic computing

Conclusion

Artificial Intelligence is reshaping industries and society in profound ways. Understanding its 
fundamentals, capabilities, and limitations is essential for anyone working in technology today.
"""

# Query examples for testing
SAMPLE_QUERIES = [
    "What is machine learning?",
    "How do neural networks work?",
    "What are the applications of NLP?",
    "Explain supervised learning",
    "What is computer vision used for?",
    "What are the ethical concerns with AI?",
    "Tell me about deep learning",
    "How does reinforcement learning work?",
    "What is the difference between AI and machine learning?",
    "What are transformer models?"
]

# Metadata examples
SAMPLE_METADATA = {
    "source": "Week 2 Demo",
    "author": "AI Learning Course",
    "date": "2024",
    "topic": "Embeddings and Vector Search"
}

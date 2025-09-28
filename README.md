# ResearchGPT Assistant

A comprehensive AI-powered research assistant that demonstrates advanced prompting techniques, multi-agent coordination, and document processing capabilities.

## Features

### ✅ **Document Processing System**
- PDF and text file extraction
- Advanced text preprocessing and cleaning
- Intelligent text chunking with overlap
- TF-IDF based similarity search

### ✅ **Advanced Prompting Techniques**
- **Chain-of-Thought (CoT) Reasoning**: Step-by-step problem solving
- **Self-Consistency Prompting**: Multiple attempts with consensus finding
- **ReAct Workflow**: Thought-action-observation cycles

### ✅ **AI Agent Architecture**
- **BaseAgent**: Common interface for all agents
- **SummarizerAgent**: Document summarization
- **QAAgent**: Question answering with context
- **ResearchWorkflowAgent**: Complete research workflows
- **AgentOrchestrator**: Multi-agent coordination

### ✅ **Comprehensive Testing System**
- Performance benchmarking
- Agent evaluation
- System integration testing
- Evaluation report generation

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ResearchGPT
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file with your Mistral API key
   echo "MISTRAL_API_KEY=your_api_key_here" > .env
   ```

## Usage

### Basic Usage

1. **Add your documents**
   ```bash
   # Add .txt or .pdf files to the data/sample_papers/ directory
   cp your_research_papers.pdf data/sample_papers/
   ```

2. **Run the system**
   ```bash
   python main.py
   ```

### Advanced Usage

1. **Run comprehensive tests**
   ```bash
   python test_system.py
   ```

2. **Use individual agents**
   ```python
   from research_agents import AgentOrchestrator
   from config import Config
   from document_processor import DocumentProcessor
   
   config = Config()
   dp = DocumentProcessor(config)
   orchestrator = AgentOrchestrator(config.__dict__, dp)
   
   # Process documents
   dp.process_document("path/to/your/document.pdf")
   dp.build_search_index()
   
   # Ask questions
   result = orchestrator.route_task("qa", {"question": "Your question here"})
   print(result)
   ```

## Configuration

The system uses environment variables for configuration. Create a `.env` file:

```env
# Required
MISTRAL_API_KEY=your_mistral_api_key

# Optional
MISTRAL_MODEL=mistral-small-latest
TEMPERATURE=0.1
DATA_DIR=data
ARTIFACTS_DIR=artifacts
RESULTS_PATH=artifacts/results.json
TEST_QUERY=What problem does HeartSenseAI solve?
```

## System Architecture

```
ResearchGPT Assistant
├── Document Processing
│   ├── PDF/Text Extraction
│   ├── Text Preprocessing
│   ├── Intelligent Chunking
│   └── TF-IDF Search Index
├── AI Agents
│   ├── SummarizerAgent
│   ├── QAAgent
│   ├── ResearchWorkflowAgent
│   └── AgentOrchestrator
├── Advanced Prompting
│   ├── Chain-of-Thought
│   ├── Self-Consistency
│   └── ReAct Workflow
└── Testing & Evaluation
    ├── Performance Benchmarking
    ├── Agent Evaluation
    └── System Integration Tests
```

## File Structure

```
ResearchGPT/
├── main.py                 # Main system demonstration
├── test_system.py          # Comprehensive testing suite
├── config.py              # Configuration management
├── document_processor.py   # Document processing system
├── research_assistant.py   # Core research assistant
├── research_agents.py      # AI agent implementations
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (create this)
├── ENV_SETUP.md           # Environment setup guide
├── data/
│   └── sample_papers/     # Add your documents here
└── artifacts/
    ├── index.json         # Generated search index
    └── results.json       # Generated results
```

## API Usage Examples

### Document Processing
```python
from document_processor import DocumentProcessor
from config import Config

config = Config()
dp = DocumentProcessor(config)

# Process a document
doc_id = dp.process_document("path/to/document.pdf")

# Build search index
dp.build_search_index()

# Find similar chunks
chunks = dp.find_similar_chunks("your query", top_k=5)
```

### Multi-Agent Coordination
```python
from research_agents import AgentOrchestrator

# Initialize orchestrator
orchestrator = AgentOrchestrator(config_dict, document_processor)

# Route tasks to specific agents
summary = orchestrator.route_task("summarizer", {"content": "Your text here"})
qa_result = orchestrator.route_task("qa", {"question": "Your question"})

# Coordinate multi-agent workflow
research_result = orchestrator.coordinate_multi_agent_workflow("research topic")
```

## Testing

The system includes comprehensive testing capabilities:

```bash
# Run full test suite
python test_system.py

# Individual component tests
python -c "
from test_system import ResearchGPTTester
tester = ResearchGPTTester()
tester.test_document_processing()
tester.test_agent_performance()
"
```

## Requirements

- Python 3.8+
- Mistral API key
- Required packages (see requirements.txt):
  - mistralai
  - PyPDF2
  - pandas
  - numpy
  - scikit-learn
  - python-dotenv
  - nltk

## Troubleshooting

### Common Issues

1. **MISTRAL_API_KEY error**
   - Ensure your API key is set in the `.env` file
   - Verify the key is valid and has sufficient credits

2. **No documents found**
   - Add `.txt` or `.pdf` files to `data/sample_papers/` directory
   - Ensure files are readable and not corrupted

3. **Import errors**
   - Activate the virtual environment: `source venv/bin/activate`
   - Install dependencies: `pip install -r requirements.txt`

### Performance Tips

- Use smaller chunk sizes for faster processing
- Limit the number of documents for testing
- Adjust temperature settings for different response styles

## License

This project is for educational and research purposes.

## Contributing

This is a demonstration project showcasing AI/ML concepts including:
- Document processing and information retrieval
- Advanced prompting strategies
- Multi-agent systems
- Machine learning integration
# Environment Setup Guide

## .env File Configuration

Your `.env` file has been created at `/Users/home/Desktop/research_GPT_assistant/ResearchGPT/.env`

### Current Configuration:
```env
# Mistral API Configuration
MISTRAL_API_KEY=93jNnMf84QmFjMoZTZ570LraSoNM44sk

# Optional: Other configuration settings
MISTRAL_MODEL=mistral-small-latest
TEMPERATURE=0.1
DATA_DIR=data
ARTIFACTS_DIR=artifacts
RESULTS_PATH=artifacts/results.json
TEST_QUERY=What problem does HeartSenseAI solve?
```

## How to Update Your API Key

1. **Edit the .env file:**
   ```bash
   nano .env
   # or
   vim .env
   # or open in your preferred editor
   ```

2. **Replace the API key:**
   ```env
   MISTRAL_API_KEY=your_new_api_key_here
   ```

3. **Save the file**

## Security Best Practices

- ✅ **DO**: Keep your `.env` file in your project directory
- ✅ **DO**: Add `.env` to your `.gitignore` file to avoid committing secrets
- ❌ **DON'T**: Share your API key publicly
- ❌ **DON'T**: Commit `.env` files to version control

## Running the Application

With the `.env` file set up, you can now run the application simply with:

```bash
cd /Users/home/Desktop/research_GPT_assistant/ResearchGPT
source venv/bin/activate
python main.py
```

No need to manually export environment variables anymore!

## Troubleshooting

If you get "MISTRAL_API_KEY is missing" error:
1. Check that the `.env` file exists in the ResearchGPT directory
2. Verify the API key is correctly formatted in the `.env` file
3. Make sure there are no extra spaces around the `=` sign
4. Restart your terminal session

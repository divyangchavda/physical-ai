"""Quick test script to verify Gemini API key before running full pipeline."""
import os
import sys

def test_gemini_api(api_key: str) -> bool:
    """Test if Gemini API key is valid and working."""
    print("Testing Gemini API key...")
    
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("❌ google-genai package not installed")
        print("   Run: pip install google-genai")
        return False
    
    try:
        # Initialize client
        client = genai.Client(api_key=api_key)
        
        # Test with a simple text prompt (no video upload needed)
        print("Sending test request to gemini-3.1-flash-lite...")
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents="Say 'API test successful' in JSON format with key 'status'",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        
        result = response.text.strip()
        print(f"✅ API Response: {result}")
        print("✅ Gemini API key is valid and working!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


def save_to_env(api_key: str) -> None:
    """Save API key to .env file."""
    env_path = ".env"
    
    # Read existing .env if it exists
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = [line for line in f.readlines() if not line.startswith("GEMINI_API_KEY=")]
    
    # Add new key
    lines.append(f"GEMINI_API_KEY={api_key}\n")
    
    with open(env_path, "w") as f:
        f.writelines(lines)
    
    print(f"✅ API key saved to {env_path}")


if __name__ == "__main__":
    # Get API key from command line or prompt
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = input("Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        sys.exit(1)
    
    # Test the key
    if test_gemini_api(api_key):
        save = input("\nSave API key to .env file? (y/n): ").strip().lower()
        if save == 'y':
            save_to_env(api_key)
            print("\n✅ Setup complete! You can now run the pipeline.")
        else:
            print("\n⚠️  API key not saved. Set it manually:")
            print(f"   set GEMINI_API_KEY={api_key}")
    else:
        print("\n❌ API key validation failed. Please check your key.")
        sys.exit(1)

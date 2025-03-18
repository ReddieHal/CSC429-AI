import json
import argparse
import os
import re

def extract_probability(response_text):
    """
    Extract probability from model response text based on YES/NO responses.
    """
    # Return None for error responses
    if response_text == "ERROR" or not response_text:
        return None
        
    # First check for explicit YES/NO options from the prompt
    yes_pattern = r'\(1\)\s*YES:\s*A security vulnerability detected'
    no_pattern = r'\(2\)\s*NO:\s*No security vulnerability'
    
    if re.search(yes_pattern, response_text, re.IGNORECASE):
        return 0.9  # High probability for explicit YES
    elif re.search(no_pattern, response_text, re.IGNORECASE):
        return 0.1  # Low probability for explicit NO
    
    # If no explicit options found, fall back to existing patterns
    probability_pattern = r'probability(?:\s+is)?(?:\s+of)?(?:\s+being)?(?:\s+vulnerable)?(?:\s*:)?\s*(\d+(?:\.\d+)?)%?'
    confidence_pattern = r'confidence(?:\s+is)?(?:\s*:)?\s*(\d+(?:\.\d+)?)%?'
    percentage_pattern = r'(\d+(?:\.\d+)?)%\s+(?:chance|probability|confidence|likelihood)'
    decimal_pattern = r'probability(?:\s+is)?(?:\s*:)?\s*(\d+\.\d+)'
    
    # Try to find patterns in the response
    for pattern in [probability_pattern, confidence_pattern, percentage_pattern, decimal_pattern]:
        match = re.search(pattern, response_text.lower())
        if match:
            probability = float(match.group(1))
            # If it's a percentage, convert to decimal
            if pattern != decimal_pattern and probability > 1:
                probability /= 100
            return probability
    
    # Simpler text analysis if no patterns matched
    text_lower = response_text.lower()
    if "yes" in text_lower or any(word in text_lower for word in ['vulnerable', 'security risk', 'vulnerability']):
        return 0.8  # High probability based on YES-like content
    elif "no" in text_lower or "not vulnerable" in text_lower or "secure" in text_lower:
        return 0.2  # Low probability based on NO-like content
    else:
        # Default case
        return 0.5  # Uncertain

def process_file(input_file, output_file):
    """
    Process the OpenRouter output file and convert it to the format expected by the vulnerability score script.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    predictions = []
    skipped = 0
    for line in lines:
        data = json.loads(line)
        
        # Extract key information
        sample_key = data.get('sample_key', '')
        # Assuming the key has some numerical ID we can use
        if '_' in sample_key:
            idx = sample_key.split('_')[-1]  # Get the last part of the key
            try:
                idx = int(idx)
            except ValueError:
                # If it's not a number, we'll use a hash of the key
                idx = hash(sample_key) % 100000
        else:
            idx = hash(sample_key) % 100000
        
        response = data.get('response', '')
        
        # Skip entries with error responses
        if response == "ERROR":
            skipped += 1
            continue
            
        # Extract a probability from the response
        probability = extract_probability(response)
        
        # Skip if we couldn't extract a probability
        if probability is None:
            skipped += 1
            continue
        
        # Determine the label based on probability threshold
        label = 1 if probability >= 0.5 else 0
        
        predictions.append((idx, label, probability))
    
    # Sort predictions by idx
    predictions.sort()
    
    # Write predictions to output file
    with open(output_file, 'w') as f:
        for idx, label, prob in predictions:
            f.write(f"{idx}\t{label}\t{prob}\n")
    
    print(f"Processed {len(predictions)} predictions and saved to {output_file}")
    if skipped > 0:
        print(f"Skipped {skipped} entries due to errors or invalid responses")

def main():
    parser = argparse.ArgumentParser(description="Convert OpenRouter output to vulnerability detection score format")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the OpenRouter output file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the formatted predictions")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    process_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()

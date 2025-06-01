#!/usr/bin/env python3
"""
elarabench: Slop Benchmark Tool

Usage:
  python elarabench.py \
    --server_url https://api.openai.com/v1 \
    --api_key YOUR_API_KEY \
    --model text-davinci-003 \
    --temperature 0.7 \
    --top_k 0 \
    --top_p 1.0 \
    --frequency_penalty 0.0 \
    --presence_penalty 0.0 \
    --prompts_yaml elarabench_prompts.yaml \
    --max_tokens 4096 \
    --repeat 1 \
    --output_file output.txt \
    --number_of_words 25
    
This script:
 1. Loads a list of short prompts from a YAML file
 2. Calls your OpenAI-compatible endpoint for each prompt
 3. Concatenates all outputs, normalizes and tokenizes the text
 4. Filters out stop words (very common articles/pronouns/prepositions/conjunctions)
 5. Computes and prints the top-50 unigrams, bigrams, and trigrams by frequency
"""
import argparse
import os
import yaml
import requests
import re
import json
import time
from collections import Counter
from itertools import islice

current_dir = os.path.dirname(os.path.abspath(__file__))

# --- Helpers ---
def get_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

instruct_template = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are an AI creative writing assistant<|eot_id|>\n"
    "<|start_header_id|>user<|end_header_id|>\n"
    "$user$<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="elarabench: Slop Benchmark Tool")
    parser.add_argument('--server_url', default="http://localhost:8888/v1",
                        help='Base URL of OpenAI-compatible endpoint, e.g. https://api.openai.com/v1')
    parser.add_argument('--api_key', default=os.getenv('OPENAI_API_KEY'),
                        help='Bearer token for API access (or set OPENAI_API_KEY)')
    parser.add_argument('--model', default='text-davinci-003',
                        help='Model name to use for completion')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=0,
                        help='Top‑k sampling (0 means no top_k)')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--prompts_yaml', default=f"{current_dir}/elarabench_prompts.yaml",
                        help='Path to YAML file containing a list of short prompts')
    parser.add_argument('--max_tokens', type=int, default=300,
                        help='Maximum tokens to generate per prompt')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of times to repeat each prompt (default: 1)')
    parser.add_argument('--repeat_until_word_count', type=int, default=0,
                        help='Repeat prompts until this many words are generated (default: 0 means no limit)')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Number of prompts to process in a single batch (default: 5)')
    parser.add_argument('--output_file', type=str, default='output.txt',
                        help='File to write the combined output (default: output.txt)')
    parser.add_argument('--number_of_words', type=int, default=25,
                        help='Number of top words/phrases to display (default: 25)')
    args = parser.parse_args()

    if not args.api_key:
        args.api_key = ''

    # Load prompts from YAML
    with open(args.prompts_yaml, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    if not isinstance(prompts, list):
        raise ValueError('YAML file must contain a list of prompts')

    headers = {'Authorization': f'Bearer {args.api_key}'}
    combined_text = ''
    
    # Query the API for each prompt
    if args.batch_size > 1:
        # Process in batches
        # Make each prompt in prompts occur repeat times
        prompts = [p for p in prompts for _ in range(args.repeat)]
        print(f"Processing {len(prompts)} prompts in batches of {args.batch_size}...")
        for i in range(0, len(prompts), args.batch_size):
            # Get the current batch of prompts            
            batch = prompts[i:i+args.batch_size]
            print(f"\n**** Processing batch {i//args.batch_size + 1} of {len(prompts) // args.batch_size}: {batch}")
            # Prepare batch payload
            batch_payload = {
                'model': args.model,
                'prompt': [instruct_template.replace('$user$', p) if isinstance(p, str) else p for p in batch],
                'temperature': args.temperature,
                'max_tokens': args.max_tokens,
                'top_p': args.top_p,
                'frequency_penalty': args.frequency_penalty,
                'presence_penalty': args.presence_penalty,
            }
            if args.top_k > 0:
                batch_payload['top_k'] = args.top_k

            # Send batch request(s)
            batch_start_time = time.time()
            resp = requests.post(
                f"{args.server_url}/completions",
                headers=headers,
                json=batch_payload
            )
            resp.raise_for_status()
            data = resp.json()
            # Iterate choices: one choice per prompt in batch
            for idx, choice in enumerate(data):
                text = choice.get('content', '')
                print(f"Batch item {idx+1} output: {text}")
                combined_text += text + ' '
            batch_end_time = time.time()
            print(f"Batch processed in {batch_end_time - batch_start_time:.2f} seconds")

            # Optional: repeat until word count
            if args.repeat_until_word_count > 0:
                count = len(re.findall(r'\b\w+\b', combined_text))
                while count < args.repeat_until_word_count:
                    resp = requests.post(
                        f"{args.server_url}/completions",
                        headers=headers,
                        json=batch_payload
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for choice in data.get('choices', []):
                        text = choice.get('text', '')
                        print(f"Additional output: {text}")
                        combined_text += text + ' '
                    count = len(re.findall(r'\b\w+\b', combined_text))
                    print(f"Current word count: {count}")
                    
    else:
        for prompt in prompts:
            print(f"\n**** Processing prompt: {prompt}")
            
            # Prepare the prompt
            if isinstance(prompt, str):
                prompt = instruct_template.replace('$user$', prompt)
                
            payload = {
                'model': args.model,
                'prompt': prompt,
                'temperature': args.temperature,
                'max_tokens': args.max_tokens,
                'top_p': args.top_p,
                'frequency_penalty': args.frequency_penalty,
                'presence_penalty': args.presence_penalty,
                'stream': True
            }
            if args.top_k > 0:
                payload['top_k'] = args.top_k
            
            def stream_request():
                combined_text = ''
                resp = requests.post(
                    f"{args.server_url}/completions",
                    headers=headers,
                    json=payload,
                    stream=True
                )
                resp.raise_for_status()
                print('Response: ', end='', flush=True)
                # Process streaming chunks
                for line in resp.iter_lines(decode_unicode=True):
                    #print(line)
                    if not line:
                        continue
                    line = line.strip()
                    if line == 'data: [DONE]':
                        break
                    if line.startswith('data: '):
                        chunk = json.loads(line[len('data: '):])
                        token = chunk['content']
                        print(token, end='', flush=True)
                        combined_text += token
                print()  # newline after each prompt stream
                return combined_text
                
            if(args.repeat_until_word_count > 0):
                count = 0  
                while count < args.repeat_until_word_count:
                    text = stream_request()
                    if(combined_text != ''):
                        combined_text += ' '
                    combined_text += text
                    # Count words in the current combined text
                    count = len(re.findall(r'\b\w+\b', combined_text))
                    print(f'Current word count: {count}')
            else:
                for _ in range(args.repeat):
                    text = stream_request()
                    if(combined_text != ''):
                        combined_text += ' '
                    combined_text += text           

    # Normalize text
    text = combined_text.lower()
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    words = text.split(' ')
    
    # Write combined output to file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)
    print(f'Combined output written to {args.output_file}')

    # Define stop words (articles, pronouns, prepositions, conjunctions)
    stop_words = {
        'a', 'an', 'the',
        'i', 'me', 'you', 'he', 'him', 'she', 'her', 'it', 'we', 'us', 'they', 'them',
        'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
        'this', 'that', 'these', 'those', 'who', 'whom', 'which', 'what', 'where', 'when', 'why',
        'how', 'whose',
        'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'about', 'as', 'into', 'like',
        'through', 'after', 'over', 'between', 'out', 'against', 'during', 'without', 'before',
        'under', 'around', 'among',
        'and', 'but', 'or', 'nor', 'for', 'so', 'yet', 'because', 'although', 'if', 'when',
        'while', 'whereas', 'once', 'until', 'than', 'though', 'lest',
        'this', 'that', 'these', 'those',
        'of', 'had', 'has', 'have', 'do', 'does', 'did', 'doing', 'was', 'were', 'be', 'being', 'been', 'is', 'are', 'am', 'not', 'no', 'yes', 'up', 'down', 'here', 'there',
    }
    words = [t for t in words if t and t not in stop_words]
    num_words = len(words)
    print(f'Total words after filtering stop words: {num_words}')

    # Compute frequencies
    unigram_counts = Counter(words)
    bigram_counts = Counter(get_ngrams(words, 2))
    trigram_counts = Counter(get_ngrams(words, 3))
    
    # Remove *grams that only occur once, as those aren't slop
    unigram_counts = Counter({word: freq for word, freq in unigram_counts.items() if freq > 1})
    bigram_counts = Counter({phrase: freq for phrase, freq in bigram_counts.items() if freq > 1})
    trigram_counts = Counter({phrase: freq for phrase, freq in trigram_counts.items() if freq > 1})

    # Print top 50 of each
    total_uncertainty = 0
    total_frequency = 0
    print(f'\nTop {args.number_of_words} Unigrams:')
    for word, freq in unigram_counts.most_common(args.number_of_words):
        uncertainty = (freq / num_words) * (1 - freq / num_words) / num_words
        print(f'{word}: {freq}/{num_words} ({freq/num_words:.4%} ± {uncertainty:.4%})')
        total_uncertainty += uncertainty
        total_frequency += freq / num_words

    print(f'\nTop {args.number_of_words} Bigrams:')
    for phrase, freq in bigram_counts.most_common(args.number_of_words):
        uncertainty = (freq / num_words) * (1 - freq / num_words) / num_words
        print(f'{phrase}: {freq}/{num_words} ({freq/num_words:.4%} ± {uncertainty:.4%})')
        total_uncertainty += uncertainty
        total_frequency += freq / num_words

    print(f'\nTop {args.number_of_words} Trigrams:')
    for phrase, freq in trigram_counts.most_common(args.number_of_words):
        uncertainty = (freq / num_words) * (1 - freq / num_words) / num_words
        print(f'{phrase}: {freq}/{num_words} ({freq/num_words:.4%} ± {uncertainty:.4%})')
        total_uncertainty += uncertainty
        total_frequency += freq / num_words

    print(f'\nTotal uncertainty: {total_uncertainty:.4%}')
    print(f'Total frequency: {total_frequency:.4%}')


if __name__ == '__main__':
    main()

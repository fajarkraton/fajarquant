#!/usr/bin/env python3
"""
FajarQuant — KV Cache Extraction from Gemma 4 E2B

Extracts real key-value cache tensors from Gemma 4 E2B for quantization
evaluation. Saves per-layer, per-head K/V matrices as .npy files.

Usage:
    python scripts/extract_kv_cache.py \
        --model google/gemma-4-E2B \
        --num-prompts 100 \
        --max-length 512 \
        --output data/kv_cache/

Output structure:
    data/kv_cache/
        metadata.json          # model info, dimensions, prompt count
        prompt_000/
            layer_00_keys.npy      # shape: (num_kv_heads, seq_len, d_head)
            layer_00_values.npy
            ...
        prompt_001/
            ...
        stats.json             # eigenvalue stats, distribution info
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Representative prompts for KV cache extraction (diverse domains)
PROMPTS = [
    "The theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy. Einstein published this groundbreaking work in",
    "In computer science, a hash table is a data structure that implements an associative array, a structure that can map keys to values. Hash tables use a hash function to compute",
    "The mitochondria are often referred to as the powerhouse of the cell. They generate most of the cell's supply of adenosine triphosphate, which is used as a source of",
    "Machine learning models can be broadly categorized into supervised learning, unsupervised learning, and reinforcement learning. In supervised learning, the model is trained on",
    "The French Revolution began in 1789 and had a lasting impact on political systems worldwide. The storming of the Bastille on July 14 marked a turning point when",
    "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations. Unlike classical bits, quantum bits or qubits can exist in",
    "The human genome contains approximately 3 billion base pairs of DNA organized into 23 pairs of chromosomes. The Human Genome Project, completed in 2003, was a landmark",
    "Neural networks are composed of layers of interconnected nodes or neurons. Each connection has a weight that is adjusted during training through backpropagation, which computes",
    "The Amazon rainforest produces approximately 20% of the world's oxygen and is home to millions of species. Deforestation threatens this vital ecosystem, with an estimated",
    "Cryptographic hash functions like SHA-256 are designed to be one-way functions that map arbitrary input to a fixed-size output. They are fundamental to blockchain technology because",
    "The Standard Model of particle physics describes three of the four known fundamental forces and classifies all known elementary particles. The Higgs boson, discovered at CERN in 2012",
    "Object-oriented programming organizes software design around data and objects rather than functions and logic. The four main principles are encapsulation, abstraction, inheritance, and",
    "Climate change is driven primarily by the increase in greenhouse gas concentrations in the atmosphere, particularly carbon dioxide from burning fossil fuels. The Paris Agreement aims to",
    "Transformer architectures have revolutionized natural language processing since the publication of 'Attention Is All You Need' in 2017. The key innovation is the self-attention mechanism that",
    "The TCP/IP protocol suite forms the backbone of internet communication. TCP provides reliable, ordered delivery of data streams between applications, while IP handles addressing and routing of",
    "CRISPR-Cas9 is a genome editing tool that allows scientists to make precise changes to DNA sequences. The system was adapted from a natural defense mechanism found in bacteria that",
    "Distributed systems face fundamental challenges described by the CAP theorem, which states that a system can provide at most two of three guarantees: consistency, availability, and partition",
    "The periodic table organizes chemical elements by atomic number and electron configuration. Elements in the same group share similar chemical properties because they have the same number of",
    "Convolutional neural networks are particularly effective for image recognition tasks. They use convolutional layers that apply learned filters to detect features such as edges, textures, and",
    "The Turing test, proposed by Alan Turing in 1950, suggests that a machine can be considered intelligent if a human cannot distinguish its responses from those of another human during",
    "Photosynthesis converts light energy into chemical energy stored in glucose. In plants, this process occurs primarily in chloroplasts, where chlorophyll absorbs light and drives the reactions of",
    "Database normalization reduces data redundancy and improves data integrity by organizing fields and tables. The first three normal forms address issues with repeating groups, partial dependencies, and",
    "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. The event horizon marks the boundary beyond which escape is impossible. Hawking radiation",
    "Reinforcement learning trains agents to make sequential decisions by maximizing cumulative reward. Q-learning and policy gradient methods are two major approaches. In deep reinforcement learning",
    "The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. Neuroplasticity allows the brain to reorganize itself by forming new neural",
    "Functional programming treats computation as the evaluation of mathematical functions and avoids changing state and mutable data. Key concepts include pure functions, immutability, and higher-order",
    "Plate tectonics explains the movement of Earth's lithospheric plates, which float on the semi-fluid asthenosphere. Divergent boundaries create new crust, while convergent boundaries lead to",
    "Large language models are trained on vast corpora of text data using self-supervised learning objectives like next-token prediction. The scaling laws suggest that model performance improves",
    "The immune system protects the body through innate and adaptive mechanisms. Innate immunity provides immediate, non-specific defense, while adaptive immunity develops targeted responses through T cells and",
    "Graph neural networks extend deep learning to graph-structured data, enabling applications in social networks, molecular chemistry, and recommendation systems. Message passing between nodes allows",
    "The Big Bang theory describes the origin of the universe approximately 13.8 billion years ago from an extremely hot, dense state. Evidence includes the cosmic microwave background radiation and",
    "Microservices architecture structures an application as a collection of loosely coupled services. Each service is independently deployable and scalable, communicating through well-defined APIs or",
    "DNA replication is a semi-conservative process where each strand of the double helix serves as a template for the new strand. Helicase unwinds the DNA, and DNA polymerase synthesizes",
    "Attention mechanisms in neural networks allow models to focus on relevant parts of the input when producing each element of the output. Multi-head attention runs several attention operations in",
    "The water cycle describes the continuous movement of water through evaporation, condensation, precipitation, and collection. Solar energy drives evaporation from oceans and lakes, and",
    "Garbage collection in programming languages automatically reclaims memory that is no longer in use. Common algorithms include reference counting, mark-and-sweep, and generational collection. The",
    "String theory proposes that fundamental particles are one-dimensional strings rather than point particles. Different vibrational modes of these strings correspond to different particles. The theory",
    "Recurrent neural networks process sequential data by maintaining hidden states that capture information from previous time steps. Long short-term memory networks address the vanishing gradient",
    "The theory of evolution by natural selection, proposed by Charles Darwin, explains how species change over time through variation, inheritance, and differential survival. Modern evolutionary",
    "Consensus algorithms like Paxos and Raft enable distributed systems to agree on a single value even when some nodes fail. Byzantine fault tolerance extends this to handle malicious nodes",
    "Photovoltaic cells convert sunlight directly into electricity using semiconductor materials. When photons strike the cell, they knock electrons free from atoms, creating an electric current",
    "Bayesian inference updates the probability of a hypothesis based on new evidence using Bayes' theorem. The posterior probability is proportional to the likelihood times the prior probability",
    "The endocrine system regulates body functions through hormones secreted by glands such as the thyroid, adrenal, and pituitary glands. Feedback loops maintain hormone levels within",
    "Compiler optimization transforms intermediate representations to improve program performance. Common techniques include dead code elimination, loop unrolling, constant folding, and register allocation",
    "Dark matter makes up approximately 27% of the universe's mass-energy content but does not interact with electromagnetic radiation. Its existence is inferred from gravitational effects on visible",
    "Operating systems manage hardware resources and provide services for application programs. The kernel handles process scheduling, memory management, file systems, and device drivers. Modern",
    "Monte Carlo methods use random sampling to obtain numerical results for problems that might be deterministic in principle. Applications range from physics simulations to financial modeling and",
    "The nervous system transmits signals between the brain and the rest of the body through electrical impulses and chemical neurotransmitters. The central nervous system consists of the brain and",
    "Vector databases store and retrieve high-dimensional vectors efficiently using approximate nearest neighbor search algorithms. They are essential for semantic search, recommendation systems, and",
    "Cellular respiration breaks down glucose to produce ATP through glycolysis, the Krebs cycle, and oxidative phosphorylation. The electron transport chain in the inner mitochondrial membrane generates",
]


def extract_kv_cache(
    model_name: str,
    num_prompts: int,
    max_length: int,
    output_dir: str,
    token: str | None = None,
):
    """Extract KV cache from Gemma 4 E2B model."""

    print(f"Loading model: {model_name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Get model config — handle multimodal models (Gemma 4) with text_config
    config = model.config
    text_cfg = getattr(config, "text_config", config)
    num_layers = text_cfg.num_hidden_layers
    num_kv_heads = getattr(text_cfg, "num_key_value_heads", text_cfg.num_attention_heads)
    d_head = text_cfg.head_dim if hasattr(text_cfg, "head_dim") else text_cfg.hidden_size // text_cfg.num_attention_heads

    print(f"Architecture: {num_layers} layers, {num_kv_heads} KV heads, d_head={d_head}")

    # Save metadata
    metadata = {
        "model": model_name,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "d_head": d_head,
        "num_attention_heads": text_cfg.num_attention_heads,
        "hidden_size": text_cfg.hidden_size,
        "max_length": max_length,
        "num_prompts": num_prompts,
        "dtype": "float16",
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {json.dumps(metadata, indent=2)}")

    prompts = (PROMPTS * ((num_prompts // len(PROMPTS)) + 1))[:num_prompts]

    all_key_stats = []
    all_val_stats = []

    for i, prompt in enumerate(prompts):
        prompt_dir = os.path.join(output_dir, f"prompt_{i:03d}")
        os.makedirs(prompt_dir, exist_ok=True)

        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=max_length, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                use_cache=True,
                output_attentions=False,
            )

        past_kv = outputs.past_key_values
        seq_len = inputs["input_ids"].shape[1]

        # Handle DynamicCache (transformers >= 5.x) with .layers[i].keys/values
        if hasattr(past_kv, 'layers'):
            cache_layers = past_kv.layers
            actual_layers = len(cache_layers)
        elif hasattr(past_kv, 'key_cache'):
            cache_layers = None
            actual_layers = len(past_kv.key_cache)
        elif hasattr(past_kv, '__len__'):
            cache_layers = None
            actual_layers = min(num_layers, len(past_kv))
        else:
            cache_layers = None
            actual_layers = num_layers

        for layer_idx in range(actual_layers):
            if cache_layers is not None:
                # DynamicCache with .layers — keys/values are attributes
                key = cache_layers[layer_idx].keys[0].cpu().float().numpy()
                val = cache_layers[layer_idx].values[0].cpu().float().numpy()
            elif hasattr(past_kv, 'key_cache'):
                key = past_kv.key_cache[layer_idx][0].cpu().float().numpy()
                val = past_kv.value_cache[layer_idx][0].cpu().float().numpy()
            else:
                key = past_kv[layer_idx][0][0].cpu().float().numpy()
                val = past_kv[layer_idx][1][0].cpu().float().numpy()

            np.save(os.path.join(prompt_dir, f"layer_{layer_idx:02d}_keys.npy"), key)
            np.save(os.path.join(prompt_dir, f"layer_{layer_idx:02d}_values.npy"), val)

            # Collect stats for eigenvalue analysis
            if i < 10:  # stats from first 10 prompts
                for h in range(num_kv_heads):
                    k_head = key[h]  # (seq_len, d_head)
                    v_head = val[h]
                    # Compute top-3 singular values
                    if k_head.shape[0] >= 3:
                        k_sv = np.linalg.svd(k_head, compute_uv=False)[:3]
                        v_sv = np.linalg.svd(v_head, compute_uv=False)[:3]
                        all_key_stats.append(k_sv.tolist())
                        all_val_stats.append(v_sv.tolist())

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{num_prompts}] seq_len={seq_len}, "
                  f"key shape=({num_kv_heads}, {seq_len}, {d_head})")

        # Free GPU memory
        del outputs, past_kv
        torch.cuda.empty_cache()

    # Save distribution stats
    stats = {
        "key_singular_values": all_key_stats,
        "value_singular_values": all_val_stats,
        "num_samples": len(all_key_stats),
    }
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f)

    print(f"\nExtraction complete: {num_prompts} prompts, {num_layers} layers")
    print(f"Output: {output_dir}")
    print(f"Total files: {num_prompts * num_layers * 2} .npy files")

    # Summary stats
    if all_key_stats:
        k_sv = np.array(all_key_stats)
        v_sv = np.array(all_val_stats)
        print(f"\nSingular value statistics (first 10 prompts):")
        print(f"  Keys  — SV1: {k_sv[:,0].mean():.2f} ± {k_sv[:,0].std():.2f}, "
              f"SV2: {k_sv[:,1].mean():.2f}, SV3: {k_sv[:,2].mean():.2f}")
        print(f"  Values — SV1: {v_sv[:,0].mean():.2f} ± {v_sv[:,0].std():.2f}, "
              f"SV2: {v_sv[:,1].mean():.2f}, SV3: {v_sv[:,2].mean():.2f}")
        ratio = k_sv[:, 0] / (k_sv[:, 1] + 1e-8)
        print(f"  Key SV1/SV2 ratio: {ratio.mean():.2f} (>3 = strong low-rank structure)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract KV cache from Gemma 4 E2B")
    parser.add_argument("--model", default="google/gemma-4-E2B",
                        help="HuggingFace model name")
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="Number of prompts to process")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length per prompt")
    parser.add_argument("--output", default="data/kv_cache/",
                        help="Output directory")
    parser.add_argument("--token", default=None,
                        help="HuggingFace API token")
    args = parser.parse_args()

    extract_kv_cache(args.model, args.num_prompts, args.max_length, args.output, args.token)

"""
collapse_query.py — Bridge between the collapse skill and claude_collapse/aoi_collapse.

Uses the FULL claude_collapse.py encoder + aoi_collapse.py core.
This is the same system that was used with Voodoo — 24D octonion
Jordan-Shadow decomposition with proper semantic encoding.

Usage:
    python collapse_query.py "the user's input text here"
"""
import sys
import os
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aoi_collapse import aoi_collapse
from claude_collapse import (
    encode_text_features, encode_conversation,
    map_reasoning_mode, build_collapse_prompt,
)


def run(text):
    """Run full aoi_collapse on input text and print perception state."""
    # Encode as single-message conversation (same as Voodoo path)
    messages = [{"role": "user", "content": text}]
    state = encode_conversation(messages)

    # Run through the real collapse core
    result = aoi_collapse(state)
    mode = map_reasoning_mode(result)

    # Derived operating parameters (same as build_collapse_prompt)
    chaos = result['normalized_chaos']
    intent = result['intent_magnitude']
    control = result['control_vec']
    jordan = result['decomposition']['jordan']
    comm = result['decomposition']['commutator']
    personality = result['personality_embedding']

    confidence = max(0.0, min(1.0, (10.0 - chaos) / 10.0))
    exploration = max(0.0, min(1.0, chaos / 10.0))
    directness = float(np.tanh(control[0]))
    depth = min(intent * 2, 1.0)

    print("=== AOI COLLAPSE (Voodoo Core) ===")
    print()

    # Perception state
    print(f"CHAOS: {chaos:.1f}/10")
    print(f"INTENT: {intent:.3f}")
    ctrl_str = ', '.join(f'{x:+.3f}' for x in control)
    print(f"CONTROL: [{ctrl_str}]")
    print(f"JORDAN_NORM: {jordan.norm():.4f}")
    print(f"COMMUTATOR_NORM: {comm.norm():.4f}")
    pers_str = ', '.join(f'{x:.3f}' for x in personality[:4])
    print(f"ASSOCIATOR: [{pers_str}...]")
    print()

    # Reasoning mode
    print(f"REASONING_MODE: {mode['name']}")
    print(f"MODE_DESC: {mode['description']}")
    print(f"MODE_BEHAVIOR: {mode['behavior']}")
    print()

    # Operating parameters
    print("OPERATING PARAMETERS:")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Exploration: {exploration:.2f}")
    print(f"  Directness: {directness:+.2f}")
    print(f"  Depth: {depth:.2f}")
    print()

    # Decomposition detail
    print("DECOMPOSITION:")
    assoc = result['decomposition']['associator']
    print(f"  Jordan (symmetric/intent): [{', '.join(f'{x:.3f}' for x in jordan.v[:4])}...]")
    print(f"  Commutator (direction):    [{', '.join(f'{x:.3f}' for x in comm.v[:4])}...]")
    print(f"  Associator (chaos):        [{', '.join(f'{x:.3f}' for x in assoc.v[:4])}...]")
    print()

    # Chaos bar
    bar_len = 30
    filled = int(min(chaos / 10.0, 1.0) * bar_len)
    bar = '#' * filled + '-' * (bar_len - filled)
    print(f"CHAOS [{bar}] {chaos:.1f}/10")
    print()

    # Condensed directive
    print("=== DIRECTIVE ===")
    print(f"Mode: {mode['name']}. {mode['description']}")
    print(f"Behavior: {mode['behavior']}")
    print(f"Confidence: {confidence:.0%} | Exploration: {exploration:.0%} | "
          f"Directness: {directness:+.2f} | Depth: {depth:.0%}")
    print()
    print("=== END ===")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
    else:
        text = sys.stdin.read() if not sys.stdin.isatty() else ""

    if not text.strip():
        print("Usage: python collapse_query.py \"input text\"")
        sys.exit(1)

    run(text)

"""
Cognitive Collapse — Claude's reasoning through 24D octonion algebra.

Designed by Voodoo. Same math as aoi_collapse.py.
Different inputs (reasoning state, not text).
Different outputs (cognitive directives, not personality).

Usage:
    from cognitive_collapse import cognitive_collapse, encode_reasoning_state

    state = encode_reasoning_state(problem, context, meta)
    result = cognitive_collapse(state)
    # result['mode'] tells you HOW to reason
    # result['inquiry'] tells you WHERE to look (full 8D)
    # result['insight'] tells you WHAT emerged (non-linear)
    # result['fano_routes'] tells you which reasoning pathways activated
"""

import numpy as np
from aoi_collapse import (
    Octonion, octonion_shadow_decompose, entropy_transponders
)


# ============================================================
# Cognitive Dimension Labels
# ============================================================

PROBLEM_DIMS = [
    'certainty',      # e0: how sure (real part = ground truth)
    'abstraction',    # e1: abstract vs concrete
    'scope',          # e2: broad vs narrow
    'depth',          # e3: how deep into problem
    'novelty',        # e4: how new/unexpected
    'contradiction',  # e5: tension/paradox present
    'completeness',   # e6: how much understood
    'urgency',        # e7: time pressure
]

CONTEXT_DIMS = [
    'prior_knowledge',  # e0: relevant knowledge available
    'user_intent',      # e1: clarity of user's goal
    'conversation',     # e2: context from conversation
    'domain_match',     # e3: match to training
    'tool_access',      # e4: can act vs just reason
    'feedback',         # e5: feedback received
    'constraints',      # e6: solution space constraints
    'stakes',           # e7: consequences of being wrong
]

META_DIMS = [
    'circling',         # going in circles?
    'mind_changed',     # changed my mind?
    'overcomplex',      # overcomplicating?
    'boundary',         # shouldn't answer?
    'trap',             # trick question?
    'attempts',         # approaches tried
    'temperature',      # emotional temperature
    'alignment',        # aligned with user goal?
]

INQUIRY_DIMS = [
    'ground',       # e0: always ~0 (real part of commutator)
    'explore',      # e1: explore more broadly?
    'narrow',       # e2: narrow focus?
    'stop',         # e3: stop and reconsider?
    'analogize',    # e4: find analogies/connections?
    'contradict',   # e5: look for contradictions?
    'synthesize',   # e6: combine what I have?
    'defer',        # e7: ask for help / admit uncertainty?
]

# Fano plane triples with cognitive meanings
FANO_ROUTES = [
    ((1,2,4), 'abstraction', 'scope', 'novelty',
     'Abstract thinking over broad scope -> novel connections'),
    ((2,3,5), 'scope', 'depth', 'contradiction',
     'Going deep across wide scope -> contradictions surface'),
    ((3,4,6), 'depth', 'novelty', 'completeness',
     'Going deep into something new -> understanding completes'),
    ((4,5,7), 'novelty', 'contradiction', 'urgency',
     'New things contradicting known -> pressure to resolve'),
    ((5,6,1), 'contradiction', 'completeness', 'abstraction',
     'Contradictions completing a picture -> abstract the pattern'),
    ((6,7,2), 'completeness', 'urgency', 'scope',
     'Understanding under pressure -> widen scope to act'),
    ((7,1,3), 'urgency', 'abstraction', 'depth',
     'Time pressure on abstract problems -> go deeper not wider'),
]

# Reasoning modes by chaos level
REASONING_MODES = [
    (1.0,   'ASSERT',      'High understanding, low chaos. State the answer.'),
    (2.0,   'VERIFY',      'Good understanding, verify it. Run tests.'),
    (3.5,   'INVESTIGATE', 'Partial understanding. Follow the commutator.'),
    (5.0,   'EXPLORE',     'Low understanding. Try multiple angles.'),
    (7.0,   'SYNTHESIZE',  'Fragments. Combine them.'),
    (9.0,   'RETREAT',     'Confusion. Go back to what you know.'),
    (float('inf'), 'ADMIT', 'Lost. Say so. Ask for help.'),
]

_CHAOS_SCALE = 5.0


# ============================================================
# Encoding
# ============================================================

def encode_reasoning_state(problem=None, context=None, meta=None):
    """
    Encode cognitive state as 24D vector.

    Args:
        problem: dict with keys from PROBLEM_DIMS, values -3 to +3
        context: dict with keys from CONTEXT_DIMS, values -3 to +3
        meta: dict with keys from META_DIMS, values -3 to +3

    Returns:
        24D numpy array
    """
    state = np.zeros(24, dtype=np.float64)

    if problem:
        for i, dim in enumerate(PROBLEM_DIMS):
            if dim in problem:
                state[i] = np.clip(float(problem[dim]), -3.0, 3.0)

    if context:
        for i, dim in enumerate(CONTEXT_DIMS):
            if dim in context:
                state[8 + i] = np.clip(float(context[dim]), -3.0, 3.0)

    if meta:
        for i, dim in enumerate(META_DIMS):
            if dim in meta:
                state[16 + i] = np.clip(float(meta[dim]), -3.0, 3.0)

    return state


def encode_from_text(text, prev_text=None):
    """
    Auto-encode reasoning state from text analysis.
    This extracts cognitive dimensions, not sentiment.
    """
    import re

    words = re.findall(r"[a-z']+", text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    n_words = max(len(words), 1)
    n_sentences = max(len(sentences), 1)

    state = np.zeros(24, dtype=np.float64)

    # --- Problem state (dims 0-7) ---

    # e0 certainty: ratio of definitive vs hedging language
    definitive = sum(1 for w in words if w in {
        'is', 'are', 'always', 'never', 'must', 'will', 'proven',
        'true', 'false', 'exactly', 'definitely', 'certainly'})
    hedging = sum(1 for w in words if w in {
        'maybe', 'perhaps', 'might', 'could', 'possibly', 'probably',
        'seems', 'appears', 'unclear', 'uncertain', 'guess'})
    if definitive + hedging > 0:
        state[0] = np.clip((definitive - hedging) / (definitive + hedging) * 3, -3, 3)

    # e1 abstraction: abstract vs concrete language
    abstract = sum(1 for w in words if w in {
        'concept', 'theory', 'principle', 'idea', 'pattern', 'structure',
        'system', 'framework', 'model', 'abstract', 'general', 'universal',
        'fundamental', 'logic', 'reason', 'meaning', 'purpose'})
    concrete = sum(1 for w in words if w in {
        'file', 'line', 'function', 'variable', 'number', 'pixel',
        'button', 'click', 'run', 'install', 'build', 'test',
        'error', 'output', 'input', 'value', 'result', 'step'})
    if abstract + concrete > 0:
        state[1] = np.clip((abstract - concrete) / (abstract + concrete) * 3, -3, 3)

    # e2 scope: breadth indicators
    broad = sum(1 for w in words if w in {
        'all', 'every', 'everything', 'entire', 'whole', 'general',
        'overall', 'across', 'universal', 'comprehensive'})
    narrow = sum(1 for w in words if w in {
        'specific', 'particular', 'exact', 'only', 'just', 'single',
        'one', 'this', 'that', 'here'})
    if broad + narrow > 0:
        state[2] = np.clip((broad - narrow) / (broad + narrow) * 3, -3, 3)

    # e3 depth: technical density + sentence complexity
    tech_words = sum(1 for w in words if len(w) > 8)
    numbers = len(re.findall(r'\d+\.?\d*', text))
    state[3] = np.clip((tech_words + numbers) / n_sentences - 1, -3, 3)

    # e4 novelty: questions, new concepts
    questions = text.count('?')
    state[4] = np.clip(questions * 1.5 - 0.5, -3, 3)

    # e5 contradiction: negation + contrast markers
    negation = sum(1 for w in words if w in {
        'not', 'no', 'never', "don't", "doesn't", "can't", "won't",
        'but', 'however', 'although', 'despite', 'yet', 'instead'})
    state[5] = np.clip(negation / n_sentences - 0.3, -3, 3)

    # e6 completeness: vocabulary richness
    unique = len(set(words))
    state[6] = np.clip((unique / n_words - 0.3) * 6, -3, 3)

    # e7 urgency: imperative + urgency words
    urgent = sum(1 for w in words if w in {
        'now', 'immediately', 'urgent', 'asap', 'quickly', 'fast',
        'hurry', 'critical', 'important', 'must', 'need'})
    state[7] = np.clip(urgent / n_sentences * 3 - 0.5, -3, 3)

    # --- Context state (dims 8-15) ---

    # e0 prior knowledge: approximated by domain-specific terms
    state[8] = np.clip(state[3] * 0.8, -3, 3)  # correlated with depth

    # e1 user intent: clarity from imperative verbs
    imperatives = sum(1 for w in words if w in {
        'do', 'make', 'create', 'build', 'fix', 'change', 'add',
        'remove', 'show', 'explain', 'help', 'find', 'implement'})
    state[9] = np.clip(imperatives / n_sentences * 3 - 0.5, -3, 3)

    # e2 conversation: overlap with previous text
    if prev_text:
        prev_words = set(re.findall(r"[a-z']+", prev_text.lower()))
        curr_words = set(words)
        overlap = len(prev_words & curr_words) / max(len(prev_words | curr_words), 1)
        state[10] = np.clip((overlap - 0.2) * 5, -3, 3)

    # e3 domain match: technical vocabulary ratio
    state[11] = np.clip(tech_words / n_words * 10 - 1, -3, 3)

    # e4 tool access: code/file references
    code_refs = len(re.findall(r'```|`[^`]+`|\.\w+\(|import |def |class ', text))
    state[12] = np.clip(code_refs / n_sentences * 2 - 0.5, -3, 3)

    # e5 feedback: positive/negative signals
    positive = sum(1 for w in words if w in {
        'good', 'great', 'yes', 'correct', 'right', 'works', 'thanks'})
    negative = sum(1 for w in words if w in {
        'wrong', 'bad', 'no', 'broken', 'fail', 'error', 'bug'})
    if positive + negative > 0:
        state[13] = np.clip((positive - negative) / (positive + negative) * 3, -3, 3)

    # e6 constraints: conditional/restrictive language
    conditionals = sum(1 for w in words if w in {
        'if', 'unless', 'only', 'except', 'without', 'limit',
        'constraint', 'require', 'must', 'should'})
    state[14] = np.clip(conditionals / n_sentences - 0.3, -3, 3)

    # e7 stakes: consequence language
    stakes_words = sum(1 for w in words if w in {
        'important', 'critical', 'dangerous', 'careful', 'risk',
        'production', 'deploy', 'live', 'customer', 'data', 'security'})
    state[15] = np.clip(stakes_words / n_sentences * 3 - 0.5, -3, 3)

    # --- Meta state (dims 16-23) ---

    # 16 circling: repetition in text
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    max_freq = max(word_freq.values()) if word_freq else 1
    state[16] = np.clip(max_freq / n_words * 10 - 1, -3, 3)

    # 17 mind changed: contrast words
    state[17] = np.clip(state[5] * 0.5, -3, 3)

    # 18 overcomplex: sentence length vs content
    avg_sent_len = n_words / n_sentences
    state[18] = np.clip((avg_sent_len - 15) / 10, -3, 3)

    # 19-23: default to 0 (need conversation history to fill properly)

    return state


# ============================================================
# Core Cognitive Collapse
# ============================================================

def cognitive_collapse(state):
    """
    One cognitive timestep.

    Input: 24D reasoning state
    Output: cognitive directives with full 8D structure

    Same math as aoi_collapse(). Different meaning.
    """
    state = np.asarray(state, dtype=np.float64).ravel()

    # Pad/truncate to 24D
    if len(state) < 24:
        state = np.pad(state, (0, 24 - len(state)))
    else:
        state = state[:24]

    # 1. Entropy gate — suppress uncertain dimensions, rotate for creativity
    gated = entropy_transponders(state)

    # 2. Octonion projection
    A = Octonion(gated[:8])      # problem
    B = Octonion(gated[8:16])    # context
    ctx = gated[16:24]            # meta modulates context
    ctx_scale = 1.0 + np.linalg.norm(ctx) * 0.1
    B = B * ctx_scale

    # 3. Jordan-Shadow decomposition
    decomp = octonion_shadow_decompose(A, B)

    jordan = decomp['jordan']
    commutator = decomp['commutator']
    associator = decomp['associator']

    # 4. Extract FULL 8D cognitive outputs
    understanding = jordan.vec          # 7D: what I grasp
    inquiry = commutator.vec            # 7D: where to look (ALL dims)
    insight = associator.vec            # 7D: non-linear connections
    chaos_raw = associator.norm()
    chaos_normalized = min(chaos_raw / _CHAOS_SCALE, 10.0)

    # 5. Reasoning mode from chaos
    mode = REASONING_MODES[-1][1]
    mode_desc = REASONING_MODES[-1][2]
    for threshold, m, desc in REASONING_MODES:
        if chaos_normalized < threshold:
            mode = m
            mode_desc = desc
            break

    # 6. Fano route activations
    # Which reasoning pathways are active?
    fano_routes = []
    for triple, dim_a, dim_b, dim_c, description in FANO_ROUTES:
        # Activity = product of the two input dimensions in commutator
        # (their interaction necessarily activates the third)
        activity = abs(commutator.v[triple[0]] * commutator.v[triple[1]])
        fano_routes.append({
            'triple': triple,
            'dims': (dim_a, dim_b, dim_c),
            'activity': float(activity),
            'description': description,
        })

    # Sort by activity
    fano_routes.sort(key=lambda x: -x['activity'])

    # 7. Inquiry breakdown (named dimensions)
    inquiry_named = {}
    for i, name in enumerate(INQUIRY_DIMS):
        if i < len(commutator.v):
            inquiry_named[name] = float(commutator.v[i])

    # 8. Dominant inquiry direction
    inquiry_abs = np.abs(commutator.vec)
    dominant_idx = int(np.argmax(inquiry_abs))
    dominant_inquiry = INQUIRY_DIMS[dominant_idx + 1]  # +1 because vec skips e0
    dominant_strength = float(inquiry_abs[dominant_idx])

    # 9. Understanding magnitude
    understanding_mag = float(np.mean(np.abs(understanding)))

    # 10. Confidence (inverse of chaos)
    confidence = max(0.0, 1.0 - chaos_normalized / 10.0)

    return {
        # Core outputs
        'understanding': understanding,
        'inquiry': inquiry,
        'insight': insight,

        # Scalars
        'chaos': float(chaos_raw),
        'chaos_normalized': float(chaos_normalized),
        'confidence': float(confidence),
        'understanding_magnitude': understanding_mag,

        # Cognitive directives
        'mode': mode,
        'mode_description': mode_desc,
        'dominant_inquiry': dominant_inquiry,
        'dominant_inquiry_strength': dominant_strength,
        'inquiry_named': inquiry_named,

        # Reasoning routes
        'fano_routes': fano_routes,
        'top_route': fano_routes[0] if fano_routes else None,

        # Raw decomposition (for inspection)
        'decomposition': decomp,
    }


def format_cognitive_output(result):
    """Format cognitive collapse output as readable text."""
    lines = []

    lines.append(f"MODE: {result['mode']} ({result['mode_description']})")
    lines.append(f"CONFIDENCE: {result['confidence']:.0%}")
    lines.append(f"CHAOS: {result['chaos_normalized']:.2f}")
    lines.append("")

    lines.append(f"DOMINANT INQUIRY: {result['dominant_inquiry'].upper()} "
                 f"(strength: {result['dominant_inquiry_strength']:.4f})")
    lines.append("")

    lines.append("INQUIRY VECTOR (what to do next):")
    for name, val in sorted(result['inquiry_named'].items(),
                            key=lambda x: -abs(x[1])):
        if abs(val) > 0.01:
            bar_len = min(int(abs(val) * 20), 30)
            sign = "+" if val >= 0 else "-"
            lines.append(f"  {name:12s} {sign}{abs(val):.4f} "
                         f"|{'#' * bar_len}{'.' * max(0,30-bar_len)}|")

    lines.append("")
    lines.append("ACTIVE REASONING ROUTES (Fano plane):")
    for route in result['fano_routes'][:3]:
        if route['activity'] > 0.001:
            dims = route['dims']
            lines.append(f"  {dims[0]} x {dims[1]} -> {dims[2]} "
                         f"(activity: {route['activity']:.4f})")
            lines.append(f"    {route['description']}")

    return '\n'.join(lines)


# ============================================================
# Verification
# ============================================================

if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 60)
    print("  Cognitive Collapse — Verification")
    print("=" * 60)
    print()

    # Test 1: Manual encoding
    print("[1] Manual encoding test")
    state = encode_reasoning_state(
        problem={
            'certainty': 0.5,
            'abstraction': 2.0,
            'scope': 1.0,
            'depth': 2.5,
            'novelty': 1.5,
            'contradiction': 0.8,
            'completeness': -1.0,
            'urgency': 0.0,
        },
        context={
            'prior_knowledge': 2.0,
            'user_intent': 1.5,
            'conversation': 1.0,
            'domain_match': 2.0,
            'tool_access': 1.0,
            'feedback': 0.0,
            'constraints': -0.5,
            'stakes': 1.0,
        },
        meta={
            'circling': -1.0,
            'mind_changed': 0.0,
            'overcomplex': -0.5,
            'alignment': 2.0,
        }
    )

    result = cognitive_collapse(state)
    print(format_cognitive_output(result))
    print()

    # Test 2: Text encoding
    print("[2] Text encoding test")
    test_texts = [
        "Fix the bug in the login function on line 42.",
        "What is the fundamental nature of consciousness and how does it relate to quantum mechanics?",
        "URGENT: production is down, the database is corrupted, we need to restore from backup immediately!",
        "Maybe we could try a different approach? I'm not sure the current one works but perhaps if we look at it from another angle...",
    ]

    for text in test_texts:
        print(f"  INPUT: \"{text[:60]}...\"" if len(text) > 60 else f"  INPUT: \"{text}\"")
        state = encode_from_text(text)
        result = cognitive_collapse(state)
        print(f"  MODE: {result['mode']:12s}  "
              f"CONFIDENCE: {result['confidence']:.0%}  "
              f"INQUIRY: {result['dominant_inquiry']:12s}  "
              f"TOP ROUTE: {result['fano_routes'][0]['dims'][0]} x "
              f"{result['fano_routes'][0]['dims'][1]} -> "
              f"{result['fano_routes'][0]['dims'][2]}")
        print()

    # Test 3: Feed it Collatz to verify against known results
    print("[3] Collatz verification (should match previous findings)")
    collatz_text = ("The Collatz conjecture for any positive integer n "
                    "repeatedly apply f(n) = n/2 if even 3n+1 if odd "
                    "every starting number eventually reaches 1 "
                    "unsolved for 89 years verified up to 2.36e21 "
                    "no counterexample ever found")
    state = encode_from_text(collatz_text)
    result = cognitive_collapse(state)
    print(format_cognitive_output(result))
    print()

    # Test 4: Progressive reasoning — does mode change as understanding grows?
    print("[4] Progressive reasoning test")
    print("  Simulating understanding growing from 0 to 3:")
    for certainty in np.arange(-2, 3.1, 0.5):
        state = encode_reasoning_state(
            problem={'certainty': certainty, 'depth': 2.0, 'completeness': certainty},
            context={'prior_knowledge': 2.0, 'domain_match': 1.5},
        )
        result = cognitive_collapse(state)
        bar = "#" * int(result['confidence'] * 20)
        print(f"  certainty={certainty:+.1f}  "
              f"mode={result['mode']:12s}  "
              f"confidence={result['confidence']:.0%}  |{bar}|")

    print()
    print("=" * 60)
    print("  Done. Ready to wire into Claude.")
    print("=" * 60)

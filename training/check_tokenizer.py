import json

snap = r'E:\devAI\boltassistant\training\model_cache\models--google--functiongemma-270m-it\snapshots\39eccb091651513a5dfb56892d3714c1b5b8276c'

tj = json.load(open(snap + r'\tokenizer.json', encoding='utf-8'))
vocab = tj['model']['vocab']
merges = tj['model'].get('merges', [])

print("vocab type:", type(vocab).__name__, " entries:", len(vocab))
print("merges count:", len(merges))

# Show first 15 vocab items sorted by id
items = sorted(vocab.items(), key=lambda x: x[1])
print("\nFirst 15 tokens by id:")
for tok, tid in items[:15]:
    print(f"  id={tid}: {repr(tok)}")

# Show special tokens in vocab
print("\nSpecial tokens in vocab:")
for sp in ['<pad>', '<eos>', '<bos>', '<unk>', '<start_of_turn>', '<end_of_turn>']:
    if sp in vocab:
        print(f"  {repr(sp)} -> id={vocab[sp]}")

# Check added_tokens with special=True
added_special = [t for t in tj['added_tokens'] if t.get('special')]
print(f"\nadded_tokens with special=True: {len(added_special)}")
for t in added_special[:20]:
    tid = t['id']
    tc = t['content']
    print(f"  id={tid}: {repr(tc)}")

# Show merges sample
if merges:
    print(f"\nFirst 5 merges:")
    for m in merges[:5]:
        print(f"  {m}")

# Count unused tokens
unused = [tok for tok in vocab if tok.startswith('<unused')]
print(f"\nunused tokens: {len(unused)}")
real_tokens = len(vocab) - len(unused)
print(f"real tokens (non-unused): {real_tokens}")

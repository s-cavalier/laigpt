from core.tokenizer import WordTokenizer

data = "the cat sat on the mat"
tok = WordTokenizer(data)

print("Vocab:", tok.keys())
print("Vocab size:", len(tok))

encoded = tok.encode("the cat sat")
print("Encoded:", encoded)

decoded = tok.decode(encoded)
print("Decoded:", decoded)

print("ID of 'mat':", tok["mat"])

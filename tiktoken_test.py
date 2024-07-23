import tiktoken
enc = tiktoken.get_encoding("o200k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

tokenizers = tiktoken.list_encoding_names()

print(tokenizers)

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4o-mini")

#encode:
enc.encode("hello world")

#print
print(enc.decode(enc.encode("hello world")))
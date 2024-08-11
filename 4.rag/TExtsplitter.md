

# Text Splitters in LangChain

LangChain provides several text splitters to help break down large pieces of text into manageable chunks. This is especially important when working with large documents or when dealing with language models that have token limits. Below are the main types of text splitters and their differences, with examples.

## 1. **Character-based Text Splitter (`CharacterTextSplitter`)**

### Description
The `CharacterTextSplitter` divides text into chunks based on a specified character or sequence of characters, such as spaces, newlines, or custom delimiters. This splitter is useful when you need to split text at specific points while maintaining complete sentences or paragraphs.

### Parameters
- `chunk_size`: The maximum number of characters in each chunk.
- `chunk_overlap`: The number of characters that should overlap between consecutive chunks.

### Example
```python
from langchain.text_splitter import CharacterTextSplitter

text = "This is a simple example text. It will be split based on characters."
splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5)
chunks = splitter.split_text(text)
print(chunks)
```

**Output:**
```python
[
  "This is a simple ex",
  "simple example tex",
  "example text. It wi",
  "text. It will be sp",
  "be split based on c"
]
```

## 2. **Sentence-based Text Splitter (`SentenceTextSplitter`)**

### Description
The `SentenceTextSplitter` splits text based on sentence boundaries, using punctuation as delimiters. This method ensures that each chunk contains whole sentences, which can help maintain the context and meaning of the text.

### Parameters
- `chunk_size`: The maximum number of sentences in each chunk.
- `chunk_overlap`: The number of sentences that should overlap between consecutive chunks.

### Example
```python
from langchain.text_splitter import SentenceTextSplitter

text = "This is the first sentence. This is the second sentence. Here is the third sentence."
splitter = SentenceTextSplitter(chunk_size=1, chunk_overlap=0)
chunks = splitter.split_text(text)
print(chunks)
```

**Output:**
```python
[
  "This is the first sentence.",
  "This is the second sentence.",
  "Here is the third sentence."
]
```

## 3. **Token-based Text Splitter (`TokenTextSplitter`)**

### Description
The `TokenTextSplitter` splits text based on tokens, which are units used by language models (e.g., words, subwords). This splitter ensures that each chunk stays within the token limit of the model, making it ideal for processing text with models like GPT-3 or GPT-4.

### Parameters
- `max_tokens`: The maximum number of tokens in each chunk.

### Example
```python
from langchain.text_splitter import TokenTextSplitter

text = "This is a token-based text splitter example. It is useful for managing token limits."
splitter = TokenTextSplitter(max_tokens=10)
chunks = splitter.split_text(text)
print(chunks)
```

**Output:**
```python
[
  "This is a token-based text splitter",
  "example. It is useful for managing token limits."
]
```

## 4. **Recursive Character-based Text Splitter (`RecursiveCharacterTextSplitter`)**

### Description
The `RecursiveCharacterTextSplitter` splits text recursively, starting with larger units like paragraphs and then working down to smaller units like sentences and words. This method is useful for preserving the hierarchical structure of the text while still fitting it into manageable chunks.

### Parameters
- `chunk_size`: The maximum number of characters in each chunk.
- `chunk_overlap`: The number of characters that should overlap between consecutive chunks.
- `separators`: A list of characters used to split the text hierarchically.

### Example
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "This is a paragraph.\n\nThis is another paragraph that will be split into smaller chunks."
splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=10)
chunks = splitter.split_text(text)
print(chunks)
```

**Output:**
```python
[
  "This is a paragraph.",
  "This is another paragraph that will be",
  "paragraph that will be split into smaller chunks."
]
```

## 5. **Token Text Splitter (`TokenTextSplitter`)**

### Description
Similar to the Token-based Text Splitter, the `TokenTextSplitter` splits text based on tokens, ensuring the chunks fit within the token limit. This is particularly useful for handling text in a way that aligns with the tokenization process of the language model.

### Parameters
- `max_tokens`: The maximum number of tokens in each chunk.

### Example
```python
from langchain.text_splitter import TokenTextSplitter

text = "TokenTextSplitter helps manage token-based text splitting, ideal for large models."
splitter = TokenTextSplitter(max_tokens=12)
chunks = splitter.split_text(text)
print(chunks)
```

**Output:**
```python
[
  "TokenTextSplitter helps manage",
  "token-based text splitting, ideal for",
  "large models."
]
```

## Conclusion

Each text splitter in LangChain serves a unique purpose, depending on the structure of the text and the specific requirements of your application. Whether you need to maintain sentence boundaries, respect token limits, or preserve the hierarchical structure of text, LangChain provides a text splitter that fits your needs.

--- 

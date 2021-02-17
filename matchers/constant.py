import string

BEGIN_TOKEN = '<'
END_TOKEN = '>'
ALPHABET = string.ascii_lowercase + BEGIN_TOKEN + END_TOKEN
VOCAB_SIZE = len(ALPHABET)

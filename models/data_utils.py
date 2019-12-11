import difflib
import os
import re
import sys
import javalang
import json

JAVA_KEYWORDS = ['abstract', 'continue', 'for', 'new', 'switch', 'package', 'synchronized', 
'do', 'if', 'private', 'this', 'break', 'implements', 'protected', 'throw', 'throws', 'else',
'import', 'public', 'case', 'instanceof', 'return', 'transient', 'catch', 'extends', 'try', 'final',
'interface', 'static', 'void', 'finally', 'super', 'while']

JAVA_TYPES = ['boolean', 'byte', 'char', 'short', 'int', 'long',
'float', 'double', 'String', 'null']

SPECIAL_TAGS = ['{', '}', '@code', '@docRoot', '@inheritDoc', '@link', '@linkplain', '@value']

class CodeToken:
    def __init__(self, value, token_change_type):
        self.value = value
        self.token_change_type = token_change_type

class CodeRepr:
    def __init__(self, tokenized_projection, subtokenized_projection,
        tokenized_diff_only, subtokenized_diff_only, diff_only):
        self.tokenized_projection = tokenized_projection
        self.subtokenized_projection = subtokenized_projection
        self.tokenized_diff_only = tokenized_diff_only
        self.subtokenized_diff_only = subtokenized_diff_only

        if diff_only:
            self.tokenized = tokenized_diff_only
            self.subtokenized = subtokenized_diff_only
        else:
            self.tokenized = tokenized_projection
            self.subtokenized = subtokenized_projection

        self.new_tokenized = [tok for tok in self.tokenized if tok.token_change_type == TokenChangeType.ADD or tok.token_change_type == TokenChangeType.KEEP]
        self.old_tokenized = [tok for tok in self.tokenized if tok.token_change_type == TokenChangeType.REMOVE or tok.token_change_type == TokenChangeType.KEEP]
        self.new_subtokenized = [tok for tok in self.subtokenized if tok.token_change_type == TokenChangeType.ADD or tok.token_change_type == TokenChangeType.KEEP]
        self.old_subtokenized = [tok for tok in self.subtokenized if tok.token_change_type == TokenChangeType.REMOVE or tok.token_change_type == TokenChangeType.KEEP]

        self.new_tokenized_add_only = [tok for tok in self.tokenized if tok.token_change_type == TokenChangeType.ADD]
        self.old_tokenized_remove_only = [tok for tok in self.tokenized if tok.token_change_type == TokenChangeType.REMOVE]
        self.new_subtokenized_add_only = [tok for tok in self.subtokenized if tok.token_change_type == TokenChangeType.ADD]
        self.old_subtokenized_remove_only = [tok for tok in self.subtokenized if tok.token_change_type == TokenChangeType.REMOVE]

class CommentLineRepr:
    def __init__(self, comment_line, tokenized_comment_line, token_change_type):
        self.comment_line = comment_line
        self.tokenized_comment_line = tokenized_comment_line
        self.token_change_type = token_change_type

class UpdateExample:
    def __init__(self, diff, add_line, remove_line, tokenized_diff_code):
        self.diff = diff
        self.add_line = add_line 
        self.remove_line = remove_line 
        self.tokenized_diff_code = tokenized_diff_code
        
        self.code_token_ids = []
        self.code_subtoken_ids = []
        self.add_comment_token_ids = []
        self.remove_comment_token_ids = []

        self.oovs = []
        self.code_extended_token_ids = []
        self.code_extended_subtoken_ids = []

        # TODO: Need to do at subtoken level as well?
        self.remove_comment_extended_token_ids = []
        
        self.add_comment_extended_token_ids = []
        self.edit_sequence_commands = []
        self.encoded_edit_sequence_commands = []
        
        self.complete_edit_sequence_commands = []
        self.complete_edit_sequence_ids = []
        self.complete_extended_edit_sequence_ids = []

        # For the purpose of encoding new and old code separately
        self.new_code_token_ids = []
        self.new_code_subtoken_ids = []
        self.old_code_token_ids = []
        self.old_code_subtoken_ids = []

    def __str__(self):
        return self.add_line.comment_line

class Change:
    def __init__(self, commit, prev_commit, old_file, new_file, old_idx=None,
        old_span=None, new_idx=None, new_span=None, content=None):
        self.commit = commit
        self.prev_commit = prev_commit
        self.old_file = old_file
        self.new_file = new_file
        self.old_idx = old_idx
        self.old_span = old_span
        self.new_idx = new_idx
        self.new_span = new_span
        self.content = content

class Diff:
    def __init__(self, diff_id, change, old_comment, new_comment,
        diff_comment, old_code, new_code, diff_code):
        self.id = diff_id
        self.change = change
        self.old_comment = old_comment
        self.new_comment = new_comment
        self.diff_comment = diff_comment
        self.old_code = old_code
        self.new_code = new_code
        self.diff_code = diff_code

class ChangeType:
    RETURN = 1
    PARAM_MEDIUM = 2
    PARAM_SIMPLE = 3
    NOISE = 4
    NEGATIVE = 5
    OTHER = 6
    BOTH = 7

class TokenChangeType:
    ADD = 0
    REMOVE = 1
    KEEP = 2

class SpecialEditToken:
    ADD = 0
    REMOVE = 1
    KEEP = 2

class AssociationToken:
    def __init__(self, token, line_idx, pos_idx, is_associated):
        self.token = token
        self.line_idx = line_idx
        self.pos_idx = pos_idx
        self.is_associated = is_associated

class Association:
    def __init__(self, assoc_id, annotation_id, diff_id, comment_line, np_chunks, associated_code_tokens, candidate_code_tokens,
        full_code_representation, src, max_position, full_comment, full_code, diff_comment, diff_code):
        self.assoc_id = assoc_id
        self.annotation_id = annotation_id
        self.diff_id = diff_id
        self.comment_line = comment_line
        self.np_chunks = np_chunks
        self.associated_code_tokens = associated_code_tokens
        self.candidate_code_tokens = candidate_code_tokens
        self.full_code_representation = full_code_representation
        self.src = src
        self.max_position = max_position
        self.full_comment = full_comment
        self.full_code = full_code
        self.diff_comment = diff_comment
        self.diff_code = diff_code

    def to_json(self):
        result = dict()
        result['id'] = self.assoc_id
        result['annotation_id'] = self.annotation_id
        result['diff_id'] = self.diff_id  
        result['comment_line'] = self.comment_line
        result['np_chunks'] = self.np_chunks
        result['associated_code_tokens'] = self.associated_code_tokens
        result['candidate_code_tokens'] = self.candidate_code_tokens
        result['full_code_representation'] = self.full_code_representation
        result['src'] = self.src
        result['max_position'] = self.max_position
        result['full_comment'] = self.full_comment
        result['full_code'] = self.full_code
        result['diff_comment'] = self.diff_comment
        result['diff_code'] = self.diff_code
        return result

def to_association(result):
    # Hack -- field was added later
    if 'annotation_id' in result:
        annotation_id = result['annotation_id']
    else:
        annotation_id = -1
    
    return Association(result['id'],
        annotation_id,
        result['diff_id'],
        result['comment_line'] ,
        result['np_chunks'],
        result['associated_code_tokens'],
        result['candidate_code_tokens'],
        result['full_code_representation'],
        result['src'],
        result['max_position'],
        result['full_comment'],
        result['full_code'],
        result['diff_comment'],
        result['diff_code'])

def remove_html_tag(line):
    clean = re.compile('<.*?>')
    line = re.sub(clean, '', line)

    for tag in SPECIAL_TAGS:
        line = line.replace(tag, '')

    return line

def is_ascii(line):
    first_condition = all(ord(c) for c in line)
    second_condition = len(re.findall(r'[^\x00-\x7F]+', line)) > 0
    return first_condition and not second_condition

def is_alphanumeric(word):
    for w in word:
        if not w.isalnum() and w not in ['-', '_']:
            return False
    return True

def clean_comment_line(line, search_token):
    cleaned_line = remove_html_tag(line[1:]).replace('*', '').strip()
    cleaned_line = re.sub(' +', ' ', cleaned_line)
    idx = cleaned_line.index(search_token)
    cleaned_line = cleaned_line[idx+len(search_token):].strip()

    to_replace = re.findall(r'[^\s]+\.[^\s]+', cleaned_line)
    for r in to_replace:
        replace_string = r.split('.')[-1]
        cleaned_line = cleaned_line.replace(r, replace_string)

    return cleaned_line

def tokenize_comment_line(comment_line):
    return re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", comment_line.encode('utf-8').strip())

def subtokenize_comment_line(comment_line):
    tokens = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", comment_line.encode('utf-8').strip())
    subtokens = []
    for text in tokens:
        subtokens = subtokens + re.sub('([a-z0-9])([A-Z])', r'\1 \2', text).split()

    subtokens = [s.lower() for s in subtokens]
    return subtokens

def tokenize_code(code_snippet, check_alphanumeric=True):
    # Hack to handle bad examples with block comments within them
    code_lines = code_snippet.split('\n')
    i = 0
    while i < len(code_lines) and  '/**' not in code_lines[i]:
        i += 1

    code_snippet = '\n'.join(code_lines[0:i])
    tokens = list(javalang.tokenizer.tokenize(code_snippet))
    tokens = [tok.value for tok in tokens if tok.value not in JAVA_KEYWORDS]
    if check_alphanumeric:
        tokens = [tok for tok in tokens if is_alphanumeric(tok)]
    return tokens

def raw_tokenize_code(code_snippet):
    code_lines = code_snippet.split('\n')
    i = 0
    while i < len(code_lines) and  '/**' not in code_lines[i]:
        i += 1

    code_snippet = '\n'.join(code_lines[0:i])
    tokens = list(javalang.tokenizer.tokenize(code_snippet))
    token_values = [tok.value for tok in tokens]
    return tokens, token_values

def subtokenize_code(code_snippet, check_alphanumeric=True):
    # Hack to handle bad examples with block comments within them
    subtokens = []
    code_lines = code_snippet.split('\n')
    i = 0
    while i < len(code_lines) and  '/**' not in code_lines[i]:
        i += 1

    code_snippet = '\n'.join(code_lines[0:i])
    tokens = list(javalang.tokenizer.tokenize(code_snippet))
    tokens = [tok.value for tok in tokens]
    for whole_text in tokens:
        # Workaround: javalang library isn't properly tokenizing strings
        splits = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", whole_text.strip())
        for text in splits:
            text_subtokens = re.sub('([a-z0-9])([A-Z])', r'\1 \2', text).split()
            subtokens.extend(text_subtokens)
    if check_alphanumeric:
        subtokens = [s for s in subtokens if is_alphanumeric(s)]
    return subtokens

def tokenize_characters(line):
    new_line = line.strip().replace(' ', '\t')
    new_line = ' '.join(new_line).replace('\t', '<SPACE>')
    return new_line.split(' ')
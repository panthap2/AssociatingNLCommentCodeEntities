import json
import numpy as np
import os
import re
import sys
from sklearn.metrics import precision_recall_fscore_support

import data_utils as utils
from numpy.linalg import norm
import javalang

_UNK = '<UNK>'

AST_TYPES = ['<DUMMY>', 'CompilationUnit', ' Import', 'Documented', 'Declaration', 'TypeDeclaration', 'PackageDeclaration',
'ClassDeclaration', 'EnumDeclaration', 'InterfaceDeclaration', 'AnnotationDeclaration', 'Type',
'BasicType', 'ReferenceType', 'TypeArgument', 'TypeParameter', 'Annotation', 'ElementValuePair',
'ElementArrayValue', 'Member', 'MethodDeclaration', 'FieldDeclaration', 'ConstructorDeclaration', 'ConstantDeclaration',
'ArrayInitializer', 'VariableDeclaration', 'LocalVariableDeclaration', 'VariableDeclarator',
'FormalParameter', 'InferredFormalParameter', 'Statement', 'IfStatement', 'WhileStatement',
'DoStatement', 'ForStatement', 'AssertStatement', 'BreakStatement', 'ContinueStatement',
'ReturnStatement', 'ThrowStatement', 'SynchronizedStatement', 'TryStatement', 'SwitchStatement',
'BlockStatement', 'StatementExpression', 'TryResource', 'CatchClause', 'CatchClauseParameter',
'SwitchStatementCase', 'ForControl', 'EnhancedForControl', 'Expression', 'Assignment', 'TernaryExpression',
'BinaryOperation', 'Cast', 'MethodReference', 'LambdaExpression', 'Primary', 'Literal', 'This', 'MemberReference',
'Invocation', 'ExplicitConstructorInvocation', 'SuperConstructorInvocation', 'MethodInvocation',
'SuperMethodInvocation', 'SuperMemberReference', 'ArraySelector', 'ClassReference',
'VoidClassReference', 'Creator', 'ArrayCreator', 'ClassCreator', 'InnerClassCreator', 'EnumBody',
'EnumConstantDeclaration', 'AnnotationMethod']

class Counter:
    def __init__ (self):
        self.count = 0

    def generate(self):
        new_id = self.count
        self.count += 1
        return new_id

class Indexer(object):
    # Borrowed from Greg Durrett's CS395T: Structured Models for NLP Fall 2017 class
    # Bijection between objects and integers starting at 0. Useful for mapping
    # labels, features, etc. into coordinates of a vector space.
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in xrange(0, len(self))])

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        return self.index_of(object) != -1

    # Returns -1 if the object isn't present, index otherwise
    def index_of(self, object):
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    # Adds the object to the index if it isn't present, always returns a nonnegative index
    def get_index(self, object, add=True):
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]

class SimpleNode:
    '''AST utility class.'''
    def __init__(self, name, parent, id, depth=None):
        self.name = name
        self.parent = parent
        self.id = id
        
        if not depth == None:
            self.depth = depth
        else:
            self.depth = self.parent.depth + 1
        
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def format(self):
        print self.name
        print [c.name for c in self.children]
        print '---------------'
        for child in self.children:
            child.format()

    def get_children(self):
        return [c.name for c in self.children]

    def get_siblings(self):
        if not self.parent:
            return []
        
        siblings = []
        for child in self.parent.children:
            if child.id != self.id:
                siblings.append(child)
        
        return [s.name for s in siblings]

    def get_non_leaf_siblings(self):
        if not self.parent:
            return []
        
        siblings = []
        for child in self.parent.children:
            if child.id != self.id and not child.is_leaf():
                siblings.append(child)
        
        return [s.name for s in siblings]

    def get_grandparent(self):
        if self.parent and self.parent.parent:
            return self.parent.parent.name
        else:
            return None

    def get_depth(self):
        return self.depth

    def is_leaf(self):
        return len(self.children) == 0

    def get_dictionary_helper(self, curr_dict):
        curr_dict[self.name] = self
        for child in self.children:
            child.get_dictionary_helper(curr_dict)
        return curr_dict

    def get_dictionary(self):
        return self.get_dictionary_helper(dict())

def transform_tree(simple_tree, ast_tree, counter):
    '''AST-parsing utility function.'''
    if not ast_tree:
        return
    
    elif isinstance(ast_tree, (unicode, str, bool)):
        new_simple_tree = SimpleNode(str(ast_tree), simple_tree, counter.generate())
        simple_tree.add_child(new_simple_tree)

    elif isinstance(ast_tree, javalang.ast.Node):
        # TODO: What if root?
        new_simple_tree = SimpleNode(str(ast_tree), simple_tree, counter.generate())
        simple_tree.add_child(new_simple_tree)
        for child_ast in ast_tree.children:
            transform_tree(new_simple_tree, child_ast, counter)

    elif isinstance(ast_tree, (list, set, tuple)):
        for child_ast in ast_tree:
            transform_tree(simple_tree, child_ast, counter)
    else:
        print ast_tree
        print type(ast_tree)
        raise ValueError('Invalid ast node type')

def parse_tree(tokenized_code_snippet):
    '''Parses the AST corresponding to the given code snippet.'''
    counter = Counter()

    parser = javalang.parser.Parser(tokenized_code_snippet)
    ast_tree = parser.parse_member_declaration()
    simple_root = SimpleNode(str(ast_tree), None, counter.generate(), depth=0)
    transform_tree(simple_root, ast_tree, counter)
    nodes_by_name = simple_root.get_dictionary()

    return nodes_by_name, simple_root

def load_embeddings(parent_directory):
    '''Loads pretrained token, subtoken, and character embeddings.'''
    files = [f for f in os.listdir(parent_directory)]
    token_file = [f for f in files if 'return_line_tokens' in f][0]
    subtoken_file = [f for f in files if 'return_line_subtokens' in f][0]
    character_file = [f for f in files if 'return_line_character' in f][0]

    with open(os.path.join(parent_directory, token_file)) as json_file:
        token_embeddings = json.load(json_file)

    with open(os.path.join(parent_directory, subtoken_file)) as json_file:
        subtoken_embeddings = json.load(json_file)

    with open(os.path.join(parent_directory, character_file)) as json_file:
        character_embeddings = json.load(json_file)

    return token_embeddings, subtoken_embeddings, character_embeddings

def get_vector(inp, embeddings, use_lower=False):
    '''Computes average over word embeddings corresponding to tokens in inp.'''
    vectors = []

    for tok in inp:
        if use_lower:
            t = tok.lower()
        else:
            t = tok
        if t in embeddings:
            vectors.append(embeddings[t])
        else:
            vectors.append(embeddings[_UNK])

    vectors = np.asarray(vectors)
    avg = np.average(vectors, axis=0)
    return avg

def load_data(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    
    return [utils.to_association(d) for d in data]

def load_data_from_object(obj):
    return [utils.to_association(d) for d in obj]

def get_label_indexer():
    label_indexer = Indexer()
    label_indexer.get_index("True") # Associated
    label_indexer.get_index("False") # Not Associated
    return label_indexer

def set_features(association, embeddings, java_utils, java_utils_methods):
    '''Extracts features for a given example.'''
    token_embeddings, subtoken_embeddings, character_embeddings = embeddings
    np_tokens = utils.tokenize_comment_line(association.np_chunks[0])
    np_subtokens = utils.subtokenize_comment_line(association.np_chunks[0])
    np_characters = utils.tokenize_characters(' '.join(np_tokens))

    code_snippet = association.full_code
    code_lines = code_snippet.split('\n')
    code_raw_tokens, code_tokens = utils.raw_tokenize_code(code_snippet)

    comment_line = utils.clean_comment_line(association.comment_line, '@return')
    comment_tokens = utils.tokenize_comment_line(comment_line)
    comment_subtokens = utils.subtokenize_comment_line(comment_line)

    try:
        ast_nodes_by_name, root = parse_tree(code_raw_tokens)
        ast_names = ast_nodes_by_name.keys()
    except:
        print "Failed to parse:"
        print code_snippet
        print '--------------------'
        ast_nodes_by_name = dict()

    return_lines = []
    for line in code_lines:
        if 'return' in line:
            return_lines.append(line)

    features = []
    lower_np_tokens = [n.lower() for n in np_tokens]

    # Java ontology features
    for j_category, j_values in java_utils.iteritems():
        contains = False
        lowered_j_values = [v.lower() for v in j_values]
        for n in lower_np_tokens:
            if n in lowered_j_values:
                contains = True
        if contains:
            features.append(1)
        else:
            features.append(0)

    # Noun phrase context
    np_nl_embedding = get_vector(np_tokens, token_embeddings['nl'], use_lower=True)
    np_code_tok_embedding = get_vector(np_tokens, token_embeddings['code'])
    np_code_subtok_embedding = get_vector(np_subtokens, subtoken_embeddings['code'], use_lower=True)
    
    np_char_embedding = get_vector(np_characters, character_embeddings['nl'], use_lower=True)

    np_context_vector = np.concatenate((np_nl_embedding, np_char_embedding))
    features = features + np_context_vector.tolist()

    # Comment context
    comment_nl_embedding = get_vector(comment_tokens, token_embeddings['nl'], use_lower=True)
    comment_code_tok_embedding = get_vector(comment_tokens, token_embeddings['code'])
    comment_code_subtok_embedding = get_vector(comment_subtokens, subtoken_embeddings['code'], use_lower=True)
    
    comment_characters = utils.tokenize_characters(' '.join(comment_tokens))
    comment_char_embedding = get_vector(comment_characters, character_embeddings['nl'], use_lower=True)
    comment_context_vector = np.concatenate((comment_nl_embedding, comment_char_embedding))
    
    features = features + comment_context_vector.tolist()

    for candidate_code_token in association.full_code_representation:
        candidate_features = features + []
        candidate_token = candidate_code_token['token']
        subtokens = utils.subtokenize_comment_line(candidate_token)
        characters = utils.tokenize_characters(candidate_token)
        containing_line = code_lines[candidate_code_token['line_idx']-1]

        line_tokens = candidate_code_token['tokenized_line']
        line_subtokens = utils.subtokenize_comment_line(' '.join(line_tokens))
        line_characters = utils.tokenize_characters(' '.join(line_tokens))

        tok_embedding = get_vector([candidate_token], token_embeddings['code'])

        # Although it isn't a comment, can use the same function
        subtok_embedding = get_vector(subtokens, subtoken_embeddings['code'], use_lower=True)
        char_embedding = get_vector(characters, character_embeddings['code'], use_lower=True)

        # Code token features
        candidate_features = np.concatenate((tok_embedding, subtok_embedding, char_embedding)).tolist() + candidate_features
        
        # Line-level features
        line_token_embedding = get_vector(line_tokens, token_embeddings['code'])
        line_subtoken_embedding = get_vector(line_subtokens, subtoken_embeddings['code'], use_lower=True)
        line_character_embedding = get_vector(line_characters, character_embeddings['code'], use_lower=True)

        candidate_features = np.concatenate((line_token_embedding, line_subtoken_embedding,
            line_character_embedding)).tolist() + candidate_features

        sim_line_np_token = cosine_similarity(line_token_embedding, np_code_tok_embedding)
        sim_line_np_subtoken = cosine_similarity(line_subtoken_embedding, np_code_subtok_embedding)
        sim_line_np_character = cosine_similarity(line_character_embedding, get_vector(np_characters,
            character_embeddings['code'], use_lower=True))

        candidate_features.append(sim_line_np_token)
        candidate_features.append(sim_line_np_subtoken)
        candidate_features.append(sim_line_np_character)

        # Java ontology features
        for j_category, j_values in java_utils.iteritems():
            if candidate_token in j_values:
                candidate_features.append(1)
            else:
                candidate_features.append(0)

        if candidate_token in java_utils_methods:
            candidate_features.append(1)
        else:
            candidate_features.append(0)

        subtoken_overlap = 0
        return_line = 0
        for sub in subtokens:
            if sub in np_subtokens:
                subtoken_overlap = 1
        
        if 'return' in containing_line:
            return_line = 1

        in_return_line = 0
        for line in return_lines:
            if candidate_token in line:
                in_return_line = 1
                break

        # Surface features
        candidate_features.append(subtoken_overlap)
        candidate_features.append(return_line)
        candidate_features.append(in_return_line)

        # Java features
        for j_type in utils.JAVA_TYPES:
            if candidate_code_token['token'] == j_type:
                candidate_features.append(1)
            else:
                candidate_features.append(0)

        if candidate_code_token['token'] in utils.JAVA_TYPES:
            candidate_features.append(1)
        else:
            candidate_features.append(0)

        for j_type in utils.JAVA_TYPES:
            if j_type in line_tokens:
                candidate_features.append(1)
            else:
                candidate_features.append(0)

        idx = line_tokens.index(candidate_token)
        if idx == 0:
            prev_token = _UNK
        else:
            prev_token = line_tokens[idx-1]

        if idx == len(line_tokens)-1:
            next_token = _UNK
        else:
            next_token = line_tokens[idx+1]

        # Java features of neighborign code tokens
        for j_type in utils.JAVA_TYPES:
            if prev_token == j_type:
                candidate_features.append(1)
            else:
                candidate_features.append(0)

        for j_type in utils.JAVA_TYPES:
            if next_token == j_type:
                candidate_features.append(1)
            else:
                candidate_features.append(0)

        for j_keyword in utils.JAVA_KEYWORDS:
            if j_keyword in line_tokens:
                candidate_features.append(1)
            else:
                candidate_features.append(0)

        for j_keyword in utils.JAVA_KEYWORDS:
            if prev_token == j_keyword:
                candidate_features.append(1)
            else:
                candidate_features.append(0)

        for j_keyword in utils.JAVA_KEYWORDS:
            if next_token == j_keyword:
                candidate_features.append(1)
            else:
                candidate_features.append(0)

        # Code embeddings of previous and next tokens
        prev_token_embedding = get_vector([prev_token], token_embeddings['code']).tolist()
        next_token_embedding = get_vector([next_token], token_embeddings['code']).tolist()
        candidate_features = candidate_features + prev_token_embedding
        candidate_features = candidate_features + next_token_embedding

        # Cosine similarity features
        candidate_features.append(get_code_token_cosine_similarity(association, candidate_code_token, token_embeddings))
        candidate_features.append(get_code_subtoken_cosine_similarity(association, candidate_code_token, subtoken_embeddings))
        candidate_features.append(get_code_character_cosine_similarity(association, candidate_code_token, character_embeddings))

        # AST features
        if candidate_token in ast_nodes_by_name:
            parent = ast_nodes_by_name[candidate_token].parent.name
            siblings = ast_nodes_by_name[candidate_token].get_siblings()
            non_leaf_siblings = ast_nodes_by_name[candidate_token].get_non_leaf_siblings()
            grandparent = ast_nodes_by_name[candidate_token].get_grandparent()
            depth = ast_nodes_by_name[candidate_token].get_depth()
        else:
            parent = '<DUMMY>'
            siblings = []
            non_leaf_siblings = []
            grandparent = '<DUMMY>'
            depth = -1

        for ast_type in AST_TYPES:
            if str(parent) == ast_type:
                candidate_features.append(1)
            else:
                candidate_features.append(0)

        for ast_type in AST_TYPES:
            if str(grandparent) == ast_type:
                candidate_features.append(1)
            else:
                candidate_features.append(0)
        
        candidate_code_token['features'] = candidate_features

def get_code_token_cosine_similarity(association, candidate, token_embeddings):
    '''Computes the cosine similarity between the token-level embedding vector corresponding
    to the NP for the given association and the given candidate code token.'''
    np_tokens = utils.tokenize_comment_line(association.np_chunks[0])
    np_token_vector = get_vector(np_tokens, token_embeddings['code'])
    candidate_vector = get_vector([candidate['token']], token_embeddings['code'])
    return cosine_similarity(np_token_vector, candidate_vector)

def get_code_subtoken_cosine_similarity(association, candidate, subtoken_embeddings):
    '''Computes the cosine similarity between the subtoken-level embedding vector corresponding
    to the NP for the given association and the given candidate code token.'''
    np_subtokens = utils.subtokenize_comment_line(association.np_chunks[0])
    np_subtoken_vector = get_vector(np_subtokens, subtoken_embeddings['code'], use_lower=True)
    candidate_subtokens = utils.subtokenize_comment_line(candidate['token'])
    candidate_subtoken_vector = get_vector(candidate_subtokens, subtoken_embeddings['code'], use_lower=True)
    return cosine_similarity(np_subtoken_vector, candidate_subtoken_vector)

def get_code_character_cosine_similarity(association, candidate, character_embeddings):
    '''Computes the cosine similarity between the character-level embedding vector corresponding
    to the NP for the given association and the given candidate code token.'''
    np_tokens = utils.tokenize_comment_line(association.np_chunks[0])
    np_characters = utils.tokenize_characters(' '.join(np_tokens))
    np_character_vector = get_vector(np_characters, character_embeddings['code'], use_lower=True)
    candidate_characters = utils.tokenize_characters(candidate['token'])
    candidate_character_vector = get_vector(candidate_characters, character_embeddings['code'], use_lower=True)
    return cosine_similarity(np_character_vector, candidate_character_vector)
    
def cosine_similarity(a, b):
    '''Computes cosine similarity between two vectors.'''
    return abs(np.dot(a, b)/(norm(a)*norm(b)))

def get_batch_cutoffs(examples, batch_size):
    '''Identifies batch cutoffs based on the number of examples and a desired batch size.
    Any leftover items are packed into a single batch, which will be slightly less than batch
    size.'''
    batch_cutoffs = []
    
    if len(examples) < batch_size:
        batch_cutoffs.append((0, len(examples), len(examples)))
        return batch_cutoffs

    count = 0
    while count + batch_size < len(examples):
        batch_cutoffs.append((count, count + batch_size, batch_size))
        count += batch_size

    if count < len(examples):
        batch_range = range(count, len(examples))
        batch_cutoffs.append((count, len(examples), len(batch_range)))
    return batch_cutoffs

def average(vals):
    '''Utility function that computes the average of a list of values.'''
    return sum(vals)/float(len(vals))

def compute_candidate_metrics(candidates):
    '''Computes metrics at the candidate-level.'''
    predictions = []
    gold = []
    total = 0
    correct = 0
    for candidate in candidates:
        if 'pseudo_class' in candidate and candidate['pseudo_class'] == 'JAVA':
            continue
        if candidate['is_associated']:
            gold_label = 1
        else:
            gold_label = 0

        predicted_label = candidate['prediction']
        if predicted_label not in [0,1]:
            raise ValueError('Invalid prediction')

        # For micro calculations
        total += 1
        predictions.append(predicted_label)
        gold.append(gold_label)
        if gold_label == predicted_label:
            correct +=1

    precision, recall, f1, _ = precision_recall_fscore_support(np.asarray(gold),
        np.asarray(predictions), average='binary', pos_label=1)
    accuracy = correct/float(total)

    return f1

def compute_metrics(associations):
    '''Computes metrics at the example-level.'''
    predictions = []
    gold = []
    total = 0
    correct = 0

    macro_precision_scores = []
    macro_recall_scores = []
    macro_f1_scores = []
    macro_accuracy_scores = []

    for association in associations:
        example_predictions = []
        example_gold = []
        example_correct = 0
        example_total = 0
        example_true_positives = 0
        example_false_negatives = 0
        example_false_positives = 0
        tokens = []

        candidates = association.full_code_representation
        for candidate in candidates:
            if 'pseudo_class' in candidate and candidate['pseudo_class'] == 'JAVA':
                continue
            if candidate['is_associated']:
                gold_label = 1
            else:
                gold_label = 0

            predicted_label = candidate['prediction']
            tokens.append(candidate['token'])
            if predicted_label not in [0,1]:
                raise ValueError('Invalid prediction')

            # For micro calculations
            total += 1
            predictions.append(predicted_label)
            gold.append(gold_label)
            if gold_label == predicted_label:
                correct +=1

            # For macro calculations
            example_total += 1
            example_predictions.append(predicted_label)
            example_gold.append(gold_label)
            if gold_label == predicted_label:
                example_correct += 1
        
        for k in range(len(example_predictions)):
            print('{}: P - {}, G - {}'.format(tokens[k], example_predictions[k], example_gold[k]))
        print('*************************')
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(np.asarray(example_gold),
            np.asarray(example_predictions), average='binary', pos_label=1)
        macro_accuracy = example_correct/float(example_total)
        macro_precision_scores.append(macro_precision)
        macro_recall_scores.append(macro_recall)
        macro_f1_scores.append(macro_f1)
        macro_accuracy_scores.append(macro_accuracy)

    precision, recall, f1, _ = precision_recall_fscore_support(np.asarray(gold), np.asarray(predictions), average='binary', pos_label=1)
    accuracy = correct/float(total)

    print("Micro Precision: " + str(precision))
    print("Micro Recall: " + str(recall))
    print("Micro F1: " + str(f1))
    print '\n'
    
    print('Macro Precision: ' + str(average(macro_precision_scores)))
    print('Macro Recall: ' + str(average(macro_recall_scores)))
    print('Macro F1: ' + str(average(macro_f1_scores)))

    return precision, recall, f1, accuracy, macro_precision, macro_recall, macro_f1, macro_accuracy

def get_f1(gold, predictions_list, sampled_indices):
    '''Computes F1 score.'''
    f1_scores = []
    sampled_gold = [gold[i] for i in sampled_indices]

    for predictions in predictions_list:
        sampled_predictions = [predictions[i] for i in sampled_indices]
        precision, recall, f1, _ = precision_recall_fscore_support(np.asarray(sampled_gold),
            np.asarray(sampled_predictions), average='binary', pos_label=1)
        f1_scores.append(f1)
    
    return sum(f1_scores)/float(len(f1_scores))

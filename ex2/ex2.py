import sys
import math

# Constants defined in the exercise description
VOCABULARY_SIZE = 300000  # The assumed language vocabulary size V


def parse_arguments():
    """
    Parses the command-line arguments and returns them.
    Exits if the number of arguments is incorrect.

    Returns:
        tuple: (dev_set_filename, test_set_filename, input_word, output_filename)
    """
    # The program expects 4 arguments in this exact order
    if len(sys.argv) != 5:
        print("Error: Expected 4 command-line arguments.")
        print("Usage: python ex2.py <development_set_filename> <test_set_filename> <INPUT-WORD> <output-filename>")
        sys.exit(1)

    dev_set_filename = sys.argv[1]
    test_set_filename = sys.argv[2]
    input_word = sys.argv[3]
    output_filename = sys.argv[4]

    return dev_set_filename, test_set_filename, input_word, output_filename


def read_and_tokenize_development_set(filename):
    """
    Reads the development set file, processes the text,
    and returns the sequence of events (tokens) and the total count.
    The input is assumed to have already had the specific punctuation removed and in lower case,
    splitting done by whitespace

    Args:
        filename (str): Path to the development set file (e.g., develop.txt).

    Returns:
        list of tokens (token_list)
    """
    token_list = []

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            header = True
            for line_number, line in enumerate(f):
                # Tokenization: split by white spaces
                line_tokens = line.strip().split()
                if not line_tokens:
                    continue  # Skip empty lines
                # The input file contains 2 lines per article - header and article text.
                if header:
                    header = False
                    continue
                header = True #alternate between header and text
                token_list.extend(line_tokens)
                #token_list = token_list + line_tokens
    except FileNotFoundError:
        print(f"Error: Development set file not found at {filename}")
        sys.exit(1)

    return token_list


def init(f, dev_file, test_file, input_word, output_file):
    """
    Writes the initialization and preprocessing outputs (Output1 - Output6)
    to the open file object.
    """
    # Header line
    f.write("#Students\tEli Kazum\t038298857\tKaren Fuchs\t302345624\n")

    # SECTION 1: INIT
    f.write(f"#Output1\t{dev_file}\n")
    f.write(f"#Output2\t{test_file}\n")
    f.write(f"#Output3\t{input_word}\n")
    f.write(f"#Output4\t{output_file}\n")
    f.write(f"#Output5\t{VOCABULARY_SIZE}\n")

    # P_uniform = 1 / V
    p_uniform = 1.0 / VOCABULARY_SIZE
    # (f) Output6: Puniform (Event = INPUT_WORD)
    f.write(f"#Output6\t{p_uniform}\n")


def development_set_preprocessing(f, token_list):
    """
    Writes preprocessing outputs for the development set (Output7).
    """
    # (a) Output7: total number of events in the development set
    num_of_events = len(token_list)
    f.write(f"#Output7\t{num_of_events}\n")


def lidstone_model_training(f, token_list, input_word):
    """
    Implements Lidstone model training section:
    Splits the development set into training and validation sets (90%/10%)

    Args:
        f (file object): The open file handle to write to.
        token_list (list): The sequence of all events S from the development set.

    Returns:
        tuple: (training_set, validation_set)
        :param token_list: list of events in file
        :param f: output file handle
        :param input_word: the INPUT_WORD from command line
    """
    num_events = len(token_list)  # Total number of events |S|

    # Calculate the size of the training set (90% of |S|)
    training_size = int(round(0.9 * num_events))
    # Split the set
    training_set = token_list[:training_size]
    validation_set = token_list[training_size:]
    validation_size = len(validation_set)

    # (a) Output8: number of events in the validation set
    f.write(f"#Output8\t{validation_size}\n")
    # (b) Output9: number of events in the training set
    f.write(f"#Output9\t{training_size}\n")

    # (c) Output10: number of different events in the training set (observed vocabulary)
    observed_vocabulary = len(set(training_set))
    f.write(f"#Output10\t{observed_vocabulary}\n")

    # (d) Output11: number of times INPUT_WORD appears in the training set
    input_word_count = training_set.count(input_word)
    f.write(f"#Output11\t{input_word_count}\n")

    return training_set, validation_set


def evaluate_lindstone_model_preplexity(training_set, validation_set, lambda_val):
    # Precompute counts
    from collections import Counter
    counts = Counter(training_set)
    N = len(training_set)

    log_prob_sum = 0.0

    for event in validation_set:
        # Lidstone: (c(x)+λ) / (N + λ|X|)
        p = (counts[event] + lambda_val) / (N + lambda_val * VOCABULARY_SIZE)
        log_prob_sum += math.log2(p)

    n = len(validation_set)
    return 2 ** (-log_prob_sum / n)


def debug_lindstone_model(f, training_set, validation_set, lambda_val):
    """
    making sure p(x∗)n0 + X p(x) = 1
    """
    from collections import Counter

    counts = Counter(training_set)
    N = len(training_set)
    observed_vocab = len(counts)
    unseen_vocab = max(VOCABULARY_SIZE - observed_vocab, 0)

    denom = N + lambda_val * VOCABULARY_SIZE

    # Probability mass over seen words
    prob_seen = sum((count + lambda_val) / denom for count in counts.values())
    # Probability mass assigned uniformly to unseen words
    prob_unseen_total = unseen_vocab * (lambda_val / denom)

    total_prob = prob_seen + prob_unseen_total
    diff = abs(1.0 - total_prob)
    
    return diff == 0


def choose_best_lambda(training_set, validation_set):
    """
    Chooses the best lambda value for the Lidstone model.
    Tests lambda values from 0 to 2 with step 0.01.
    """
    best_lambda = 0.01
    best_perplexity = float('inf')
    
    # Grid search: loop over lambda values from 0 to 2 with step 0.01
    # This gives us 201 values: 0, 0.01, 0.02, ..., 1.99, 2.0
    for i in range(0, 200):  # 0 to 200 inclusive = 201 values
        lambda_val = round(i * 0.01, 2) + 0.01
        perplexity = evaluate_lindstone_model_preplexity(training_set, validation_set, lambda_val)
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_lambda = lambda_val
    return best_lambda


def lidstone_model_evaluation(f, training_set, validation_set, best_lambda, input_word):
    """
    Evaluates the model with the given lambda value.
    """
    
    # output 12 - MLE of input word without smoothing P_MLE(x) = (c(x)) / N
    mle_input_word = training_set.count(input_word) / len(training_set) 
    f.write(f"#Output12\t{mle_input_word}\n")
    
    # output 13 - MLE of unseen word based on training without smoothing P_MLE(x) = 0 / N
    mle_unseen_word = 0 / len(training_set)
    f.write(f"#Output13\t{mle_unseen_word}\n")
    
    # output 14 - Lidstone model evaluation of input word (P(input_word)) given lambda=0.1: P_Lidstone(x) = (c(x)+λ) / (|S| + λ|X|)
    lidstone_model_evaluation_input_word = (training_set.count(input_word) + 0.1) / (len(training_set) + 0.1 * VOCABULARY_SIZE)
    f.write(f"#Output14\t{lidstone_model_evaluation_input_word}\n")
    
    # output 15 - Lidstone model evaluation of unseen word (P(unseen_word)) given lambda=0.1: P_Lidstone(x) = λ / (|S| + λ|X|)
    lidstone_model_evaluation_unseen_word = 0.1 / (len(training_set) + 0.1 * VOCABULARY_SIZE)
    f.write(f"#Output15\t{lidstone_model_evaluation_unseen_word}\n")
    
    # output 16 - The perplexity of the Lidstone model on input word given lambda=0.01
    perplexity = evaluate_lindstone_model_preplexity(training_set, validation_set, 0.01)
    f.write(f"#Output16\t{perplexity}\n")
    
    # output 17 - The perplexity of the Lidstone model on unseen word given lambda=0.1 
    perplexity = evaluate_lindstone_model_preplexity(training_set, validation_set, 0.1)
    f.write(f"#Output17\t{perplexity}\n")

    # output 18 - The perplexity of the Lidstone model on unseen word given lambda=1 
    perplexity = evaluate_lindstone_model_preplexity(training_set, validation_set, 1)
    f.write(f"#Output18\t{perplexity}\n")
    
    # output 19 - The value of best lambda to minimize preplexity on validation set
    f.write(f"#Output19\t{best_lambda}\n")
    
    # output 20 - The perplexity of the Lidstone model on input word given best lambda
    perplexity = evaluate_lindstone_model_preplexity(training_set, validation_set, best_lambda)
    f.write(f"#Output20\t{perplexity}\n")
    
    # Section 5: debug Lindstone model code by making sure p(x∗)n0 + X p(x) = 1
    debug_lindstone_model(f, training_set, validation_set, best_lambda)
    debug_lindstone_model(f, training_set, validation_set, 0.1)
    debug_lindstone_model(f, training_set, validation_set, 1)
 
    
def debug_held_out_model(f, held_out_model, training_set, held_out_set):
    """
    making sure p(x∗)n0 + X p(x) = 1
    """
    # held_out_model is expected to be a mapping: word -> P_heldout(word)
    all_words = set(training_set) | set(held_out_set)
    total_prob = sum(held_out_model.get(word, 0.0) for word in all_words)
    diff = abs(1.0 - total_prob)
    f.write(f"#DebugHeldOut\t{total_prob}\t{diff}\n")
    return diff == 0
    
def compute_held_out_model(training_set, held_out_set):
    """
    Computes the held-out model probabilities for all words.
    Formula: P_heldout(x) = t_r / (N_H * n_r)
    where:
    - r = c_T(x) = count of word x in training set
    - t_r = sum of c_H(y) for all words y where c_T(y) = r
    - N_H = total number of events in held-out set
    - n_r = number of distinct words that appeared exactly r times in training
    
    Args:
        training_set: list of words in training set
        held_out_set: list of words in held-out set
    
    Returns:
        dict: Dictionary mapping words to their held-out probabilities
    """
    from collections import Counter
    
    # Count occurrences of each word in training set (c_T)
    training_counts = Counter(training_set)
    
    # Count occurrences of each word in held-out set (c_H)
    held_out_counts = Counter(held_out_set)
    
    # N_H = total number of events in held-out set
    N_H = len(held_out_set)
    
    # Get all unique words (from both training and held-out sets)
    all_words = set(training_set) | set(held_out_set)
    
    # Precompute t_r and n_r for each frequency r
    # Group words by their frequency in training
    freq_to_words = {}  # r -> list of words with frequency r
    for word in all_words:
        r = training_counts[word]
        if r not in freq_to_words:
            freq_to_words[r] = []
        freq_to_words[r].append(word)
    
    # Compute t_r and n_r for each frequency r
    freq_to_prob = {}  # r -> probability for words with frequency r
    
    for r, words_with_r in freq_to_words.items():
        t_r = 0
        n_r = 0
        if r == 0:
            # Count all occurrences of unseen words in held-out set
            t_r = sum(held_out_counts[word] for word in words_with_r if word in held_out_counts)
            # n_0 should be the total number of unseen word types
            n_r = VOCABULARY_SIZE - len(training_counts)
        else:
            # For seen words (r > 0): count words that appeared r times in training
            # t_r = sum of c_H(y) for all words y where c_T(y) = r
            # n_r = number of distinct words that appeared r times in training
            for word, count in training_counts.items():
                if count == r:
                    t_r += held_out_counts[word]  # c_H(word), defaults to 0 if not in held_out_counts
                    n_r += 1
        
        # Compute probability for this frequency r
        if n_r == 0:
            prob = 0.0
        else:
            prob = t_r / (N_H * n_r)
        
        freq_to_prob[r] = prob
    
    # Build dictionary mapping each word to its probability
    probabilities = {}
    for word in all_words:
        r = training_counts[word]
        probabilities[word] = freq_to_prob[r]
    
    return probabilities, freq_to_prob    


def held_out_model_training(f, development_set, input_word):
    
    """
    Trains the held-out model on the development set.
    """
    # split the development set into training and validation sets 1/2 - 1/2
    training_size = int(round(0.5 * len(development_set)))
    training_set = development_set[:training_size]
    held_out_set = development_set[training_size:]
    
    # output 21 - number of events in the training set
    f.write(f"#Output21\t{len(training_set)}\n")
    
    # output 22 - number of events in the held-out set
    f.write(f"#Output22\t{len(held_out_set)}\n")
    
    # Compute held-out model probabilities for all words
    held_out_probabilities, freq_to_prob = compute_held_out_model(training_set, held_out_set)
    
    # output 23 - p(input_word) of held out model
    held_out_model_input_word = held_out_probabilities.get(input_word, 0.0)
    f.write(f"#Output23\t{held_out_model_input_word}\n")
    
    # output 24 - p(unseen_word) of held out model
    held_out_model_unseen_word = freq_to_prob.get(0, 0.0)
    f.write(f"#Output24\t{held_out_model_unseen_word}\n")
    
    return held_out_probabilities, training_set, held_out_set


def evaluate_held_out_model_preplexity(held_out_model, test_set):
    """
    Compute hold out model preplexity
    args:
    held_out_model (dict): Dictionary mapping words to their held-out probabilities
    """
    log_prob_sum = 0.0
    epsilon = 1e-12  # avoid log(0) for unseen or zero-probability words

    for event in test_set:
        p = held_out_model.get(event, 0.0)
        if p <= 0.0:
            p = epsilon
        log_prob_sum += math.log2(p)

    n = len(test_set)
    return 2 ** (-log_prob_sum / n)


def evaluation_on_test_set(f, training_set_lindstone, held_out_model, test_set, input_word, best_lambda):
    """
    """
    # Output25: total number of events in the test set
    events_count_test = len(test_set)
    f.write(f"#Output25\t{events_count_test}\n")
    
    # Output26: The perplexity of the test set according to the Lidstone model with the λ that you chose during development
    test_preplexity = evaluate_lindstone_model_preplexity(training_set_lindstone, test_set, best_lambda)
    f.write(f"#Output26\t{test_preplexity}\n")
    
    # Output27: The perplexity of the test set according to your held-out model
    test_held_out_preplexity = evaluate_held_out_model_preplexity(held_out_model, test_set)
    f.write(f"#Output27\t{test_held_out_preplexity}\n")
    
    # Output28: If your Lidstone model is a better language model for the test set than the held-out model then output the string ’L’, otherwise output ’H’
    if test_preplexity < test_held_out_preplexity:
        f.write("#Output28\tL\n")
    else:
        f.write("#Output28\tH\n")
        
    
    
    

def main(dev_set_filename, test_set_filename, input_word, output_filename):
    """
    Main function for Exercise 2.
    """
    # Parse Input
    #dev_set_filename, test_set_filename, input_word, output_filename = parse_arguments()

    # Development set preprocessing
    # token_list represents sequence of events
    token_list = read_and_tokenize_development_set(dev_set_filename)

    try:
        with open(output_filename, 'w') as f:
            # Calculate and write output section 1
            init(f, dev_set_filename, test_set_filename, input_word, output_filename)

            # Section 2: Development set preprocessing
            development_set_preprocessing(f, token_list)

            # Section 3: Lidstone model training 
            training_set_lindstone, validation_set = lidstone_model_training(f, token_list, input_word)
            
            best_lambda = choose_best_lambda(training_set_lindstone, validation_set) 
            
            lidstone_model_evaluation(f, training_set_lindstone, validation_set, best_lambda, input_word) 
            
            # Section 4:  Held-out model training
            development_set = training_set_lindstone + validation_set
            held_out_model, training_set_heldout, held_out_set = held_out_model_training(f, development_set, input_word)
            
            # Section 5: debug Held-out model code by making sure p(x∗)n0 + X p(x) = 1
            debug_held_out_model(f, held_out_model, training_set_heldout, held_out_set)
            
            # Section 6: evaluation on test set
            test_set = read_and_tokenize_development_set(test_set_filename)
            evaluation_on_test_set(f, training_set_lindstone, held_out_model, test_set, input_word, best_lambda)

    except IOError as e:
        print(f"Error writing to file {output_filename}: {e}")


if __name__ == "__main__":
    main("develop.txt", "test.txt", "a", "output.txt")
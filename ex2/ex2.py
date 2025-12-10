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


def read_and_tokenize_data(filename):
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
                # The input file contains 2 lines per article - header and article text
                if header:
                    header = False
                    continue
                header = True  # Alternate between header and text
                token_list.extend(line_tokens)
    except FileNotFoundError:
        print(f"Error: Set file not found at {filename}")
        sys.exit(1)

    return token_list


def init(f, dev_file, test_file, input_word, output_file):
    """
    Writes the initialization and preprocessing outputs (Output1 - Output6) to the file.
    
    Args:
        f: File handle to write output to
        dev_file (str): Path to the development set file
        test_file (str): Path to the test set file
        input_word (str): The input word from command line arguments
        output_file (str): Path to the output file
        
    Outputs:
        Header line with student information
        #Output1: Development set filename
        #Output2: Test set filename
        #Output3: Input word
        #Output4: Output filename
        #Output5: Vocabulary size (V)
        #Output6: Uniform probability P_uniform = 1/V for the input word
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
    # Output6: P_uniform (Event = INPUT_WORD)
    f.write(f"#Output6\t{p_uniform}\n")


def development_set_preprocessing(f, token_list):
    """
    Writes preprocessing outputs for the development set.
    
    Args:
        f: File handle to write output to
        token_list (list): List of tokens/events from the development set
        
    Outputs:
        #Output7: Total number of events in the development set
    """
    # Output7: total number of events in the development set
    num_of_events = len(token_list)
    f.write(f"#Output7\t{num_of_events}\n")

def model_evaluation_on_test_set(f, test_token_list):
    """
    Writes model evaluation outputs for the test set (Output26-29).
    """
    # (a) Output26: total number of events in the test set
    num_of_events = len(test_token_list)
    f.write(f"#Output25\t{num_of_events}\n")



def model_perplexity(token_list, training_counts, lambda_val, n_training):
    """
    Compute perplexity of token_list under Lidstone model with given lambda.
    """
    n = len(token_list)
    if n == 0:
        return float('inf')
    log_prob_sum = 0.0
    denom = n_training + lambda_val * VOCABULARY_SIZE
    for w in token_list:
        cw = training_counts.get(w, 0)
        prob = (cw + lambda_val) / denom
        # safety: prob should be > 0 due to smoothing
        log_prob_sum += math.log(prob)
    return math.exp(- (log_prob_sum / n))

def lidstone_model_training(f, token_list, input_word):
    """
    Implements Lidstone model training section by splitting the development set into training and validation sets.
    
    Splits the development set into training (90%) and validation (10%) sets, then outputs
    statistics about the training set including vocabulary size and input word frequency.

    Args:
        f: File handle to write output to
        token_list (list): The sequence of all events S from the development set
        input_word (str): The input word to count in the training set

    Returns:
        tuple: (training_set, validation_set) - Two lists containing the split tokens
        
    Outputs:
        #Output8: Number of events in the validation set
        #Output9: Number of events in the training set
        #Output10: Number of different events in the training set (observed vocabulary)
        #Output11: Number of times INPUT_WORD appears in the training set
    """
    num_events = len(token_list)  # Total number of events |S|

    # Calculate the size of the training set (90% of |S|)
    training_size = int(round(0.9 * num_events))
    # Split the set
    training_set = token_list[:training_size]
    validation_set = token_list[training_size:]
    validation_size = len(validation_set)

    # Output8: number of events in the validation set
    f.write(f"#Output8\t{validation_size}\n")
    
    # Output9: number of events in the training set
    f.write(f"#Output9\t{training_size}\n")

    # Output10: number of different events in the training set (observed vocabulary)
    observed_vocabulary = len(set(training_set))
    f.write(f"#Output10\t{observed_vocabulary}\n")

    # Output11: number of times INPUT_WORD appears in the training set
    input_word_count = training_set.count(input_word)
    f.write(f"#Output11\t{input_word_count}\n")

    return training_set, validation_set


def evaluate_lindstone_model_preplexity(training_set, validation_set, lambda_val):
    """
    Evaluates the perplexity of the Lidstone model on a validation set.
    
    Computes perplexity using the Lidstone smoothing formula:
    P_Lidstone(x) = (c(x) + λ) / (N + λ|X|)
    where c(x) is the count of x in training, N is training set size, and |X| is vocabulary size.
    
    Args:
        training_set (list): List of tokens used to train the model
        validation_set (list): List of tokens to evaluate perplexity on
        lambda_val (float): The smoothing parameter λ (lambda) for Lidstone smoothing
        
    Returns:
        float: Perplexity of the validation set under the Lidstone model
    """
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
    Debugs the Lidstone model by verifying that probability mass sums to 1.
    
    Checks that the sum of probabilities over all seen words plus the probability
    mass assigned to unseen words equals 1: p(x*)n0 + Σp(x) = 1
    
    Args:
        f: File handle to write debug output to
        training_set (list): List of tokens used to train the model
        validation_set (list): List of tokens in validation set (not used in calculation)
        lambda_val (float): The smoothing parameter λ (lambda) for Lidstone smoothing
        
    Returns:
        bool: True if the probability sum is within tolerance (1e-13) of 1, False otherwise
        
    Outputs:
        #DebugLindstone: Total probability and difference from 1.0
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
    f.write(f"#DebugLindstone\t{total_prob}\t{diff}\n")
    return diff < 1e-13


def choose_best_lambda(training_set, validation_set):
    """
    Chooses the best lambda (λ) value for the Lidstone model using grid search.
    
    Tests lambda values from 0.01 to 2.0 with step 0.01 and selects the value
    that minimizes perplexity on the validation set.
    
    Args:
        training_set (list): List of tokens used to train the model
        validation_set (list): List of tokens used to validate and select best lambda
        
    Returns:
        float: The lambda value (0.01 to 2.0) that minimizes validation perplexity
    """
    best_lambda = 0.01
    best_perplexity = float('inf')
    
    # Grid search: loop over lambda values from 0.01 to 2.0 with step 0.01
    # This gives us 200 values: 0.01, 0.02, ..., 1.99, 2.0
    for i in range(0, 200):  # 0 to 199 inclusive = 200 values
        lambda_val = round(i * 0.01, 2) + 0.01
        perplexity = evaluate_lindstone_model_preplexity(training_set, validation_set, lambda_val)
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_lambda = lambda_val
    return round(best_lambda, 2)


def lidstone_model_evaluation(f, training_set, validation_set, best_lambda, input_word):
    """
    Evaluates the Lidstone model and writes all evaluation outputs (Output12 - Output20).
    
    Computes and outputs MLE probabilities, Lidstone probabilities for different lambda values,
    perplexities for various lambda values, and the best lambda with its perplexity.
    Also performs debug checks to verify probability mass sums to 1.
    
    Args:
        f: File handle to write output to
        training_set (list): List of tokens used to train the model
        validation_set (list): List of tokens used for validation
        best_lambda (float): The optimal lambda value chosen during development
        input_word (str): The input word to evaluate probabilities for
        
    Outputs:
        #Output12: MLE probability of input word (without smoothing)
        #Output13: MLE probability of unseen word (0)
        #Output14: Lidstone probability of input word with λ=0.1
        #Output15: Lidstone probability of unseen word with λ=0.1
        #Output16: Perplexity with λ=0.01
        #Output17: Perplexity with λ=0.1
        #Output18: Perplexity with λ=1.0
        #Output19: Best lambda value
        #Output20: Perplexity with best lambda
    """
    
    # Output12: MLE of input word without smoothing P_MLE(x) = (c(x)) / N
    mle_input_word = training_set.count(input_word) / len(training_set)
    f.write(f"#Output12\t{mle_input_word}\n")
    
    # Output13: MLE of unseen word based on training without smoothing P_MLE(x) = 0 / N
    mle_unseen_word = 0 / len(training_set)
    f.write(f"#Output13\t{mle_unseen_word}\n")
    
    # Output14: Lidstone model evaluation of input word (P(input_word)) given lambda=0.1
    # P_Lidstone(x) = (c(x)+λ) / (|S| + λ|X|)
    lidstone_model_evaluation_input_word = (training_set.count(input_word) + 0.1) / (len(training_set) + 0.1 * VOCABULARY_SIZE)
    f.write(f"#Output14\t{lidstone_model_evaluation_input_word}\n")
    
    # Output15: Lidstone model evaluation of unseen word (P(unseen_word)) given lambda=0.1
    # P_Lidstone(x) = λ / (|S| + λ|X|)
    lidstone_model_evaluation_unseen_word = 0.1 / (len(training_set) + 0.1 * VOCABULARY_SIZE)
    f.write(f"#Output15\t{lidstone_model_evaluation_unseen_word}\n")
    
    # Output16: The perplexity of the Lidstone model on validation set given lambda=0.01
    perplexity = evaluate_lindstone_model_preplexity(training_set, validation_set, 0.01)
    f.write(f"#Output16\t{perplexity}\n")
    
    # Output17: The perplexity of the Lidstone model on validation set given lambda=0.1
    perplexity = evaluate_lindstone_model_preplexity(training_set, validation_set, 0.1)
    f.write(f"#Output17\t{perplexity}\n")

    # Output18: The perplexity of the Lidstone model on validation set given lambda=1.0
    perplexity = evaluate_lindstone_model_preplexity(training_set, validation_set, 1)
    f.write(f"#Output18\t{perplexity}\n")
    
    # Output19: The value of best lambda to minimize perplexity on validation set
    f.write(f"#Output19\t{best_lambda}\n")
    
    # Output20: The perplexity of the Lidstone model on validation set given best lambda
    perplexity = evaluate_lindstone_model_preplexity(training_set, validation_set, best_lambda)
    f.write(f"#Output20\t{perplexity}\n")


def debug_held_out_model(f, held_out_model, training_set, held_out_set):
    """
    Debugs the held-out model by verifying that probability mass sums to 1.
    
    Checks that the sum of probabilities over all words in the held-out model equals 1.
    
    Args:
        f: File handle to write debug output to
        held_out_model (dict): Dictionary mapping words to their held-out probabilities
        training_set (list): List of tokens in training set (used to get all words)
        held_out_set (list): List of tokens in held-out set (used to get all words)
        
    Returns:
        bool: True if the probability sum is within tolerance (1e-13) of 1, False otherwise
        
    Outputs:
        #DebugHeldOut: Total probability and difference from 1.0
    """
    # held_out_model is expected to be a mapping: word -> P_heldout(word)
    all_words = set(training_set) | set(held_out_set)
    total_prob = sum(held_out_model.get(word, 0.0) for word in all_words)
    diff = abs(1.0 - total_prob)
    f.write(f"#DebugHeldOut\t{total_prob}\t{diff}\n")
    return diff < 1e-13

  
def compute_held_out_model(training_set, held_out_set):
    """
    Computes the held-out model probabilities for all words.
    
    Uses the formula: P_heldout(x) = t_r / (N_H * n_r)
    where:
    - r = c_T(x) = count of word x in training set
    - t_r = sum of c_H(y) for all words y where c_T(y) = r
    - N_H = total number of events in held-out set
    - n_r = number of distinct words that appeared exactly r times in training
    
    Args:
        training_set (list): List of words/tokens in training set
        held_out_set (list): List of words/tokens in held-out set
    
    Returns:
        tuple: (word_to_prob, freq_to_prob)
            - word_to_prob (dict): Dictionary mapping words to their held-out probabilities
            - freq_to_prob (dict): Dictionary mapping frequency r to held-out probability
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
    train_freq_to_words = {}  # r -> list of words with frequency r in training set
    train_freq_to_tr_nr = {}  # r -> [t_r - sum of c_H(y) for all words y where c_T(y) = r, n_r - number of distinct words that appeared exactly r times in training]
    
    for word in all_words:
        r = training_counts[word]
        if r not in train_freq_to_words:
            train_freq_to_words[r] = []
        train_freq_to_words[r].append(word)
        if r not in train_freq_to_tr_nr:
            train_freq_to_tr_nr[r] = [0, 0]
        train_freq_to_tr_nr[r][0] += (held_out_counts.get(word, 0.0))
        train_freq_to_tr_nr[r][1] += 1
         
    # Compute probabilities for each word and frequency
    word_to_prob = {}  # word -> probability for this word
    freq_to_prob = {}  # frequency r -> probability
    for word in all_words:
        r = training_counts[word]
        t_r, n_r = train_freq_to_tr_nr[r]
        if n_r == 0:
            prob = 0.0
        else:
            prob = t_r / (N_H * n_r)
        word_to_prob[word] = prob
        freq_to_prob[r] = prob
    
    return word_to_prob, freq_to_prob


def held_out_model_training(f, development_set, input_word):
    """
    Trains the held-out model on the development set.
    
    Splits the development set into training and held-out sets (50%/50%), computes
    held-out model probabilities, and outputs statistics about the model.
    
    Args:
        f: File handle to write output to
        development_set (list): List of tokens from the development set
        input_word (str): The input word to compute probability for
        
    Returns:
        tuple: (held_out_probabilities, training_set, held_out_set)
            - held_out_probabilities (dict): Dictionary mapping words to their held-out probabilities
            - training_set (list): List of tokens in training portion
            - held_out_set (list): List of tokens in held-out portion
            
    Outputs:
        #Output21: Number of events in the training set
        #Output22: Number of events in the held-out set
        #Output23: Probability of input_word according to held-out model
        #Output24: Probability of unseen word according to held-out model
    """
    # Split the development set into training and held-out sets 1/2 - 1/2
    training_size = int(round(0.5 * len(development_set)))
    training_set = development_set[:training_size]
    held_out_set = development_set[training_size:]
    
    # Output21: number of events in the training set
    f.write(f"#Output21\t{len(training_set)}\n")
    
    # Output22: number of events in the held-out set
    f.write(f"#Output22\t{len(held_out_set)}\n")
    
    # Compute held-out model probabilities for all words
    held_out_probabilities, freq_to_prob = compute_held_out_model(training_set, held_out_set)
    
    # Output23: p(input_word) of held-out model
    held_out_model_input_word = held_out_probabilities.get(input_word, 0.0)
    f.write(f"#Output23\t{held_out_model_input_word}\n")
    
    # Output24: p(unseen_word) of held-out model
    held_out_model_unseen_word = freq_to_prob.get(0, 0.0)
    f.write(f"#Output24\t{held_out_model_unseen_word}\n")
    
    return held_out_probabilities, training_set, held_out_set


def evaluate_held_out_model_preplexity(held_out_model, test_set):
    """
    Computes the perplexity of the held-out model on a test set.
    
    Perplexity is calculated as 2^(-average_log_probability), where the average
    log probability is computed over all events in the test set.
    
    Args:
        held_out_model (dict): Dictionary mapping words to their held-out probabilities
        test_set (list): List of words/tokens from test set to evaluate
        
    Returns:
        float: Perplexity of the test set under the held-out model
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
    Evaluates both the Lidstone model and held-out model on the test set and writes the results.
    Computes and outputs the perplexity of the test set for both models, then determines
    which model performs better. Lower perplexity indicates a better language model.
    
    Args:
        f: File handle to write output to
        training_set_lindstone (list): Training set used for the Lidstone model (list of tokens)
        held_out_model (dict): Dictionary mapping words to their held-out probabilities
        test_set (list): List of words/tokens from the test set to evaluate
        input_word (str): Input word (not used in this function but kept for consistency)
        best_lambda (float): The λ (lambda) value chosen during development for the Lidstone model
        
    Outputs:
        #Output25: Total number of events in the test set
        #Output26: Perplexity of the test set according to the Lidstone model with best_lambda
        #Output27: Perplexity of the test set according to the held-out model
        #Output28: 'L' if Lidstone model is better (lower perplexity), 'H' if held-out model is better
    """
    # Output25: total number of events in the test set
    events_count_test = len(test_set)
    f.write(f"#Output25\t{events_count_test}\n")
    
    # Output26: The perplexity of the test set according to the Lidstone model with the λ that you chose during development
    test_lind_preplexity = evaluate_lindstone_model_preplexity(training_set_lindstone, test_set, best_lambda)
    f.write(f"#Output26\t{test_lind_preplexity}\n")
    
    # Output27: The perplexity of the test set according to your held-out model
    test_held_out_preplexity = evaluate_held_out_model_preplexity(held_out_model, test_set)
    f.write(f"#Output27\t{test_held_out_preplexity}\n")
    
    # Output28: If your Lidstone model is a better language model for the test set than the held-out model then output the string ’L’, otherwise output ’H’
    if test_lind_preplexity < test_held_out_preplexity:
        f.write("#Output28\tL\n")
    else:
        f.write("#Output28\tH\n")


def main():
    """
    Main function that orchestrates the entire Exercise 2 pipeline.
    
    Processes the development and test sets, trains and evaluates both Lidstone and
    held-out models, and writes all outputs to a file.
        
    Pipeline:
        1. Initialize and write header information (Output1-Output6)
        2. Preprocess development set (Output7)
        3. Train and evaluate Lidstone model (Output8-Output20)
        4. Train held-out model (Output21-Output24)
        5. Debug held-out / Lindstone model
        6. Evaluate both models on test set (Output25-Output28)
    """
    # Parse Input
    dev_set_filename, test_set_filename, input_word, output_filename = parse_arguments()

    # Development set preprocessing
    # token_list represents sequence of events
    dev_token_list = read_and_tokenize_data(dev_set_filename)

    try:
        with open(output_filename, 'w') as f:
            # Calculate and write output section 1
            init(f, dev_set_filename, test_set_filename, input_word, output_filename)

            # Section 2: Development set preprocessing
            development_set_preprocessing(f, dev_token_list)

            # Section 3: Lidstone model training
            training_set_lindstone, validation_set = lidstone_model_training(f, token_list, input_word)            
            best_lambda = choose_best_lambda(training_set_lindstone, validation_set)           
            lidstone_model_evaluation(f, training_set_lindstone, validation_set, best_lambda, input_word)
            
            # Section 4: Held-out model training
            development_set = training_set_lindstone + validation_set
            held_out_model, training_set_heldout, held_out_set = held_out_model_training(f, development_set, input_word)
            
            # **- REMOVE COMMENT TO DEBUG -**
            # Section 5: Debug held-out model code by making sure prob sums to 1
            # debug_lindstone_model(f, training_set, validation_set, best_lambda)
            # debug_lindstone_model(f, training_set, validation_set, 0.1)
            # debug_lindstone_model(f, training_set, validation_set, 1)
            # debug_held_out_model(f, held_out_model, training_set_heldout, held_out_set)  
            
            # Section 6: Evaluation on test set
            test_set = read_and_tokenize_development_set(test_set_filename)
            evaluation_on_test_set(f, training_set_lindstone, held_out_model, test_set, input_word, best_lambda)

    except IOError as e:
        print(f"Error writing to file {output_filename}: {e}")


if __name__ == "__main__":
    main()
    # python ex2.py develop.txt test.txt honduras output.txt
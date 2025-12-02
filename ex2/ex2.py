import sys

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
        tuple: (list of tokens (token_list), total number of events)
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
    f.write("#Students\tEli Kazum\t038298857\tKeren Fuchs\t987654321\n")

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


def lidstone_model_training(f, token_list):
    """
    Implements Lidstone model training section:
    Splits the development set into training and validation sets (90%/10%)

    Args:
        f (file object): The open file handle to write to.
        token_list (list): The sequence of all events S from the development set.

    Returns:
        tuple: (training_set, validation_set)
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

    return training_set, validation_set



def main():
    """
    Main function for Exercise 2.
    """
    # Parse Input
    dev_set_filename, test_set_filename, input_word, output_filename = parse_arguments()

    # Development set preprocessing
    # token_list represents sequence of events
    token_list = read_and_tokenize_development_set(dev_set_filename)

    try:
        with open(output_filename, 'w') as f:
            # Calculate and write output section 1
            init(f, dev_set_filename, test_set_filename, input_word, output_filename)

            # Section 2: Development set preprocessing
            development_set_preprocessing(f, token_list)

            # Section 3: Lidstone model training (Output8, Output9)
            training_set, validation_set = lidstone_model_training(f, token_list)

    except IOError as e:
        print(f"Error writing to file {output_filename}: {e}")


if __name__ == "__main__":
    main()
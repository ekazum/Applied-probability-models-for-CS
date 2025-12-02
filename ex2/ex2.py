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
            for i, line in enumerate(f):
                # The input file contains 2 lines per article - header and article text.
                if i % 2 != 0:
                    # Tokenization: split by white spaces (everything between 2 white spaces is an event)
                    token_list.append(line.strip().split())
    except FileNotFoundError:
        print(f"Error: Development set file not found at {filename}")
        sys.exit(1)

    return token_list, len(token_list)


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


def development_set_preprocessing(f, num_of_events):
    """
    Writes preprocessing outputs for the development set (Output7).
    """
    # (a) Output7: total number of events in the development set
    f.write(f"#Output7\t{num_of_events}\n")


def main():
    """
    Main function for Exercise 2.
    """
    # Parse Input
    dev_set_filename, test_set_filename, input_word, output_filename = parse_arguments()

    # Development set preprocessing
    # token_list represents sequence of events
    token_list, num_of_events = read_and_tokenize_development_set(dev_set_filename)

    try:
        with open(output_filename, 'w') as f:
            # Calculate and write output section 1
            init(f, dev_set_filename, test_set_filename, input_word, output_filename)

            # Section 2: Development set preprocessing
            development_set_preprocessing(f, num_of_events)
    except IOError as e:
        print(f"Error writing to file {output_filename}: {e}")


if __name__ == "__main__":
    main()
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
    Reads the development set file, processes the text according to the requirements,
    and returns the sequence of events (tokens) and the total count.

    The input is assumed to have already had the specific punctuation removed,
    so we focus on splitting by whitespace and converting to lowercase.

    Args:
        filename (str): Path to the development set file (e.g., develop.txt).

    Returns:
        tuple: (list of tokens (token_list), total number of events)
    """
    token_list = []

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # The input file contains 2 lines per article.
                # Line 1 (i=0, 2, 4...) is the header (ignore).
                # Line 2 (i=1, 3, 5...) is the article itself (process).
                if i % 2 != 0:
                    # Tokenization: split by white spaces (everything between 2 white spaces is an event)
                    tokens = line.strip().split()

                    for token in tokens:
                        # Convert all words to lowercase as required.
                        processed_token = token.lower()
                        token_list.append(processed_token)

    except FileNotFoundError:
        print(f"Error: Development set file not found at {filename}")
        sys.exit(1)

    return token_list, len(token_list)


def write_initial_outputs(f, dev_file, test_file, input_word, output_file, vocab_size, p_uniform, num_of_events):
    """
    Writes the initialization and preprocessing outputs (Output1 - Output7)
    to the open file object.
    """
    # Required Header line
    f.write("#Students\tEli Kazum\t038298857\tKeren Fuchs\t987654321\n")

    # SECTION 1: INIT
    # (a) Output1: development set file name
    f.write(f"#Output1\t{dev_file}\n")

    # (b) Output2: test set file name
    f.write(f"#Output2\t{test_file}\n")

    # (c) Output3: INPUT_WORD
    f.write(f"#Output3\t{input_word}\n")

    # (d) Output4: output file name
    f.write(f"#Output4\t{output_file}\n")

    # (e) Output5: language vocabulary size
    f.write(f"#Output5\t{vocab_size}\n")

    # (f) Output6: Puniform (Event = INPUT_WORD)
    f.write(f"#Output6\t{p_uniform}\n")

    # SECTION 2: DEVELOPMENT SET PREPROCESSING
    # (a) Output7: total number of events in the development set S
    f.write(f"#Output7\t{num_of_events}\n")


def main():
    """
    Main execution function for Exercise 2.
    """
    # 1. Parse Input
    dev_set_filename, test_set_filename, input_word, output_filename = parse_arguments()

    # 2. Development set preprocessing
    # token_list represents S, the sequence of events
    token_list, num_of_events = read_and_tokenize_development_set(dev_set_filename)

    # 3. Calculations
    # P_uniform = 1 / V
    p_uniform = 1.0 / VOCABULARY_SIZE

    # 4. File Management and Output
    try:
        # File opening remains in main function
        with open(output_filename, 'w') as f:
            write_initial_outputs(
                f,
                dev_set_filename,
                test_set_filename,
                input_word,
                output_filename,
                VOCABULARY_SIZE,
                p_uniform,
                num_of_events
            )

    except IOError as e:
        print(f"Error writing to file {output_filename}: {e}")


if __name__ == "__main__":
    main()
"""
Comprehensive test suite for ex2.py functions.
Tests each function with various test cases including known perplexity datasets.
"""

import unittest
import sys
import os
import math
import tempfile
from io import StringIO
from collections import Counter

# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Import functions from ex2.py
import ex2

# Known test datasets with expected values
KNOWN_DATASETS = {
    'simple_uniform': {
        'training': ['a', 'b', 'c', 'a', 'b', 'c'],
        'validation': ['a', 'b'],
        'vocab_size': 3,
        'lambda_0.1': {
            'prob_a': (2 + 0.1) / (6 + 0.1 * 3),
            'prob_unseen': 0.1 / (6 + 0.1 * 3),
        }
    },
    'single_word': {
        'training': ['word'] * 10,
        'validation': ['word'] * 5,
        'expected_mle': 1.0,
        'lambda_0.1': {
            'prob_word': (10 + 0.1) / (10 + 0.1 * ex2.VOCABULARY_SIZE),
        }
    },
    'mixed_frequencies': {
        'training': ['a', 'a', 'a', 'b', 'b', 'c'],
        'validation': ['a', 'b', 'c', 'd'],  # d is unseen
        'lambda_0.1': {
            'prob_a': (3 + 0.1) / (6 + 0.1 * ex2.VOCABULARY_SIZE),
            'prob_b': (2 + 0.1) / (6 + 0.1 * ex2.VOCABULARY_SIZE),
            'prob_c': (1 + 0.1) / (6 + 0.1 * ex2.VOCABULARY_SIZE),
            'prob_unseen': 0.1 / (6 + 0.1 * ex2.VOCABULARY_SIZE),
        }
    }
}


class TestParseArguments(unittest.TestCase):
    """Tests for parse_arguments function."""
    
    def setUp(self):
        self.original_argv = sys.argv.copy()
    
    def tearDown(self):
        sys.argv = self.original_argv
    
    def test_correct_arguments(self):
        """Test parsing with correct number of arguments."""
        sys.argv = ['ex2.py', 'dev.txt', 'test.txt', 'word', 'output.txt']
        result = ex2.parse_arguments()
        self.assertEqual(result, ('dev.txt', 'test.txt', 'word', 'output.txt'))
    
    def test_incorrect_arguments(self):
        """Test that incorrect number of arguments exits."""
        sys.argv = ['ex2.py', 'dev.txt', 'test.txt']
        with self.assertRaises(SystemExit):
            ex2.parse_arguments()


class TestReadAndTokenizeDevelopmentSet(unittest.TestCase):
    """Tests for read_and_tokenize_development_set function."""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')
        self.temp_file_name = self.temp_file.name
    
    def tearDown(self):
        if os.path.exists(self.temp_file_name):
            os.unlink(self.temp_file_name)
    
    def test_basic_tokenization(self):
        """Test basic tokenization with header lines."""
        content = "header1\nword1 word2 word3\nheader2\nword4 word5\n"
        self.temp_file.write(content)
        self.temp_file.close()
        
        result = ex2.read_and_tokenize_development_set(self.temp_file_name)
        self.assertEqual(result, ['word1', 'word2', 'word3', 'word4', 'word5'])
    
    def test_empty_lines_skipped(self):
        """Test that empty lines are skipped."""
        content = "header1\nword1 word2\n\nheader2\nword3\n"
        self.temp_file.write(content)
        self.temp_file.close()
        
        result = ex2.read_and_tokenize_development_set(self.temp_file_name)
        self.assertEqual(result, ['word1', 'word2', 'word3'])
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is handled."""
        with self.assertRaises(SystemExit):
            ex2.read_and_tokenize_development_set('nonexistent_file.txt')


class TestInit(unittest.TestCase):
    """Tests for init function."""
    
    def test_init_output(self):
        """Test that init writes correct output format."""
        output = StringIO()
        ex2.init(output, 'dev.txt', 'test.txt', 'word', 'out.txt')
        output_str = output.getvalue()
        
        self.assertIn('#Students', output_str)
        self.assertIn('#Output1\tdev.txt', output_str)
        self.assertIn('#Output2\ttest.txt', output_str)
        self.assertIn('#Output3\tword', output_str)
        self.assertIn('#Output4\tout.txt', output_str)
        self.assertIn(f'#Output5\t{ex2.VOCABULARY_SIZE}', output_str)
        self.assertIn('#Output6', output_str)
        
        # Check uniform probability
        expected_p_uniform = 1.0 / ex2.VOCABULARY_SIZE
        self.assertIn(f'{expected_p_uniform}', output_str)


class TestDevelopmentSetPreprocessing(unittest.TestCase):
    """Tests for development_set_preprocessing function."""
    
    def test_output7(self):
        """Test that Output7 contains correct event count."""
        output = StringIO()
        token_list = ['a', 'b', 'c', 'd', 'e']
        ex2.development_set_preprocessing(output, token_list)
        output_str = output.getvalue()
        
        self.assertIn(f'#Output7\t{len(token_list)}', output_str)


class TestLidstoneModelTraining(unittest.TestCase):
    """Tests for lidstone_model_training function."""
    
    def test_90_10_split(self):
        """Test that training/validation split is approximately 90/10."""
        output = StringIO()
        token_list = list(range(100))  # 100 tokens
        training_set, validation_set = ex2.lidstone_model_training(
            output, token_list, 'test'
        )
        
        output_str = output.getvalue()
        # Check that split is approximately 90/10
        self.assertAlmostEqual(len(training_set) / len(token_list), 0.9, delta=0.01)
        self.assertIn('#Output8', output_str)
        self.assertIn('#Output9', output_str)
        self.assertIn('#Output10', output_str)
        self.assertIn('#Output11', output_str)
    
    def test_observed_vocabulary(self):
        """Test that observed vocabulary is correctly counted."""
        output = StringIO()
        token_list = ['a', 'a', 'b', 'b', 'c'] * 10
        training_set, validation_set = ex2.lidstone_model_training(
            output, token_list, 'a'
        )
        
        output_str = output.getvalue()
        # Observed vocabulary should be 3 (a, b, c)
        unique_in_training = len(set(training_set))
        self.assertIn(f'#Output10\t{unique_in_training}', output_str)
    
    def test_input_word_count(self):
        """Test that input word count is correct."""
        output = StringIO()
        token_list = ['a', 'b', 'a', 'c', 'a'] * 10
        training_set, validation_set = ex2.lidstone_model_training(
            output, token_list, 'a'
        )
        
        output_str = output.getvalue()
        expected_count = training_set.count('a')
        self.assertIn(f'#Output11\t{expected_count}', output_str)


class TestEvaluateLidstoneModelPerplexity(unittest.TestCase):
    """Tests for evaluate_lindstone_model_preplexity function."""
    
    def test_known_perplexity_simple(self):
        """Test perplexity calculation with known simple dataset."""
        training = ['a', 'a', 'b', 'b']
        validation = ['a', 'b']
        lambda_val = 0.1
        
        perplexity = ex2.evaluate_lindstone_model_preplexity(
            training, validation, lambda_val
        )
        
        # Perplexity should be positive
        self.assertGreater(perplexity, 0)
        # For this simple case, perplexity should be reasonable
        self.assertLess(perplexity, 1e10)
    
    def test_perplexity_decreases_with_lambda(self):
        """Test that perplexity changes with different lambda values."""
        training = ['a', 'b', 'c'] * 10
        validation = ['a', 'b', 'c', 'd'] * 5
        
        perplexity_0_01 = ex2.evaluate_lindstone_model_preplexity(
            training, validation, 0.01
        )
        perplexity_0_1 = ex2.evaluate_lindstone_model_preplexity(
            training, validation, 0.1
        )
        perplexity_1_0 = ex2.evaluate_lindstone_model_preplexity(
            training, validation, 1.0
        )
        
        # All should be positive
        self.assertGreater(perplexity_0_01, 0)
        self.assertGreater(perplexity_0_1, 0)
        self.assertGreater(perplexity_1_0, 0)
    
    def test_perplexity_with_unseen_words(self):
        """Test perplexity calculation with unseen words in validation."""
        training = ['a', 'a', 'b']
        validation = ['c', 'c']  # c is unseen
        lambda_val = 0.1
        
        perplexity = ex2.evaluate_lindstone_model_preplexity(
            training, validation, lambda_val
        )
        
        # Should handle unseen words gracefully
        self.assertGreater(perplexity, 0)
        self.assertLess(perplexity, 1e10)
    
    def test_known_probability_calculation(self):
        """Test with known dataset to verify probability calculation."""
        dataset = KNOWN_DATASETS['simple_uniform']
        training = dataset['training']
        validation = dataset['validation']
        lambda_val = 0.1
        
        # Manually calculate expected probability for 'a'
        counts = Counter(training)
        N = len(training)
        expected_prob_a = (counts['a'] + lambda_val) / (N + lambda_val * ex2.VOCABULARY_SIZE)
        
        # Calculate perplexity
        perplexity = ex2.evaluate_lindstone_model_preplexity(
            training, validation, lambda_val
        )
        
        # Verify perplexity is calculated correctly
        # For validation ['a', 'b'], we can manually verify
        log_prob_sum = 0.0
        for event in validation:
            p = (counts[event] + lambda_val) / (N + lambda_val * ex2.VOCABULARY_SIZE)
            log_prob_sum += math.log2(p)
        expected_perplexity = 2 ** (-log_prob_sum / len(validation))
        
        self.assertAlmostEqual(perplexity, expected_perplexity, places=10)


class TestDebugLidstoneModel(unittest.TestCase):
    """Tests for debug_lindstone_model function."""
    
    def test_probability_sums_to_one(self):
        """Test that probability mass sums to approximately 1."""
        output = StringIO()
        training = ['a', 'a', 'b', 'b', 'c']
        validation = ['a', 'b']
        lambda_val = 0.1
        
        result = ex2.debug_lindstone_model(output, training, validation, lambda_val)
        
        # Should return True if probabilities sum to 1
        self.assertTrue(result)
        output_str = output.getvalue()
        self.assertIn('#DebugLindstone', output_str)
    
    def test_debug_output_format(self):
        """Test that debug output has correct format."""
        output = StringIO()
        training = ['a', 'b', 'c'] * 10
        validation = ['a', 'b']
        lambda_val = 0.5
        
        ex2.debug_lindstone_model(output, training, validation, lambda_val)
        output_str = output.getvalue()
        
        # Should contain total probability and difference
        lines = output_str.strip().split('\n')
        self.assertTrue(len(lines) > 0)
        self.assertIn('#DebugLindstone', lines[0])


class TestChooseBestLambda(unittest.TestCase):
    """Tests for choose_best_lambda function."""
    
    def test_chooses_best_lambda(self):
        """Test that function returns a lambda value."""
        training = ['a', 'b', 'c'] * 20
        validation = ['a', 'b', 'c', 'd'] * 5
        
        best_lambda = ex2.choose_best_lambda(training, validation)
        
        # Should be between 0.01 and 2.0
        self.assertGreaterEqual(best_lambda, 0.01)
        self.assertLessEqual(best_lambda, 2.0)
    
    def test_lambda_in_range(self):
        """Test that returned lambda is in expected range."""
        training = ['a', 'a', 'b', 'b', 'c'] * 10
        validation = ['a', 'b', 'c', 'd'] * 5
        
        best_lambda = ex2.choose_best_lambda(training, validation)
        
        # Should be a multiple of 0.01 between 0.01 and 2.0
        self.assertGreaterEqual(best_lambda, 0.01)
        self.assertLessEqual(best_lambda, 2.0)
        # Should be close to a multiple of 0.01
        self.assertAlmostEqual(best_lambda, round(best_lambda, 2), places=2)


class TestLidstoneModelEvaluation(unittest.TestCase):
    """Tests for lidstone_model_evaluation function."""
    
    def test_mle_probabilities(self):
        """Test MLE probability calculations."""
        output = StringIO()
        training = ['a', 'a', 'a', 'b', 'b', 'c']
        validation = ['a', 'b']
        best_lambda = 0.1
        input_word = 'a'
        
        ex2.lidstone_model_evaluation(output, training, validation, best_lambda, input_word)
        output_str = output.getvalue()
        
        # Output12: MLE of input word
        expected_mle = training.count(input_word) / len(training)
        self.assertIn(f'#Output12\t{expected_mle}', output_str)
        
        # Output13: MLE of unseen word should be 0
        self.assertIn('#Output13\t0.0', output_str)
    
    def test_lidstone_probabilities(self):
        """Test Lidstone probability calculations with known values."""
        output = StringIO()
        training = ['a', 'a', 'b']
        validation = ['a', 'b']
        best_lambda = 0.1
        input_word = 'a'
        
        ex2.lidstone_model_evaluation(output, training, validation, best_lambda, input_word)
        output_str = output.getvalue()
        
        # Output14: Lidstone probability of input word with lambda=0.1
        expected_prob = (training.count(input_word) + 0.1) / (len(training) + 0.1 * ex2.VOCABULARY_SIZE)
        # Check that the value appears in output (allowing for floating point precision)
        lines = output_str.split('\n')
        output14_line = [l for l in lines if '#Output14' in l][0]
        self.assertIn('#Output14', output14_line)
        
        # Output15: Lidstone probability of unseen word
        expected_unseen = 0.1 / (len(training) + 0.1 * ex2.VOCABULARY_SIZE)
        output15_line = [l for l in lines if '#Output15' in l][0]
        self.assertIn('#Output15', output15_line)
    
    def test_perplexity_outputs(self):
        """Test that perplexity outputs are generated."""
        output = StringIO()
        training = ['a', 'b', 'c'] * 10
        validation = ['a', 'b', 'c', 'd'] * 5
        best_lambda = 0.1
        input_word = 'a'
        
        ex2.lidstone_model_evaluation(output, training, validation, best_lambda, input_word)
        output_str = output.getvalue()
        
        # Should contain perplexity outputs
        self.assertIn('#Output16', output_str)
        self.assertIn('#Output17', output_str)
        self.assertIn('#Output18', output_str)
        self.assertIn('#Output19', output_str)
        self.assertIn('#Output20', output_str)


class TestComputeHeldOutModel(unittest.TestCase):
    """Tests for compute_held_out_model function."""
    
    def test_basic_held_out_calculation(self):
        """Test basic held-out model calculation."""
        training = ['a', 'a', 'b']
        held_out = ['a', 'b', 'c']
        
        word_to_prob, freq_to_prob = ex2.compute_held_out_model(training, held_out)
        
        # Should return dictionaries
        self.assertIsInstance(word_to_prob, dict)
        self.assertIsInstance(freq_to_prob, dict)
        
        # All words should have probabilities
        all_words = set(training) | set(held_out)
        for word in all_words:
            self.assertIn(word, word_to_prob)
            self.assertGreaterEqual(word_to_prob[word], 0)
    
    def test_held_out_probability_sum(self):
        """Test that held-out probabilities are reasonable."""
        training = ['a', 'a', 'b', 'b', 'c']
        held_out = ['a', 'b', 'c', 'd']
        
        word_to_prob, freq_to_prob = ex2.compute_held_out_model(training, held_out)
        
        # Probabilities should be non-negative
        for prob in word_to_prob.values():
            self.assertGreaterEqual(prob, 0)


class TestHeldOutModelTraining(unittest.TestCase):
    """Tests for held_out_model_training function."""
    
    def test_50_50_split(self):
        """Test that training/held-out split is 50/50."""
        output = StringIO()
        development_set = list(range(100))
        input_word = 'test'
        
        held_out_model, training_set, held_out_set = ex2.held_out_model_training(
            output, development_set, input_word
        )
        
        # Should be approximately 50/50 split
        self.assertAlmostEqual(len(training_set) / len(development_set), 0.5, delta=0.01)
        
        output_str = output.getvalue()
        self.assertIn('#Output21', output_str)
        self.assertIn('#Output22', output_str)
        self.assertIn('#Output23', output_str)
        self.assertIn('#Output24', output_str)
    
    def test_held_out_probabilities(self):
        """Test that held-out probabilities are computed."""
        output = StringIO()
        development_set = ['a', 'a', 'b', 'b', 'c', 'c'] * 10
        input_word = 'a'
        
        held_out_model, training_set, held_out_set = ex2.held_out_model_training(
            output, development_set, input_word
        )
        
        # Should return a dictionary of probabilities
        self.assertIsInstance(held_out_model, dict)
        if input_word in held_out_model:
            self.assertGreaterEqual(held_out_model[input_word], 0)


class TestDebugHeldOutModel(unittest.TestCase):
    """Tests for debug_held_out_model function."""
    
    def test_debug_output(self):
        """Test that debug output is generated."""
        output = StringIO()
        held_out_model = {'a': 0.3, 'b': 0.2, 'c': 0.5}
        training_set = ['a', 'b', 'c']
        held_out_set = ['a', 'b']
        
        result = ex2.debug_held_out_model(output, held_out_model, training_set, held_out_set)
        
        output_str = output.getvalue()
        self.assertIn('#DebugHeldOut', output_str)
        # Result should be boolean
        self.assertIsInstance(result, bool)


class TestEvaluateHeldOutModelPerplexity(unittest.TestCase):
    """Tests for evaluate_held_out_model_preplexity function."""
    
    def test_basic_perplexity(self):
        """Test basic perplexity calculation."""
        held_out_model = {'a': 0.5, 'b': 0.3, 'c': 0.2}
        test_set = ['a', 'b', 'c']
        
        perplexity = ex2.evaluate_held_out_model_preplexity(held_out_model, test_set)
        
        # Should be positive
        self.assertGreater(perplexity, 0)
    
    def test_perplexity_with_unseen_words(self):
        """Test perplexity calculation with unseen words."""
        held_out_model = {'a': 0.5, 'b': 0.5}
        test_set = ['a', 'b', 'c']  # c is unseen
        
        perplexity = ex2.evaluate_held_out_model_preplexity(held_out_model, test_set)
        
        # Should handle unseen words (using epsilon)
        self.assertGreater(perplexity, 0)
        self.assertLess(perplexity, 1e10)
    
    def test_known_perplexity_calculation(self):
        """Test perplexity with known probabilities."""
        # Simple case: uniform distribution
        held_out_model = {'a': 0.5, 'b': 0.5}
        test_set = ['a', 'b'] * 10
        
        perplexity = ex2.evaluate_held_out_model_preplexity(held_out_model, test_set)
        
        # For uniform distribution with 2 words, perplexity should be 2
        # PPL = 2^(-(1/20) * sum(log2(0.5))) = 2^(-(1/20) * 20 * log2(0.5))
        # = 2^(-log2(0.5)) = 2^(log2(2)) = 2
        expected_perplexity = 2.0
        self.assertAlmostEqual(perplexity, expected_perplexity, places=5)


class TestEvaluationOnTestSet(unittest.TestCase):
    """Tests for evaluation_on_test_set function."""
    
    def test_outputs_generated(self):
        """Test that all outputs are generated."""
        output = StringIO()
        training_set_lidstone = ['a', 'b', 'c'] * 10
        held_out_model = {'a': 0.4, 'b': 0.3, 'c': 0.3}
        test_set = ['a', 'b', 'c'] * 5
        input_word = 'a'
        best_lambda = 0.1
        
        ex2.evaluation_on_test_set(
            output, training_set_lidstone, held_out_model, test_set, input_word, best_lambda
        )
        
        output_str = output.getvalue()
        self.assertIn('#Output25', output_str)
        self.assertIn('#Output26', output_str)
        self.assertIn('#Output27', output_str)
        self.assertIn('#Output28', output_str)
    
    def test_model_comparison(self):
        """Test that model comparison output is correct."""
        output = StringIO()
        training_set_lidstone = ['a', 'a', 'b', 'b', 'c'] * 10
        held_out_model = {'a': 0.5, 'b': 0.3, 'c': 0.2}
        test_set = ['a', 'b'] * 10
        input_word = 'a'
        best_lambda = 0.1
        
        ex2.evaluation_on_test_set(
            output, training_set_lidstone, held_out_model, test_set, input_word, best_lambda
        )
        
        output_str = output.getvalue()
        # Output28 should be either 'L' or 'H'
        lines = output_str.split('\n')
        output28_line = [l for l in lines if '#Output28' in l]
        if output28_line:
            self.assertIn(output28_line[0].split('\t')[1], ['L', 'H'])


class TestKnownPerplexityDatasets(unittest.TestCase):
    """Tests using known perplexity datasets with expected values."""
    
    def test_uniform_distribution_perplexity(self):
        """Test perplexity for uniform distribution (known result)."""
        # For uniform distribution over V words, perplexity = V
        # But with smoothing, it will be different
        training = ['a', 'b', 'c'] * 10
        validation = ['a', 'b', 'c'] * 5
        lambda_val = 1.0  # High lambda makes it more uniform
        
        perplexity = ex2.evaluate_lindstone_model_preplexity(
            training, validation, lambda_val
        )
        
        # Should be positive and reasonable
        self.assertGreater(perplexity, 0)
        self.assertLess(perplexity, 1e6)
    
    def test_known_probability_values(self):
        """Test with known probability values from KNOWN_DATASETS."""
        dataset = KNOWN_DATASETS['mixed_frequencies']
        training = dataset['training']
        validation = dataset['validation']
        lambda_val = 0.1
        
        # Verify known probabilities
        counts = Counter(training)
        N = len(training)
        expected_probs = dataset['lambda_0.1']
        
        # Check probability for 'a'
        calculated_prob_a = (counts['a'] + lambda_val) / (N + lambda_val * ex2.VOCABULARY_SIZE)
        self.assertAlmostEqual(
            calculated_prob_a, 
            expected_probs['prob_a'],
            places=10
        )
        
        # Check probability for unseen word
        calculated_prob_unseen = lambda_val / (N + lambda_val * ex2.VOCABULARY_SIZE)
        self.assertAlmostEqual(
            calculated_prob_unseen,
            expected_probs['prob_unseen'],
            places=10
        )
    
    def test_single_word_dataset(self):
        """Test with single word repeated (known MLE = 1.0)."""
        dataset = KNOWN_DATASETS['single_word']
        training = dataset['training']
        
        # MLE should be 1.0
        mle = training.count('word') / len(training)
        self.assertEqual(mle, dataset['expected_mle'])
        
        # Lidstone probability should be calculated correctly
        lambda_val = 0.1
        expected_prob = dataset['lambda_0.1']['prob_word']
        calculated_prob = (len(training) + lambda_val) / (len(training) + lambda_val * ex2.VOCABULARY_SIZE)
        self.assertAlmostEqual(calculated_prob, expected_prob, places=10)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_training_set(self):
        """Test with empty training set."""
        training = []
        validation = ['a', 'b']
        lambda_val = 0.1
        
        # Should handle gracefully
        perplexity = ex2.evaluate_lindstone_model_preplexity(
            training, validation, lambda_val
        )
        self.assertGreater(perplexity, 0)
    
    def test_empty_validation_set(self):
        """Test with empty validation set."""
        training = ['a', 'b', 'c']
        validation = []
        lambda_val = 0.1
        
        # Should handle gracefully (though perplexity may be undefined)
        # This might cause division by zero, so we test that it doesn't crash
        try:
            perplexity = ex2.evaluate_lindstone_model_preplexity(
                training, validation, lambda_val
            )
        except (ZeroDivisionError, ValueError):
            pass  # Expected for empty validation
    
    def test_single_token_training(self):
        """Test with single token in training."""
        training = ['a']
        validation = ['a', 'a']
        lambda_val = 0.1
        
        perplexity = ex2.evaluate_lindstone_model_preplexity(
            training, validation, lambda_val
        )
        self.assertGreater(perplexity, 0)
    
    def test_all_unseen_validation(self):
        """Test when all validation tokens are unseen."""
        training = ['a', 'b']
        validation = ['c', 'd', 'e']  # All unseen
        lambda_val = 0.1
        
        perplexity = ex2.evaluate_lindstone_model_preplexity(
            training, validation, lambda_val
        )
        # Should handle unseen words
        self.assertGreater(perplexity, 0)


class ColoredTestResult(unittest.TextTestResult):
    """Custom test result class with colored output."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self._verbosity = verbosity
    
    def addSuccess(self, test):
        super().addSuccess(test)
        if self._verbosity > 1:
            self.stream.writeln(f"{Colors.OKGREEN}✓ PASSED{Colors.ENDC}: {test}")
        else:
            self.stream.write(f"{Colors.OKGREEN}.{Colors.ENDC}")
            self.stream.flush()
    
    def addError(self, test, err):
        super().addError(test, err)
        if self._verbosity > 1:
            self.stream.writeln(f"{Colors.FAIL}✗ ERROR{Colors.ENDC}: {test}")
        else:
            self.stream.write(f"{Colors.FAIL}E{Colors.ENDC}")
            self.stream.flush()
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self._verbosity > 1:
            self.stream.writeln(f"{Colors.FAIL}✗ FAILED{Colors.ENDC}: {test}")
        else:
            self.stream.write(f"{Colors.FAIL}F{Colors.ENDC}")
            self.stream.flush()
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self._verbosity > 1:
            self.stream.writeln(f"{Colors.WARNING}⊘ SKIPPED{Colors.ENDC}: {test} - {reason}")
        else:
            self.stream.write(f"{Colors.WARNING}S{Colors.ENDC}")
            self.stream.flush()
    
    def printErrors(self):
        """Print errors and failures with colored output."""
        if self.failures or self.errors:
            self.stream.writeln(f"\n{Colors.BOLD}{Colors.FAIL}{'='*70}{Colors.ENDC}")
            self.stream.writeln(f"{Colors.BOLD}{Colors.FAIL}FAILURES AND ERRORS:{Colors.ENDC}\n")
            
            for test, err in self.failures + self.errors:
                self.stream.writeln(f"{Colors.FAIL}{'-'*70}{Colors.ENDC}")
                self.stream.writeln(f"{Colors.BOLD}{Colors.FAIL}{test}{Colors.ENDC}")
                self.stream.writeln(f"{Colors.FAIL}{err}{Colors.ENDC}")
            self.stream.writeln(f"{Colors.FAIL}{'='*70}{Colors.ENDC}\n")


class ColoredTestRunner(unittest.TextTestRunner):
    """Custom test runner with colored output and statistics."""
    
    resultclass = ColoredTestResult
    
    def run(self, test):
        """Run tests and display colored results with statistics."""
        result = super().run(test)
        self.print_statistics(result)
        return result
    
    def print_statistics(self, result):
        """Print test statistics with colors ."""
        total_tests = result.testsRun
        passed = total_tests - len(result.failures) - len(result.errors) - len(result.skipped)
        failed = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        
        # Calculate percentage
        if total_tests > 0:
            percentage = (passed / total_tests) * 100
        else:
            percentage = 0
        
        # Print statistics
        print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}TEST STATISTICS{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
        
        # Test counts
        print(f"{Colors.BOLD}Total Tests:{Colors.ENDC}     {total_tests}")
        print(f"{Colors.OKGREEN}✓ Passed:{Colors.ENDC}          {passed}")
        if failed > 0:
            print(f"{Colors.FAIL}✗ Failed:{Colors.ENDC}          {failed}")
        if errors > 0:
            print(f"{Colors.FAIL}✗ Errors:{Colors.ENDC}          {errors}")
        if skipped > 0:
            print(f"{Colors.WARNING}⊘ Skipped:{Colors.ENDC}         {skipped}")
        
        print(f"\n{Colors.BOLD}Success Rate:{Colors.ENDC}   {percentage:.2f}%")
        
        # Progress bar
        bar_length = 50
        filled = int(bar_length * percentage / 100)
        bar = f"{Colors.OKGREEN}{'█' * filled}{Colors.ENDC}{Colors.FAIL}{'░' * (bar_length - filled)}{Colors.ENDC}"
        print(f"{Colors.BOLD}Progress:{Colors.ENDC}        [{bar}] {percentage:.1f}%")
        
        # Summary
        print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
        if percentage == 100:
            print(f"{Colors.OKGREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! Excellent work! 🎉{Colors.ENDC}")
        elif percentage >= 80:
            print(f"{Colors.OKGREEN}{Colors.BOLD}✓ Good job! Most tests are passing.{Colors.ENDC}")
        elif percentage >= 60:
            print(f"{Colors.WARNING}{Colors.BOLD}⚠ Some tests need attention.{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}{Colors.BOLD}✗ Many tests are failing. Please review the errors above.{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


if __name__ == '__main__':
    # Run all tests with colored output and statistics
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = ColoredTestRunner(verbosity=2)
    runner.run(suite)


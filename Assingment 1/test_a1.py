import unittest
import a1
import numpy as np


class MyTestCase(unittest.TestCase):
    global alice, anna
    alice = a1.load_data("alice_in_wonderland.tsv", 0)
    anna = a1.load_data("anna_karenina.tsv", 2)

    global alice_pre, anna_pre
    alice_pre = []
    for sent in alice:
        alice_pre.append(a1.preprocess_sentence(sent))

    anna_pre = []
    for sent in anna:
        anna_pre.append(a1.preprocess_sentence(sent))

    def test_load_data1(self):
        expected = "Alice was beginning"
        actual = alice[0][:19]
        self.assertEqual(expected, actual)

    def test_load_data3(self):
        expected = "ALL HAPPY FAMILIES"
        actual = anna[0][:18]
        self.assertEqual(expected, actual)

    def test_preprocess_sentence1(self):
        sentence = "This is a sentence."
        expected = ["sentence"]
        actual = a1.preprocess_sentence(sentence)
        self.assertEqual(expected, actual)

    def test_preprocess_sentence2(self):
        sentence = "Please, God, let the tests pass this time!"
        expected = ['god', 'let', 'test', 'pass', 'time']
        actual = a1.preprocess_sentence(sentence)
        self.assertEqual(expected, actual)

    def test_co_occurrence_matrix1(self):
        dataset = [
            ['a', 'b', 'c', 'd'],
            ['a', 'a', 'a', 'b'],
            ['c', 'b', 'a', 'b']
        ]
        expected = (4, 4)
        actual = a1.co_occurrence_matrix(dataset).shape
        self.assertEqual(expected, actual)

    def test_co_occurrence_matrix2(self):
        dataset = [
            ['a', 'b', 'c', 'd'],
            ['a', 'a', 'a', 'b'],
            ['c', 'b', 'a', 'b']
        ]
        expected = np.array([
            [6, 6, 2, 1],
            [6, 2, 3, 1],
            [2, 3, 0, 1],
            [1, 1, 1, 0]
        ])
        actual = a1.co_occurrence_matrix(dataset)
        self.assertTrue(np.array_equal(expected, actual))

    def test_co_occurrence_matrix3(self):
        dataset = [
            ['a', 'b', 'a', 'b', 'c', 'd', 'e', 'f'],
            ['c', 'c', 'a', 'e', 'd', 'b', 'f'],
        ]
        expected = (6, 6)
        actual = a1.co_occurrence_matrix(dataset).shape
        self.assertEqual(expected, actual)

    def test_co_occurrence_matrix4(self):
        dataset = [
            ['a', 'b', 'a', 'b', 'c', 'd', 'e'],
            ['c', 'c', 'a', 'e', 'd', 'b', 'f'],
        ]
        expected = np.array([
            [2, 5, 4, 2, 2, 1],
            [5, 2, 3, 3, 2, 1],
            [4, 3, 2, 3, 3, 0],
            [2, 3, 3, 0, 2, 1],
            [2, 2, 3, 2, 0, 1],
            [1, 1, 0, 1, 1, 0]
        ])
        actual = a1.co_occurrence_matrix(dataset)
        self.assertTrue(np.array_equal(expected, actual))

    def test_ppmi_matrix1(self):
        dataset = [
            ['a', 'b', 'c', 'd'],
            ['a', 'a', 'a', 'b'],
            ['c', 'b', 'a', 'b']
        ]
        cooc_mtx = a1.co_occurrence_matrix(dataset)
        expected = np.array([
            [0., 0.26303441, 0., 0.],
            [0.26303441, 0., 0.5849625, 0.],
            [0., 0.5849625, 0., 1.],
            [0., 0., 1., 0.]
        ])
        actual = a1.ppmi_matrix(cooc_mtx)
        self.assertEqual(np.round(expected[0][0], 2), np.round(actual[0][0], 2))
        self.assertEqual(np.round(expected[0][1], 2), np.round(actual[0][1], 2))

    def test_ppmi1(self):
        dataset = [
            ['a', 'b', 'c', 'd'],
            ['a', 'a', 'a', 'b'],
            ['c', 'b', 'a', 'b']
        ]
        ppmi_mtx = a1.ppmi_matrix(a1.co_occurrence_matrix(dataset))
        expected = .58
        actual = np.round(a1.ppmi('b', 'c', ppmi_mtx, dataset), 2)
        self.assertEqual(expected, actual)

    def test_ppmi3(self):
        dataset = [
            ['a', 'b', 'a', 'b', 'c', 'd', 'e'],
            ['c', 'c', 'a', 'e', 'd', 'b', 'f'],
        ]
        ppmi_mtx = a1.ppmi_matrix(a1.co_occurrence_matrix(dataset))
        expected = .3
        actual = np.round(a1.ppmi('b', 'd', ppmi_mtx, dataset), 2)
        self.assertEqual(expected, actual)

    def test_get_word_vectors1(self):
        dataset = [
            ['a', 'b', 'c', 'd'],
            ['a', 'a', 'a', 'b'],
            ['c', 'b', 'a', 'b']
        ]
        ppmi_mtx = a1.ppmi_matrix(a1.co_occurrence_matrix(dataset))
        expected = {
            'a': np.array([0., 0.26, 0., 0.]),
            'b': np.array([0.26, 0., 0.58, 0.]),
            'c': np.array([0., 0.58, 0., 1.]),
            'd': np.array([0., 0., 1., 0.])
        }
        actual = a1.get_word_vectors(ppmi_mtx, dataset)
        for word in actual:
            actual[word] = np.round(actual[word], 2)

        self.assertEqual(expected.keys(), actual.keys())
        for exp_val, act_val in zip(expected.values(), actual.values()):
            self.assertTrue(np.array_equal(exp_val, act_val))

    def test_k_most_similar1(self):
        cooc_mtx = a1.co_occurrence_matrix(anna_pre)
        ppmi_mtx = a1.ppmi_matrix(cooc_mtx)
        word_vectors = a1.get_word_vectors(ppmi_mtx, anna_pre)
        expected = [
            'coachman',
            'german',
            'mantelpiece',
            'kitchen',
            'send'
        ]
        actual = a1.k_most_similar('day', 5, word_vectors)

        self.assertEqual(expected, actual)

    def test_k_most_similar2(self):
        cooc_mtx = a1.co_occurrence_matrix(anna_pre)
        ppmi_mtx = a1.ppmi_matrix(cooc_mtx, smoothing=5)
        word_vectors = a1.get_word_vectors(ppmi_mtx, anna_pre)
        expected = [
            'know',
            'think',
            'time',
            'come',
            'levin'
        ]
        actual = a1.k_most_similar('day', 5, word_vectors)

        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()

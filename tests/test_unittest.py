import unittest
import logging
import sys

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(
            format='(In %(module)s) %(message)s',
            stream=sys.stdout,
        )
        logging.root.setLevel(logging.DEBUG)
        logging.debug('set up')

    def tearDown(self):
        print('tear down')

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()

import unittest
import dose
import amorphization as GAN
import numpy as np
from utils import write_file, backup_file
import os

class TestFunctions(unittest.TestCase):
    def test_locate_value(self):
        index1 = GAN.locate_value(4.8, np.array([1,2,3,3,3,3,3,4,5,6,7,8]))
        self.assertEqual(index1, 8)
        
        index2 = GAN.locate_value(4, np.array([1,2,3,3,3,3,3,4,5,6,7,8]))
        self.assertEqual(index2, 8)

        index3 = GAN.locate_value(3, np.array([1,2,3,3,3,3,3,4,5,6,7,8]))
        self.assertEqual(index3, 7)

        index3 = GAN.locate_value(2.9999999, np.array([1,2,3,3,3,3,3,4,5,6,7,8]))
        self.assertEqual(index3, 2)

class TestFileManipulations(unittest.TestCase):
    def test_write_file(self):
        a = [1, 2, 3]
        b = ['a', 'b', 'c']
        c = ["tttmg", "gbtt", "sqy"]
        file_name = "test_write_file.txt"
        write_file(file_name, a, b, c)

        expected_content = "1\ta\ttttmg\n2\tb\tgbtt\n3\tc\tsqy\n"

        with open(file_name) as file:
            result = file.read()
        
        os.remove(file_name)
        self.assertTrue(result == expected_content)


if __name__ == '__main__':
    unittest.main()


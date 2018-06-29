import unittest
from DBM.test.ManagerTest import PreProcessManagerTest


def suite():
    # テストスイートを定義します
    test_suite = unittest.TestSuite()
    # addTestを用いてテストスイートに追加していきます
    test_suite.addTest(unittest.makeSuite(PreProcessManagerTest.PreProcessManagerTest))
    return test_suite

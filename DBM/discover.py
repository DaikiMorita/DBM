import unittest


def suite():
    """
    creates test suite
    :return:
    """
    test_suite = unittest.TestSuite()
    # tries all test python files which match the specified pattern as a parameter
    all_test_suite = unittest.defaultTestLoader.discover("DBM.test", pattern="test_*.py")
    for ts in all_test_suite:
        test_suite.addTest(ts)
    return test_suite

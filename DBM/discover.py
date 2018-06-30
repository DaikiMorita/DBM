import unittest


# テストスイートを作成します
def suite():
    # 前述と同じく、TestSuiteから空っぽのテストスイートを作成します
    test_suite = unittest.TestSuite()
    # discoverメソッドを用いて、testディレクトリ以下からテストクラスを見つけます
    all_test_suite = unittest.defaultTestLoader.discover("DBM.test", pattern="test_*.py")
    # 見つけたテストクラスをテストスイートに追加します
    for ts in all_test_suite:
        test_suite.addTest(ts)
    return test_suite

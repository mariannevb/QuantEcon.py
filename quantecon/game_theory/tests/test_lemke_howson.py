"""
Author: Daisuke Oyama

Tests for lemke_howson.py

"""
from numpy.testing import assert_allclose
from quantecon.game_theory import NormalFormGame, lemke_howson


class TestLemkeHowson():
    def setUp(self):
        self.game_dicts = []

        # From von Stengel 2007 in Algorithmic Game Theory
        bimatrix = [[(3, 3), (3, 2)],
                    [(2, 2), (5, 6)],
                    [(0, 3), (6, 1)]]
        NEs_dict = {0: ([1, 0, 0], [1, 0]),
                    1: ([0, 1/3, 2/3], [1/3, 2/3]),}
        d = {'g': NormalFormGame(bimatrix),
             'NEs_dict': NEs_dict}
        self.game_dicts.append(d)

    def test_lemke_howson(self):
        for d in self.game_dicts:
            for k in d['NEs_dict'].keys():
                NE_computed = lemke_howson(d['g'], init_pivot=k)
                for action_computed, action in zip(NE_computed,
                                                   d['NEs_dict'][k]):
                    assert_allclose(action_computed, action)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)

import os, pytest

from mercury.dataschema.anonymize import Anonymize


def test_anonymize():
    os.environ['MERCURY_ANONYMIZE_DATASCHEMA_KEY'] = 'Mickey Mouse'

    am1 = Anonymize()
    am2 = Anonymize(6*6)

    assert am1.hash_key == am2.hash_key

    del os.environ['MERCURY_ANONYMIZE_DATASCHEMA_KEY']

    an1 = Anonymize(0)
    an2 = Anonymize(20*6)
    an3 = Anonymize(0, True)

    assert an1.hash_key == an2.hash_key and an1.hash_key == an3.hash_key and an1.hash_key != am1.hash_key

    pl = ['a', 'little', 'bit', 'of text.', 'a', 'ittle', 'bit', 'more.', 'A']

    cp_am1 = am1.anonymize_list(pl)

    assert [len(s) for s in cp_am1] == [16 for _ in range(9)]
    assert cp_am1[0] == cp_am1[4] and cp_am1[1] != cp_am1[5] and cp_am1[2] == cp_am1[6] and cp_am1[0] != cp_am1[8]

    cp_am2 = am2.anonymize_list(pl)

    assert [len(s) for s in cp_am2] == [6 for _ in range(9)]
    assert cp_am2[0] == cp_am2[4] and cp_am2[1] != cp_am2[5] and cp_am2[2] == cp_am2[6] and cp_am2[0] != cp_am2[8]
    assert [s.startswith(t) for s, t in zip(cp_am1, cp_am2)] == [True for _ in range(9)]

    cp_an1 = an1.anonymize_list(pl)

    assert [len(s) > 16 for s in cp_an1] == [True for _ in range(9)]
    assert cp_an1[0] == cp_an1[4] and cp_an1[1] != cp_an1[5] and cp_an1[2] == cp_an1[6] and cp_an1[0] != cp_an1[8]

    cp_an2 = an2.anonymize_list(pl)

    assert [len(s) for s in cp_an2] == [20 for _ in range(9)]
    assert cp_an2[0] == cp_an2[4] and cp_an2[1] != cp_an2[5] and cp_an2[2] == cp_an2[6] and cp_an2[0] != cp_an2[8]

    cp_an3 = an3.anonymize_list(pl)

    assert [len(s) > 16 for s in cp_an3] == [True for _ in range(9)]
    assert cp_an3[0] != cp_an3[4] and cp_an3[1] != cp_an3[5] and cp_an3[2] != cp_an3[6] and cp_an3[0] != cp_an3[8]
    assert [len(t) - len(s) for s, t in zip(cp_an1, cp_an3)] == [16 for _ in range(9)]

    with pytest.raises(ValueError):
        pl = am1.deanonymize_list(cp_am1)

    with pytest.raises(ValueError):
        pl = am2.deanonymize_list(cp_am2)

    with pytest.raises(ValueError):
        pl = an2.deanonymize_list(cp_an2)

    pl1 = an1.deanonymize_list(cp_an1)

    assert pl1 == pl

    pl3 = an3.deanonymize_list(cp_an3)

    assert pl3 == pl

    bm1 = Anonymize()

    assert am1.hash_key != bm1.hash_key

    cp_bm1 = bm1.anonymize_list(pl)

    assert cp_am1 != cp_bm1

    bm1.set_key('Mickey Mouse')

    assert am1.hash_key == bm1.hash_key

    cp_bm1 = bm1.anonymize_list(pl)

    assert cp_am1 == cp_bm1

import pytest
import venv.uk.gov.ons.src.baseMethod as baseMethod



def test_NullParams():
    #Test that null values cause exception

    with pytest.raises(Exception):
        bob = None
        boris = baseMethod.baseMethod(bob)

def test_NonNullParams():
    #test that if passed parameters that are not null, will not raise exception
    bob = "Mike"
    boris = baseMethod.baseMethod(bob)
    assert(type(boris) == baseMethod.baseMethod)



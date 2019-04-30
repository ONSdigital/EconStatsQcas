import pytest
import venv.uk.gov.ons.src.baseMethod as baseMethod



def test_BaseMethod():
    #Test that null values cause exception

    with pytest.raises(Exception):
        bob = None
        boris = baseMethod.baseMethod(bob)




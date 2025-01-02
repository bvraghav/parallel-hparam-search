from phs.example import add_one
import random

def test_add_one() :
  n = random.random()
  assert (1+n == add_one(n))

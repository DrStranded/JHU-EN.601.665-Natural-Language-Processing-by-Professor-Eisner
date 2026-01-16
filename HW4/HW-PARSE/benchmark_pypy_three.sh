#!/bin/bash

# Create test file with first 3 sentences
head -3 wallstreet.sen > wallstreet_three.sen

echo "========================================="
echo "Testing FIRST 3 sentences with PyPy3"
echo "========================================="

echo ""
echo "parse.py (original):"
time pypy3 parse.py wallstreet.gr wallstreet_three.sen > /dev/null 2>&1

echo ""
echo "parse2.py (E.1 + E.5 optimizations):"
time pypy3 parse2.py wallstreet.gr wallstreet_three.sen > /dev/null 2>&1

echo ""
echo "Cleaning up..."
rm wallstreet_three.sen
echo "Done!"

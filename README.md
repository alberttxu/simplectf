# simplectf

simplectf is a python program that estimates Contrast Transfer Function (CTF) parameters from aligned mrc files.

## Install
    $ python3 -m pip install simplectf

## Example
    $ simplectf --pixelsize 0.87 --voltage 200 --cs 2.7 image.mrc
    creating ctf model
    searching for optimal values
    preprocessing time: 1.43
    search time: 3.32
    total time: 4.75

    defocus values: 1.0806, 1.0701 microns
    astigmatism angle: -0.00 degrees
    phase shift: 0.00 degrees
    cross correlation: 0.141275
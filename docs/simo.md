# Reproducing SIMO-BSRNN

TAC module already tested. We have observed that the proposed TanH does not really work, but prelu is better.
The use of masker context does not work, yields very poor performance. To be improved.

The joint bandsplit scheme, still with source-specific models, and using a joint encoder / not sub last target :

|                             | vocals |  bass  |  drums |  other | average|
|-----------------------------|--------|--------|--------|--------|--------|
|  base                       |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|  joint refined bandsplit    |   8.6  |   6.7  |   9.5  |   5.0  |   7.4  |
|  simo, subtract last target |   8.1  |   6.7  |   8.7  |   5.5  |   7.3  |
|  simo, one model per source |   8.2  |   6.7  |   8.9  |   5.6  |   7.4  |


|  simo, opt                  |  11.26 |   9.76 | 11.76  |  8.38  |   10.29  |

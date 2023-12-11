# 3D render prototype

```sh
$ python3 3drender.py
```

This script was written for a Linear Algebra evaluation, part of the Data Science and
Artificial Intelligence graduation course on FGV - EMAp. There were 10 possible
themes to choose for this work, and we (me and github.com/paulocgr) have chosen
polygon rendering.

Three-dimensional rendering is a very interesting topic. There are a lot of algorithms
on that subject, that goes from basic things like rotating figures to advanced ones
like simulation of precise lighting. In this script, we implemented some of the most
basic algorithms that turns the 3d rendering possible, most of them that are
linear transformations. They are strongly tied to linear algebra, since most
of them depends on vectors and matrices (some advanced ones not implemented here
depends even on eigenvalues and eigenvectors!).

Some implemented features were **scale**, **translation** and **rotation** for
shapes, **perspective** vision to turn the render scene more simillar to real
human vision, **backface culling** to reduce the amount of rendered figures and so on.

Since the script was written for an university work in Python language, the
implementations aren't so economic and fast as they could be, so this isn't meant
to be used on any big project. This is to exemplify the importance of linear algebra
on the computer 3d rendering.

## Libraries

To create the script, only `numpy` (responsible for efficient implementations of
linear algebra concepts) and `pygame` (responsible for implementing line drawing
functions) were used.

## Controls

To movement in the world space, the **W, A, S, D, up and down** keys can be
used for translation, **N, M, J, K, left and right** keys can be used for rotating
the camera along three different axis and **U and I** keys can be used for changing
the distance of the near plane used for perspective (a.k.a changing FOV)

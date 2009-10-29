# TODO - redundant file, and functionality, as we use scipy!

import math

def dotproduct(p, q):
	return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]

def norm(p):
	sum = 0
	for i in p:
		sum += i ** 2
	return math.sqrt(sum)

def crossproduct(p, q):
	return (p[1] * q[2] - p[2] * q[1],
		    p[2] * q[0] - p[0] * q[2],
		    p[0] * q[1] - p[1] * q[0])


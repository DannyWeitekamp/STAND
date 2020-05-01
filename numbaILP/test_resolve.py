
import numba
from numba import njit, jitclass
from numba import b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.pycc import CC
from compile_template import compile_template

@jitclass([('x', i4),('y', i4)])
class Point(object):
	def __init__(self,x,y):
		self.x = x
		self.y = y

@njit(nogil=True,fastmath=True,cache=True)
def assign_point(p,x,y):
	p.x =x
	p.y =y

@njit(nogil=True,fastmath=True,cache=True)
def noop(x):
	return x*2

@njit(nogil=True,fastmath=True,cache=True)
def poop(x):
	p = Point(x,x*2)
	return p.x+p.y

@njit(nogil=True,fastmath=True)
def my_f(a,b,c):
	return a+b(a)+c(a)

cc = CC('my_module')

@cc.export('pointMaker',Point.class_type.instance_type(f4,f4)) 
@njit(nogil=True,fastmath=True) 
def pointMaker(x,y):
	return Point(x,y)

#'@njit(nogil=True,fastmath=True,cache=True) \n' + \
#"@cc.export('{}', '{}') \n" + \


njit_my_f_noop_poop = compile_template(my_f,{'b' : noop,'c':poop},cc,'f4(f4)')
print(njit_my_f_noop_poop(2))

cc.compile()

from my_module  import my_f_noop_poop
from my_module  import pointMaker as pM

print(my_f_noop_poop(2))
point = pM(1,2)
# print(point.toString())
assign_point(point,2,3)
print(point.x,point.y)

print(cc.__dict__)
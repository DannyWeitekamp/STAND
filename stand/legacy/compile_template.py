import numba
from numba import njit#, jitclass
from numba import b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.pycc import CC
from numba.core.dispatcher import Dispatcher
from types import FunctionType

def to_str(x):
	if(isinstance(x,(FunctionType,Dispatcher))):
		return x.__code__.co_name
	else:
		return str(x)

def compile_template(f,template_vals,cc_module=None,fsig="",_globals=None):
	exec_code = '@njit(nogil=True,fastmath=True) \n' + \
				'def {}({}): \n' + \
				'	return {}({}) \n' + \
				'out_func = {}'

				
	allv = f.__code__.co_varnames[:f.__code__.co_argcount]
	n = f.__code__.co_name
	# print(allv)
	tv = template_vals

	in_args = [x for x in allv if x not in template_vals]
	f_name = n+"_"+"_".join([to_str(x) for x in template_vals.values()])
	print(f_name)
	if(cc_module is not None):
		aot_header = "@cc.export('{}', {}) \n" 
		aot_header = aot_header.format(f_name, fsig)
		exec_code = aot_header + exec_code

	exec_code = exec_code.format( f_name, ", ".join(in_args),n,
		", ".join(in_args + list(template_vals.keys())), f_name)
		
	l = {}
	if(_globals is None): _globals = globals()
	_globals = _globals.copy()
	_globals.update(template_vals)
	_globals.update({n:f})
	if(cc_module is not None):
		 _globals.update({'cc' : cc_module})
	# print(_globals.keys())
	# print(exec_code)
	exec(exec_code,_globals,l)
	return l['out_func']

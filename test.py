from numpy import*
from scipy import optimize 
n=100
def f(x):
         f = zeros([n])
         for i in arange(0,n-1,1):
                  f[i] = (3 + 2*x[i])*x[i] - x[i-1] - 2*x[i+1] - 2
         f [0] = (3 + 2*x[0] )*x[0] - 2*x[1] - 3
         f[n-1] = (3 + 2*x[n-1] )*x[n-1] - x[n-2] - 4
         return f
x0 =zeros([n])
#sol = optimize.root(f,x0, method='krylov')
#print('Solution:\n', sol.x)
print(f(1))
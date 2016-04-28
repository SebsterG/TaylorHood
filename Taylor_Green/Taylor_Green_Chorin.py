from dolfin import *
from mshr import *
set_log_active(False)
N = 64
#mesh = UnitSquareMesh(N,N)
domain = Rectangle(dolfin.Point(-1., -1.), dolfin.Point(1., 1.))
mesh  = generate_mesh(domain,N)
#plot(mesh,interactive=True)
"""class PeriodicBoundaryX(SubDomain):
	def inside(self,x,on_boundary):
		return x[0] < (-1.0 + DOLFIN_EPS) and\
				x[0] >(-1.0 - DOLFIN_EPS) and \
				on_boundary
	def map(self,x,y):
		y[0] = x[0] - 2.0
		y[1] = x[1]
class PeriodicBoundaryY(SubDomain):
	def inside(self,x,on_boundary):
		return x[1] < (-1.0 + DOLFIN_EPS) and\
				x[1] >(-1.0 - DOLFIN_EPS) and \
				on_boundary
	def map(self,x,y):
		y[0] = x[0] - 2.0
		y[1] = x[1]"""
class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], -1) or near(x[1], -1)) and
                (not ((near(x[0], -1) and near(x[1], 1)) or
                        (near(x[0], 1) and near(x[1], -1)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 2.
            y[1] = x[1] - 2.
        elif near(x[0], 1):
            y[0] = x[0] - 2.
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 2.
V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain=PeriodicBoundary())
Q = FunctionSpace(mesh,"CG", 1, constrained_domain=PeriodicBoundary())
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

"""PB = PeriodicBoundary()
periodicBoundaryX = PeriodicBoundaryX()
periodicBoundaryY = PeriodicBoundaryY()
bound = FacetFunction("size_t", mesh)
bound.set_all(0)
#periodicBoundaryX.mark(bound, 1)
#periodicBoundaryY.mark(bound, 2)
plot(bound,interactive=True)"""




nu = 0.1
nu = Constant(nu)
u_0=Expression(('-sin(pi*x[1])*cos(pi*x[0])*exp(-2.*pi*pi*nu*t)'\
,'sin(pi*x[0])*cos(pi*x[1])*exp(-2.*pi*pi*nu*t)'),nu=nu,t=0.0)
p_0=Expression('-(cos(2*pi*x[0])+cos(2*pi*x[1]))*exp(-4.*pi*pi*nu*t)/4.',nu=nu,t=0.0)
u0 = interpolate(Expression(('-sin(pi*x[1])*cos(pi*x[0])','sin(pi*x[0])*cos(pi*x[1])')),V)

plot(u0,interactive=True)
u1 = Function(V)
p1 = Function(Q)
#p1=interpolate(Expression('-(cos(2*pi*x[0])+cos(2*pi*x[1]))*exp(-4.*pi*pi*nu*t)/4.',nu=nu,t=0),Q)
#bc1 = DirichletBC(V, u_0, periodicBoundaryX)
#bc2 = DirichletBC(Q, p_0, periodicBoundaryX)
bcs=[]
bcp=[]

dt = 0.01

k = Constant(dt)
f = Constant((0, 0))
# first without Pressure
#F1 = (1/k)*inner(u-u0,v)*dx + inner(dot(grad(u0),u0),v)*dx + nu*inner(grad(u),grad(v))*dx - inner(f, v)*dx
F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx +  nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# correction with Pressure
a2 = -k*inner(grad(p),grad(q))*dx
L2 = div(u1)* q *dx

# last step

a3 = inner(u,v)*dx
L3 = inner(u1,v)*dx - k*inner(grad(p1),v)*dx


ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

T = 10.0
t = dt
counter = 0
while t < T + DOLFIN_EPS:
	# Update pressure boundary condition
	#v_theta.t = t
	solve(a1==L1,u1,bcs)#, "gmres", "default")

	#pressure correction
	solve(a2==L2,p1,bcp)#, "gmres", "default")
	#print norm(p1)

	#last step
	solve(a3==L3,u1,bcs)#, "gmres", "default")

	print "Timestep: ", t
	if (counter%10)==0:
		u_0.t = t
		p_0.t = t
		p_e = project(p_0,Q)
		u_e = project(u_0,V)
		print "u-error",errornorm(u_e,u1,norm_type="l2")
		print "p-error",errornorm(p_e,p1,norm_type="l2")

		#plot(u1,rescale=False)
		ufile << u1
		pfile << p1
	#plot(p1,rescale=True)
	u0.assign(u1)
	counter+=1
	t += dt

#u_0.t = t
#print "error",errornorm(u_e,u1,norm_type="l2")
#interactive()
